from typing import List, Dict, Optional, Any, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.data_utils import dequantize_verts

import MinkowskiEngine as ME
from .mink_resnet_in import ResNetOur
from .vertex_model import VertexModel
from warmup_scheduler import GradualWarmupScheduler

class PositionEmbeddingSine3D(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, voxel_dim, num_pos_feats=32, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale
        self.voxel_dim = voxel_dim
        assert(voxel_dim > 0)
    
    def forward(self, voxel_coord):
        if self.normalize:
            voxel_coord = self.scale * voxel_coord / (self.voxel_dim - 1)

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=voxel_coord.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
        
        pos = voxel_coord[:, :, None] / dim_t
        pos_x = pos[:, 0]
        pos_y = pos[:, 1]
        pos_z = pos[:, 2]#in shape[n, pos_feature_dim]
        pos_x = torch.stack((pos_x[:, 0::2].sin(), pos_x[:, 1::2].cos()), dim=2).flatten(1)
        pos_y = torch.stack((pos_y[:, 0::2].sin(), pos_y[:, 1::2].cos()), dim=2).flatten(1)
        pos_z = torch.stack((pos_z[:, 0::2].sin(), pos_z[:, 1::2].cos()), dim=2).flatten(1)
        pos = torch.cat((pos_x, pos_y, pos_z), dim=1)#.permute(0, 3, 1, 2)
        return pos

class Sparse_Backbone_Minkowski(ME.MinkowskiNetwork):
  def __init__(self):
    super(Sparse_Backbone_Minkowski, self).__init__(3) #dimension = 3
    self.sparseModel = ResNetOur(in_channels=4, out_channels=64*6, D=3,flag_expand=False) #position, normal, 1
    
  def forward(self,x):
    input = ME.SparseTensor(features=x[1], coordinates=x[0])
    #print(input.F)
    #print(input.C)
    out=self.sparseModel(input)
    return out


class PointCloudToVertexModel(VertexModel):
    def __init__(
        self,
        decoder_config: Dict[str, Any],
        quantization_bits: int,
        device: str,
        use_discrete_embeddings: bool = True,
        max_num_input_verts: int = 2500,
        learning_rate: float = 3e-4,
        step_size: int = 5000,
        gamma: float = 0.9995,
    ) -> None:
        """Initializes the resnet module along with an embedder

        Args:
            decoder_config: Dictionary with TransformerDecoder config. Decoder config has to include num_layers, hidden_size, and fc_size.
            quantization_bits: Number of quantization bits used in mesh preprocessing
            use_discrete_embeddings: Whether or not we're working with quantized vertices
            max_num_input_verts: Maximum number of vertices. Used for learned position embeddings.
            learning_rate: Learning rate for adam optimizer
            step_size: How often to use lr scheduler
            gamma: Decay rate for lr scheduler
        """
        super(PointCloudToVertexModel, self).__init__(
            decoder_config=decoder_config,
            quantization_bits=quantization_bits,
            max_num_input_verts=max_num_input_verts,
            use_discrete_embeddings=use_discrete_embeddings,
            learning_rate=learning_rate,
            step_size=step_size,
            gamma=gamma,
            device=device
        )
        self.emb_net = Sparse_Backbone_Minkowski()
        # for param in self.emb_net.parameters():
        #     param.requires_grad = False
        self.embedder = nn.Linear(2, self.embedding_dim)
        self.position_encoding = PositionEmbeddingSine3D(32, 128, normalize=True)

    def _prepare_context(self, context: Dict[str, torch.Tensor]) -> Tuple[None, torch.Tensor]:
        """Creates image embeddings using resnet and flattened image

        Args:
            context: A dictionary that contains an image

        Returns:
            sequential_context_embeddings: Processed image embeddings
        """
        coords, feats = context["pc_coords"], context["pc_feats"]
        output = self.emb_net([coords, feats])
        sparse_locations = output.C
        sparse_features = output.F

        #Padding voxel features and corner points
        batch_idx = sparse_locations[:,0]#which sample each voxel belongs to
        sparse_locations = sparse_locations[:,1:] // output.tensor_stride[0]
        input_padding_mask = torch.zeros_like(batch_idx, dtype=torch.bool, device=sparse_features.device)

        batch_size = len(context["vertices_flat"])
        batch_number_samples = []
        for i in range(batch_size):
          batch_number_samples.append((batch_idx == i).sum())
        pad_dim = max(batch_number_samples)

        voxel_pos_embedding = self.position_encoding(sparse_locations)
        voxel_feature = torch.split(sparse_features, batch_number_samples)
        voxel_pos_embedding = torch.split(voxel_pos_embedding, batch_number_samples)
        input_padding_mask = torch.split(input_padding_mask, batch_number_samples)

        batch_voxel_feature = []
        input_padding_mask_list = []
        position_embedding_list = []
        for i in range(batch_size):
            batch_voxel_feature.append(nn.functional.pad(voxel_feature[i], (0, 0, 0, pad_dim - voxel_feature[i].shape[0])))
            input_padding_mask_list.append(nn.functional.pad(input_padding_mask[i], (0, pad_dim - voxel_feature[i].shape[0]), value=True))
            position_embedding_list.append(nn.functional.pad(voxel_pos_embedding[i], (0, 0, 0, pad_dim - voxel_feature[i].shape[0])))

        voxel_features = torch.stack(batch_voxel_feature, dim=1)
        voxel_features_padding_mask = torch.stack(input_padding_mask_list, dim=0)
        voxel_position_encoding = torch.stack(position_embedding_list, dim=1)

        sequential_context_embeddings = (voxel_features+voxel_position_encoding).permute(1,0,2)
        # sequential_context_embeddings = self.dim_converter(sequential_context_embeddings)

        return None, sequential_context_embeddings

    def configure_optimizers(self, total_steps:int, warmup_steps:int) -> Dict[str, Any]:
        """Configure optimizer and learning rate scheduler

        Returns:
            dict: Dictionary with optimizer and lr scheduler
        """
        pc_vertex_model_optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        pc_vertex_model_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(pc_vertex_model_optimizer, T_max=total_steps, eta_min=0, last_epoch=-1)
        pc_vertex_model_scheduler = GradualWarmupScheduler(pc_vertex_model_optimizer, multiplier=1, total_epoch=warmup_steps, after_scheduler=pc_vertex_model_scheduler)
        return {"optimizer": pc_vertex_model_optimizer, "scheduler_warmup": pc_vertex_model_scheduler}
