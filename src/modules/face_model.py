from typing import Dict, Optional, Tuple, Any
import math
import pdb
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from scipy.spatial import ConvexHull

from src.utils.data_utils import quantize_verts

from .polygen_encoder import PolygenEncoder
from .polygen_decoder import TransformerDecoder
from .utils import top_k_logits, top_p_logits
from warmup_scheduler import GradualWarmupScheduler


class FaceModel(nn.Module):
    def __init__(
        self,
        encoder_config: Dict[str, Any],
        decoder_config: Dict[str, Any],
        device: str,
        class_conditional: bool = True,
        num_classes: int = 55,
        decoder_cross_attention: bool = True,
        use_discrete_vertex_embeddings: bool = True,
        quantization_bits: int = 8,
        max_seq_length: int = 5000,
        learning_rate: float = 3e-4,
        step_size: int = 5000,
        gamma: float = 0.9995,
    ) -> None:
        """Autoregressive generative model of face vertices

        Args:
            encoder_config: Dictionary representing config for PolygenEncoder
            decoder_config: Dictionary representing config for TransformerDecoder
            class_conditional: If we are using global context embeddings based on class labels
            num_classes: How many distinct classes in the dataset
            decoder_cross_attention: If we are using cross attention within the decoder
            use_discrete_vertex_embeddings: Are the inputted vertices quantized
            quantization_bits: How many bits are we using to encode the vertices
            max_seq_length: Max number of face indices we can generate
            learning_rate: Learning rate for adam optimizer
            step_size: How often to use lr scheduler
            gamma: Decay rate for lr scheduler
        """
        super(FaceModel, self).__init__()
        self.device = device
        self.encoder_config = encoder_config
        self.decoder_config = decoder_config
        self.embedding_dim = decoder_config["hidden_size"]
        self.class_conditional = class_conditional
        self.num_classes = num_classes
        self.max_seq_length = max_seq_length
        self.decoder_cross_attention = decoder_cross_attention
        self.use_discrete_vertex_embeddings = use_discrete_vertex_embeddings
        self.quantization_bits = quantization_bits

        self.encoder = PolygenEncoder(**encoder_config)
        self.decoder = TransformerDecoder(device=device, **decoder_config)

        self.class_embedder = nn.Embedding(self.num_classes, embedding_dim=self.embedding_dim)
        self.coord0_embedder = nn.Embedding(2 ** self.quantization_bits, self.embedding_dim)
        self.coord1_embedder = nn.Embedding(2 ** self.quantization_bits, self.embedding_dim)
        self.coord2_embedder = nn.Embedding(2 ** self.quantization_bits, self.embedding_dim)

        self.pos_embedder = nn.Embedding(self.max_seq_length, self.embedding_dim)

        self.linear_layer = nn.Linear(self.embedding_dim, self.embedding_dim)
        stopping_embeddings_tensor = torch.randn([1, 2, self.embedding_dim], device=self.device)
        self.stopping_embeddings = nn.Parameter(stopping_embeddings_tensor)
        zero_embeddings_tensor = torch.randn([1, 1, self.embedding_dim], device=self.device)
        self.zero_embed = nn.Parameter(zero_embeddings_tensor)

        self.learning_rate = learning_rate
        self.step_size = step_size
        self.gamma = gamma

    def _embed_class_label(self, labels: torch.Tensor) -> torch.Tensor:
        """Embeds class labels if class_conditional is true

        Args:
            labels: A tensor of shape [batch_size,] that represents all class labels

        Returns:
            global_context_embeddings: A tensor of shape [batch_size, embed_size]
        """
        return self.class_embedder(labels.to(torch.int64))

    def _prepare_context(
        self, context: Dict[str, Any]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Prepares vertex, global context and sequential context embeddings

        Args:
            context: A dictionary with keys of class_label (if class conditional is true), vertices and vertices_mask

        Returns:
            vertex_embeddings: Value embeddings for vertices of shape [batch_size, num_vertices + 2, embed_size]
            global_context_embedding: Embeddings for class label of shape [batch_size, embed_size]
            sequential_context_embeddings: Result of applying vertex mask to vertex embeddings of shape [batch_size, num_vertices + 2, embed_size]
        """
        if self.class_conditional:
            global_context_embedding = self._embed_class_label(context["class_label"])
        else:
            global_context_embedding = None

        vertex_embeddings = self._embed_vertices(context["vertices"], context["vertices_mask"])
        if self.decoder_cross_attention:
            sequential_context_embeddings = (
                vertex_embeddings * F.pad(context["vertices_mask"], [2, 0, 0, 0], value=1)[..., None]
            )
        else:
            sequential_context_embeddings = None
        return (
            vertex_embeddings,
            global_context_embedding,
            sequential_context_embeddings,
        )

    def _embed_vertices(self, vertices: torch.Tensor, vertices_mask: torch.Tensor) -> torch.Tensor:
        """Provides value embeddings for vertices

        Args:
            vertices: A tensor of shape [batch_size, num_vertices, 3]. Represents vertices in the generated mesh.
            vertices_mask: A tensor of shape [batch_size, num_vertices]. Provides information about which vertices are complete.

        Returns:
            vertex_embeddings: A tensor of shape [batch_size, num_vertices + 2, embed_size]. Represents vertex embeddings with concatenated learnt stopping tokens.
        """
        if self.use_discrete_vertex_embeddings:
            vertex_embeddings = 0.0
            verts_quantized = quantize_verts(vertices, self.quantization_bits).to(torch.long)
            vertex_embeddings = (
                self.coord0_embedder(verts_quantized[..., 0])
                + self.coord1_embedder(verts_quantized[..., 1])
                + self.coord2_embedder(verts_quantized[..., 2])
            )

        else:
            raise Exception("Support for continuous vertex embeddings doesn't exist yet")

        vertex_embeddings = vertex_embeddings * vertices_mask[..., None]
        stopping_embeddings = torch.repeat_interleave(self.stopping_embeddings, vertices.shape[0], dim=0)
        vertex_embeddings = torch.cat([stopping_embeddings, vertex_embeddings.to(torch.float32)], dim=1)

        vertex_embeddings = self.encoder(vertex_embeddings.transpose(0, 1)).transpose(0, 1)
        return vertex_embeddings

    def _embed_inputs(
        self,
        faces_long: torch.Tensor,
        vertex_embeddings: torch.Tensor,
        global_context_embedding: Optional[torch.Tensor] = None,
    ):
        """Provides embeddings for sampled faces

        Args:
            faces_long: A tensor of shape [batch_size, sampled_faces]
            vertex_embeddings: A tensor of shape [batch_size, num_vertices + 2, embed_size]
            global_context_embedding: If it exists its a tensor of shape [batch_size, embed_size]

        Returns:
            embeddings: A tensor of shape [num_faces + 1, batch_size, embed_size].
                        The first two dimensions are transposed such that they can be fed directly to the decoder.
        """
        face_embeddings = torch.zeros(
            size=[
                vertex_embeddings.shape[0],
                faces_long.shape[1],
                vertex_embeddings.shape[2],
            ]
        )

        for i in range(vertex_embeddings.shape[0]):
            face_embeddings[i] = vertex_embeddings[i, faces_long[i]]

        face_embeddings = face_embeddings.type_as(faces_long)
        pos_embeddings = self.pos_embedder(torch.arange(faces_long.shape[1]).type_as(faces_long)).type_as(faces_long)

        batch_size = face_embeddings.shape[0]

        if global_context_embedding is None:
            zero_embed_tiled = torch.repeat_interleave(self.zero_embed, batch_size, dim=0)
        else:
            zero_embed_tiled = global_context_embedding[:, None]

        embeddings = face_embeddings + pos_embeddings
        embeddings = torch.cat([zero_embed_tiled, embeddings], dim=1).transpose(0, 1).to(torch.float32)

        return embeddings

    def _project_to_pointers(self, inputs: torch.Tensor) -> torch.Tensor:
        """Passes inputs through a linear layer

        Args:
            inputs: A tensor of shape [...., embed_size]

        Returns:
            linear_outputs: A tensor of shape [..., embed_size]
        """
        return self.linear_layer(inputs)

    def _create_dist(
        self,
        vertex_embeddings: torch.Tensor,
        vertices_mask: torch.Tensor,
        faces_long: torch.Tensor,
        global_context_embedding: Optional[torch.Tensor] = None,
        sequential_context_embeddings: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        cache: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Outputs logits that can be used to create a categorical distribution

        Args:
            vertex_embeddings: A tensor of shape [batch_size, num_vertices + 2, embed_size] representing value embeddings for vertices
            vertices_mask: A tensor of shape [batch_size, num_vertices], representing which vertices are complete
            faces_long: A tensor of shape [batch_size, sampled_faces] representing currently sampled face indices
            global_context_embedding: A tensor of shape [batch_size, embed_size]
            sequential_context_embeddings: A tensor of shape [batch_size, num_vertices + 2, embed_size]
            temperature: A constant to normalize logits
            temperature: Scalar softmax temperature > 0.
            top_k: Number of tokens to keep for top-k sampling.
            top-p: Proportion of probability mass to keep for top-p sampling.
            cache: A list of dictionaries in the following format: {'k': torch.Tensor, 'v': torch.Tensor}. Each dictionary in the list represents the cache at the respective decoder layer.

        Returns:
            logits: Logits of shape [batch_size, sequence_length, num_vertices] that can be used to create a categorical distribution over vertex indices.
        """

        decoder_inputs = self._embed_inputs(
            faces_long.to(torch.int64),
            vertex_embeddings,
            global_context_embedding,
        )

        # check whether we are starting a sequence, or continuing a previous one
        if cache is not None:
            cached_decoder_inputs = decoder_inputs[-1:, :]
        else:
            cached_decoder_inputs = decoder_inputs
        decoder_outputs = self.decoder(
            cached_decoder_inputs,
            cache=cache,
            sequential_context_embeddings=sequential_context_embeddings.transpose(0, 1).type_as(faces_long),
        )

        pred_pointers = self._project_to_pointers(decoder_outputs.transpose(0, 1))

        num_dimensions = len(vertex_embeddings.shape)
        penultimate_dim, last_dim = num_dimensions - 2, num_dimensions - 1
        vertex_embeddings_transposed = vertex_embeddings.transpose(penultimate_dim, last_dim)

        logits = torch.matmul(pred_pointers, vertex_embeddings_transposed)
        logits = logits / math.sqrt(self.embedding_dim)

        # each example in the batch needs to have max_num_vertices, so that we can create a batch from multiple classes
        f_verts_mask = F.pad(vertices_mask, [2, 0, 0, 0], value=1)[:, None]

        logits = logits * f_verts_mask
        logits = logits - (1.0 - f_verts_mask) * 1e9
        logits = logits / temperature

        logits = top_k_logits(logits, top_k)
        logits = top_p_logits(logits, top_p)

        return logits

    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        """Forward method for Face Model

        Args:
            batch: A dictionary with keys for vertices, vertices_mask and faces

        Returns:
            logits: Logits of shape [batch_size, sequence_length, num_vertices] that can be used to create a categorical distribution over vertex indices.
        """
        vertex_embeddings, global_context, seq_context = self._prepare_context(batch)
        logits = self._create_dist(
            vertex_embeddings,
            batch["vertices_mask"],
            batch["faces"][:, :-1],
            global_context_embedding=global_context,
            sequential_context_embeddings=seq_context,
        )
        return logits

    def training_step(self, face_model_batch: Dict[str, Any], batch_idx: int) -> torch.float32:
        """Pytorch Lightning training step method

        Args:
            face_model_batch: A dictionary that contains batch data
            batch_idx: Which batch we are processing

        Returns:
            face_loss: NLL loss of generated categorical distribution
        """
        face_logits = self(face_model_batch)
        face_pred_dist = torch.distributions.categorical.Categorical(logits=face_logits)
        face_loss = -torch.sum(face_pred_dist.log_prob(face_model_batch["faces"]) * face_model_batch["faces_mask"])
        self.log("train_loss", face_loss)
        return face_loss

    def validation_step(self, val_batch, batch_idx):
        """Pytorch Lightning validation step

        Args:
            val_batch: A dictionary that contains batch data
            batch_idx: Which batch we are processing

        Returns:
            face_loss: NLL loss of generated categorical distribution
        """

        with torch.no_grad():
            face_logits = self(val_batch)
            face_pred_dist = torch.distributions.categorical.Categorical(logits=face_logits)
            face_loss = -torch.sum(face_pred_dist.log_prob(val_batch["faces"]) * val_batch["faces_mask"])
        self.log("val_loss", face_loss)
        return face_loss

    def configure_optimizers(self, total_steps:int) -> Dict[str, Any]:
        """Configure optimizer and learning rate scheduler

        Returns:
            dict: Dictionary with optimizer and lr scheduler
        """
        face_model_optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        face_model_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(face_model_optimizer, T_max=total_steps, eta_min=0, last_epoch=-1)
        face_model_scheduler_warmup = GradualWarmupScheduler(face_model_optimizer, multiplier=1, total_epoch=50000, after_scheduler=face_model_scheduler)
        return {"optimizer": face_model_optimizer, "scheduler_warmup": face_model_scheduler_warmup}

    def sample(
        self,
        context: Dict[str, Any],
        max_sample_length: int = 5000,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        only_return_complete: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Autoregressive sampling method to generate faces

        Args:
            context: A dictionary with keys for vertices and vertices_mask.
            max_sample_length: Maximum length of sampled faces. Sequences that do not complete are truncated.
            temperature: Scalar softmax temperature > 0.
            top_k: Number of tokens to keep for top-k sampling.
            top-p: Proportion of probability mass to keep for top-p sampling.
            only_return_complete: If True, only return completed samples. Otherwise return all samples along with completed indicator.

        Returns:
            outputs: Output dictionary with fields
                'context': A dictionary that contains modifications made to keys in the original context dictionary
                'completed': Tensor with shape [batch_size,]. Represents which faces have been fully sampled
                'faces': Tensor of shape [batch_size, num_faces]. Represents sampled faces.
                'num_face_indices': A tensor of shape [batch_size,]. Represents ending point of every sampled face.
        """
        vertex_embeddings, global_context, seq_context = self._prepare_context(context)
        num_samples = vertex_embeddings.shape[0]

        def _loop_body(i: int, samples: torch.Tensor, cache: Dict) -> Tuple[int, torch.Tensor]:
            """While-loop body for autoregression calculation.

            Args:
                i: Current iteration in the loop
                samples: tensor of shape [batch_size, i].
                cache: A list of dictionaries in the following format: {'k': torch.Tensor, 'v': torch.Tensor}. Each dictionary in the list represents the cache at the respective decoder layer.
            Returns:
                next_iter: i + 1.
                samples: tensor of shape [batch_size, i + 1].
            """

            logits = self._create_dist(
                vertex_embeddings,
                context["vertices_mask"],
                samples,
                global_context_embedding=global_context,
                sequential_context_embeddings=seq_context,
                cache=cache,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
            pred_dist = torch.distributions.categorical.Categorical(logits=logits)
            next_sample = pred_dist.sample()[:, -1:].to(torch.int32)
            samples = torch.cat([samples, next_sample], dim=1)
            return i + 1, samples

        def _stopping_cond(samples: torch.Tensor) -> bool:
            """Stopping condition for sampling while-loop. Looking for stop token (represented by 0)

            Args:
                samples: tensor of shape of [batch_size, i], where i is the current iteration.
            Returns:
                token_found: Boolean that represents if stop token has been found in every row of samples.
            """
            nonzero_boolean_matrix = torch.ne(samples, 0)
            reduced_matrix = torch.all(
                nonzero_boolean_matrix, dim=-1
            )  # Checks if stopping token is present in every row
            return torch.any(reduced_matrix)

        samples = torch.zeros([num_samples, 0], dtype=torch.int32).cuda() #yujia:cuda()
        cache = self.decoder.initialize_cache(num_samples)
        max_sample_length = max_sample_length or self.max_seq_length
        j = 0
        while _stopping_cond(samples) and j < max_sample_length:
            j, samples = _loop_body(j, samples, cache)

        completed_samples_boolean = samples == 0  # Checks for stopping token in every row of sampled faces
        complete_samples = torch.any(
            completed_samples_boolean, dim=-1
        )  # Tells us which samples are complete and which aren't
        sample_length = samples.shape[-1]  # Number of sampled faces
        max_one_ind, _ = torch.max(
            torch.arange(sample_length)[None].cuda() * (samples == 1).to(torch.int32),  #yujia:cuda()
            dim=-1,
        )  # Checking for new face tokens
        max_one_ind = max_one_ind.to(torch.int32)
        zero_inds = (torch.argmax((completed_samples_boolean).to(torch.int32), dim=-1)).to(
            torch.int32
        )  # Figuring out where the zeros are in every row
        num_face_indices = torch.where(complete_samples, zero_inds, max_one_ind) + 1  # How many vertices in each face

        faces_mask = (torch.arange(sample_length)[None].cuda() < num_face_indices[:, None] - 1).to(  #yujia:cuda()
            torch.int32
        )  # Faces mask turns the last true to false in each row

        samples = samples * faces_mask

        faces_mask = (torch.arange(sample_length)[None].cuda() < num_face_indices[:, None]).to(torch.int32)  #yujia:cuda()

        pad_size = max_sample_length - sample_length
        samples = F.pad(samples, [0, pad_size, 0, 0])

        if only_return_complete:
            samples = samples[complete_samples]
            num_face_indices = num_face_indices[complete_samples]
            for key in context:
                if key == 'files_list':
                    context[key] = [data for data, flag in zip(context[key], complete_samples) if flag]
                else:
                    context[key] = context[key][complete_samples]
            complete_samples = complete_samples[complete_samples]

        outputs = {
            "context": context,
            "completed": complete_samples,
            "faces": samples,
            "num_face_indices": num_face_indices,
        }

        return outputs


    def sample_mask(
        self,
        context: Dict[str, Any],
        max_sample_length: int = 5000,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        only_return_complete: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Autoregressive sampling method to generate faces

        Args:
            context: A dictionary with keys for vertices and vertices_mask.
            max_sample_length: Maximum length of sampled faces. Sequences that do not complete are truncated.
            temperature: Scalar softmax temperature > 0.
            top_k: Number of tokens to keep for top-k sampling.
            top-p: Proportion of probability mass to keep for top-p sampling.
            only_return_complete: If True, only return completed samples. Otherwise return all samples along with completed indicator.

        Returns:
            outputs: Output dictionary with fields
                'context': A dictionary that contains modifications made to keys in the original context dictionary
                'completed': Tensor with shape [batch_size,]. Represents which faces have been fully sampled
                'faces': Tensor of shape [batch_size, num_faces]. Represents sampled faces.
                'num_face_indices': A tensor of shape [batch_size,]. Represents ending point of every sampled face.
        """
        vertex_embeddings, global_context, seq_context = self._prepare_context(context)
        num_samples = vertex_embeddings.shape[0]

        def _loop_body(i: int, samples: torch.Tensor, cache: Dict) -> Tuple[int, torch.Tensor]:
            """While-loop body for autoregression calculation.

            Args:
                i: Current iteration in the loop
                samples: tensor of shape [batch_size, i].
                cache: A list of dictionaries in the following format: {'k': torch.Tensor, 'v': torch.Tensor}. Each dictionary in the list represents the cache at the respective decoder layer.
            Returns:
                next_iter: i + 1.
                samples: tensor of shape [batch_size, i + 1].
            """

            logits = self._create_dist(
                vertex_embeddings,
                context["vertices_mask"],
                samples,
                global_context_embedding=global_context,
                sequential_context_embeddings=seq_context,
                cache=cache,
                temperature=temperature,
                top_k=top_k,
                top_p=1.0,
            )    

            # 1. Masking the last generated token
            if samples.size(1) > 0:
                last_token = samples[:, -1]
                logits[:,:, last_token.long()] = -1e9

            # 2. Ensuring the first number of a new sub-sequence isn't less than the first number in the previous sub-sequence
            sub_sequence_starts = (samples == 1).nonzero()
            if sub_sequence_starts.size(0) == 1:
                last_start = sub_sequence_starts[-1, 1]
                logits[:, :, 2:samples[0][0]] = -1e9
            elif sub_sequence_starts.size(0) > 1:
                last_start = sub_sequence_starts[-1, 1]
                second_last_start = sub_sequence_starts[-2, 1]
                min_val = samples[torch.arange(samples.size(0)), second_last_start+1]
                logits[:, 2:min_val] = -1e9

            # 3. Ensuring numbers within a sub-sequence are greater than the first number in that sub-sequence
            if samples.size(1) > 0:
                if sub_sequence_starts.size(0) == 0:
                    logits[:,:,2:samples[0, 0]] = -1e9
                else:
                    if last_start.item() != i - 1:
                        logits[:,:,2:samples[0, last_start.item()+1]] = -1e9

            # 4. Ensuring numbers within a sub-sequence are unique
            if samples.size(1) > 0:
                if sub_sequence_starts.size(0) == 0:
                    for idx in range(samples.size(1)):
                        logits[:,:,samples[0, idx]] = -1e9
                else:
                    if last_start.item() != i - 1:
                        for idx in range(last_start.item()+1, samples.size(1)):
                            logits[:,:, samples[0, idx]] = -1e9
                                    
            pred_dist = torch.distributions.categorical.Categorical(logits=logits)
            next_sample = pred_dist.sample()[:, -1:].to(torch.int32)
            samples = torch.cat([samples, next_sample], dim=1)
            return i + 1, samples

        def _stopping_cond(samples: torch.Tensor) -> bool:
            """Stopping condition for sampling while-loop. Looking for stop token (represented by 0)

            Args:
                samples: tensor of shape of [batch_size, i], where i is the current iteration.
            Returns:
                token_found: Boolean that represents if stop token has been found in every row of samples.
            """
            nonzero_boolean_matrix = torch.ne(samples, 0)
            reduced_matrix = torch.all(
                nonzero_boolean_matrix, dim=-1
            )  # Checks if stopping token is present in every row
            return torch.any(reduced_matrix)

        samples = torch.zeros([num_samples, 0], dtype=torch.int32).cuda() #yujia:cuda()
        cache = self.decoder.initialize_cache(num_samples)
        max_sample_length = max_sample_length or self.max_seq_length
        j = 0
        while _stopping_cond(samples) and j < max_sample_length:
            j, samples = _loop_body(j, samples, cache)

        completed_samples_boolean = samples == 0  # Checks for stopping token in every row of sampled faces
        complete_samples = torch.any(
            completed_samples_boolean, dim=-1
        )  # Tells us which samples are complete and which aren't
        sample_length = samples.shape[-1]  # Number of sampled faces
        max_one_ind, _ = torch.max(
            torch.arange(sample_length)[None].cuda() * (samples == 1).to(torch.int32),  #yujia:cuda()
            dim=-1,
        )  # Checking for new face tokens
        max_one_ind = max_one_ind.to(torch.int32)
        zero_inds = (torch.argmax((completed_samples_boolean).to(torch.int32), dim=-1)).to(
            torch.int32
        )  # Figuring out where the zeros are in every row
        num_face_indices = torch.where(complete_samples, zero_inds, max_one_ind) + 1  # How many vertices in each face

        faces_mask = (torch.arange(sample_length)[None].cuda() < num_face_indices[:, None] - 1).to(  #yujia:cuda()
            torch.int32
        )  # Faces mask turns the last true to false in each row

        samples = samples * faces_mask

        faces_mask = (torch.arange(sample_length)[None].cuda() < num_face_indices[:, None]).to(torch.int32)  #yujia:cuda()

        pad_size = max_sample_length - sample_length
        samples = F.pad(samples, [0, pad_size, 0, 0])

        if only_return_complete:
            samples = samples[complete_samples]
            num_face_indices = num_face_indices[complete_samples]
            for key in context:
                if key == 'files_list':
                    context[key] = [data for data, flag in zip(context[key], complete_samples) if flag]
                else:
                    context[key] = context[key][complete_samples]
            complete_samples = complete_samples[complete_samples]

        outputs = {
            "context": context,
            "completed": complete_samples,
            "faces": samples,
            "num_face_indices": num_face_indices,
        }

        return outputs