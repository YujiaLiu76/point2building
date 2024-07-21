from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.nn import MultiheadAttention, Linear, Dropout, LayerNorm, ReLU, Parameter

from .utils import get_clones


class PolygenDecoderLayer(nn.TransformerDecoderLayer):
    """
    Decoder module as described in Vaswani et al. 2017. Uses masked self-attention and non-masked cross attention for sequential context modules.
    Implements Cache for faster decoding. The role of the cache is to store key-value pairs such that these pairs don't have to be regenerated
    through lengthy matrix multiplications for each forward pass in the decoder.
    """

    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.2,
        re_zero: bool = True,
    ) -> None:
        """
        Initializes PolygenDecoderLayer
        Args:
            d_model: Size of the embedding vectors.
            nhead: Number of multihead attention heads.
            dim_feedforward: size of fully connected layer.
            dropout: Dropout rate applied after ReLU in each connected layer.
            re_zero: If True, Alpha scale residuals with zero initialization.
        """
        super(PolygenDecoderLayer, self).__init__(d_model, nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)

        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)

        self.activation = ReLU()

        self.re_zero = re_zero
        self.alpha = Parameter(data=torch.Tensor([0.0]))
        self.beta = Parameter(data=torch.Tensor([0.0]))
        self.gamma = Parameter(data=torch.Tensor([0.0]))

    def forward(
        self,
        tgt: torch.Tensor,
        memory: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
        cache: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Forward method of Decoder Layer

        Args:
            tgt: A Tensor of shape [sequence_length, batch_size, embed_size]. Represents the input sequence.
            memory: A Tensor of shape [source_sequence_length, batch_size, embed_size]. Represents the sequence from the last layer of the encoder.
            tgt_mask: A Tensor of shape [sequence_length, sequence_length]. The mask for the target sequence.
            memory_mask: A Tensor of shape [sequence_length, source_sequence_length]. The mask for the memory sequence.
            tgt_key_padding_mask: A Tensor of shape [batch_size, sequence_length]. A Tensor that ignores specified padding elements in the target sequence.
            memory_key_padding_mask: A Tensor of shape [batch_size, source_sequence_length]. A Tensor that ignores specified padding elements in the memory sequence.
            cache: A Dictionary in the following format: {'k': torch.Tensor, 'v': torch.Tensor}. Used for fast decoding.

        Returns:
            tgt: A Tensor of shape [sequence_length, batch_size, embed_size]. The resultant tensor after the forward loop of one decoder layer.
        """
        if cache is not None:
            saved_key = cache["k"]
            saved_value = cache["v"]
            key = cache["k"] = torch.cat([saved_key, tgt], axis=0)
            value = cache["v"] = torch.cat([saved_value, tgt], axis=0)
        else:
            key = tgt
            value = tgt
        tgt2 = self.norm1(tgt)
        tgt2 = self.self_attn(tgt, key, value, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        if self.re_zero:
            tgt2 = tgt2 * self.alpha
        tgt = tgt + self.dropout1(tgt2)
        if memory is not None:
            tgt2 = self.norm2(tgt)
            tgt2 = self.multihead_attn(
                tgt,
                memory.float(),
                memory.float(),
                attn_mask=memory_mask,
                key_padding_mask=memory_key_padding_mask,
            )[0]
            if self.re_zero:
                tgt2 = tgt2 * self.beta
            tgt2 = self.dropout2(tgt2)
            tgt = tgt + tgt2
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear1(tgt2)
        tgt2 = self.activation(tgt2)
        tgt2 = self.linear2(tgt2)
        if self.re_zero:
            tgt2 = tgt2 * self.gamma
        tgt2 = self.dropout(tgt2)
        tgt = tgt + tgt2
        return tgt


class PolygenDecoder(nn.Module):
    """
    A modified version of Pytorch's Transformer Decoder implementation that takes into account the concept of a cache for fast decoding.
    """

    def __init__(
        self,
        decoder_layer: nn.TransformerDecoderLayer,
        num_layers: int,
        norm: Optional[nn.Module] = None,
    ) -> None:
        """
        Initialization of PolygenDecoder
        Args:
            decoder_layer: A Pytorch Module of type nn.TransformerDecoderLayer.
            num_layers: The number of decoder layers.
            norm: The type of layer norm to be applied at the end of the forward method of the Decoder.
        """
        super(PolygenDecoder, self).__init__()
        self.layers = get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(
        self,
        tgt: torch.Tensor,
        memory: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
        cache: Optional[Tuple] = None,
    ) -> torch.Tensor:
        """
        Forward method of Decoder Layer

        Args:
            tgt: A Tensor of shape [sequence_length, batch_size, embed_size]. Represents the input sequence.
            memory: A Tensor of shape [source_sequence_length, batch_size, embed_size]. Represents the sequence from the last layer of the encoder.
            tgt_mask: A Tensor of shape [sequence_length, sequence_length]. The mask for the target sequence.
            memory_mask: A Tensor of shape [sequence_length, source_sequence_length]. The mask for the memory sequence.
            tgt_key_padding_mask: A Tensor of shape [batch_size, sequence_length]. A Tensor that ignores specified padding elements in the target sequence.
            memory_key_padding_mask: A Tensor of shape [batch_size, source_sequence_length]. A Tensor that ignores specified padding elements in the memory sequence.
            cache: A list of dictionaries in the following format: {'k': torch.Tensor, 'v': torch.Tensor}. Each dictionary in the list represents the cache at the respective decoder layer.
        Returns:
            tgt: A Tensor of shape [sequence_length, batch_size, embed_size]. The resultant tensor after the forward loop of all the decoder layers.
        """
        output = tgt

        for i, mod in enumerate(self.layers):
            if cache is not None:
                layer_cache = cache[i]
                tgt_mask = None
            else:
                layer_cache = None
            output = mod(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                cache=layer_cache,
            )

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        device: str,
        hidden_size: int = 256,
        fc_size: int = 1024,
        num_heads: int = 4,
        layer_norm: bool = True,
        num_layers: int = 8,
        dropout_rate: float = 0.2,
    ) -> None:
        """TransformerDecoder that combines PolygenDecoderLayer and PolygenDecoder

        Args:
            hidden_size: Size of the embedding vectors.
            fc_size: Size of the fully connected layer.
            num_heads: Number of multihead attention heads.
            layer_norm: Boolean variable that signifies if layer normalization should be used.
            num_layers: Number of decoder layers in the decoder.
            dropout_rate: Dropout rate applied immediately after the ReLU in each fully connected layer.
        """
        super(TransformerDecoder, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.decoder = PolygenDecoder(
            PolygenDecoderLayer(
                d_model=hidden_size,
                nhead=num_heads,
                dim_feedforward=fc_size,
                dropout=dropout_rate,
            ),
            num_layers=num_layers,
            norm=LayerNorm(self.hidden_size),
        )

    def initialize_cache(self, batch_size) -> Dict[str, torch.Tensor]:
        """
        Initializes the cache to be used in fast decoding

        Args:
            batch_size: Batch size of the inputs.
        Returns:
            cache: A list of dictionaries where each dictionary contains a key and value for a specific Decoder Layer.
        """
        k = torch.zeros([0, batch_size, self.hidden_size], device=self.device)
        v = torch.zeros([0, batch_size, self.hidden_size], device=self.device)
        cache = [{"k": k, "v": v} for _ in range(self.num_layers)]
        return cache

    def generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """
        Generates a target mask for the input sequence

        Args:
            sz: The Input Sequence Length.
        Returns:
            mask: A lower triangular matrix of shape [sequence_length, sequence_length].
        """
        mask = (torch.triu(torch.ones(sz, sz, device=self.device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(
        self,
        inputs: torch.Tensor,
        sequential_context_embeddings: Optional[torch.Tensor] = None,
        cache: Optional[tuple] = None,
    ) -> torch.Tensor:
        """The forward method of the Transformer Decoder

        Args:
            inputs: A Tensor of shape [sequence_length, batch_size, embed_size]. Represents the input sequence.
            cache: A list of dictionaries in the following format: {'k': torch.Tensor, 'v': torch.Tensor}. Each dictionary in the list represents the cache at the respective decoder layer.
        Returns:
            out: A Tensor of shape [sequence_length, batch_size, embed_size]. The resultant tensor after the forward loop of all the decoder layers.
        """
        sz = inputs.shape[0]
        mask = self.generate_square_subsequent_mask(sz)
        out = self.decoder(inputs, memory=sequential_context_embeddings, tgt_mask=mask, cache=cache)
        return out
