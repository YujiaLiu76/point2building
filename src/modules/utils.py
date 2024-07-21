import copy
import torch
import torch.nn as nn
from torch.nn import ModuleList
import torch.nn.functional as F


def top_k_logits(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Masks logits such that logits not in top-k are small

    Args:
        logits: tensor representing network predictions
        k: how many logits to not filter out

    Returns:
        logits: logits with top-k logits remaining intact
    """

    if k == 0:
        return logits
    else:
        values, _ = torch.topk(logits, k)
        k_largest = torch.min(values)
        logits = torch.where(torch.le(logits, k_largest), torch.ones_like(logits) * -1e9, logits)
        return logits

def top_p_logits(logits: torch.Tensor, p: float) -> torch.Tensor:
    """Masks logits using nucleus (top-p) sampling

    Args:
        logits: Network predictions
        top-p: What probability of the predictions we want to keep unmasked
    Returns:
        logits: logits with top-p prob mass remaining intact
    """

    if p == 1:
        return logits
    else:
        logit_shape = logits.shape
        seq, dim = logit_shape[1], logit_shape[2]
        logits = torch.reshape(logits, [-1, dim])
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        cumulative_probs = torch.roll(cumulative_probs, 1, -1)
        cumulative_probs[:, 0] = 0
        sorted_indices_to_remove = (cumulative_probs > p).to(logits.dtype)
        logits_ordered = sorted_logits - sorted_indices_to_remove * 1e9
        logits = logits_ordered.gather(1, sorted_indices.argsort(-1))
        return torch.reshape(logits, [-1, seq, dim])

def get_clones(module: nn.Module, N: int) -> ModuleList:
    """Clone a module n-times

    Args:
        module: module to clone
        N: how many times to clone the module
    Returns:
        module_list: ModuleList that contains module n times
    """
    return ModuleList([copy.deepcopy(module) for _ in range(N)])

def embedding_to_padding(emb: torch.Tensor) -> torch.Tensor:
    """
    Calculates the padding mask based on which all embeddings are all zero.
    Args:
        emb: A Tensor with shape [..., depth]
    Returns:
        A float tensor with shape [...]. Each element is 1 if its corresponding embedding vector is all zero, and is 0 otherwise.
    """
    emb_sum = torch.sum(torch.abs(emb), dim=-1)
    float_emb_sum = emb_sum.to(torch.float32)
    return (float_emb_sum == 0.0).transpose(0, 1)