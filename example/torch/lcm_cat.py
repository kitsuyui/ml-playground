from typing import List

import torch
from torch import Tensor


def lcm_cat(tensors: List[Tensor], batch_first: bool = False) -> Tensor:
    """lcm_cat

    Concatenate tensors after expanding the sequence length to the
    least common multiple of the sequence lengths in the batch.
    """
    if batch_first:
        seq_dim = 1
    else:
        seq_dim = 0

    tensors = tensors[:]

    # LCM (Least Common Multiple) of sequence lengths
    lcm_seq_size = torch.tensor(1)
    for t in tensors:
        t_seq_size = torch.tensor(t.shape[seq_dim])
        torch.lcm(lcm_seq_size, t_seq_size, out=lcm_seq_size)

    # Repeat tensors to match the LCM
    for i, t in enumerate(tensors):
        t = t.flatten(start_dim=2).unsqueeze_(-1)
        seq_size = t.shape[seq_dim]
        repeat_times = lcm_seq_size // seq_size
        if batch_first:
            t = t.repeat(1, repeat_times, 1, 1)
        else:
            t = t.repeat(repeat_times, 1, 1, 1)
        tensors[i] = t.flatten(start_dim=2)

    # Concatenate
    return torch.cat(tensors, dim=-1)
