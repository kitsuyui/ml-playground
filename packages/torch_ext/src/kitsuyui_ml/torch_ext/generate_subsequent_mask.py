"""
generate square subsequent mask for transformer models.

nin = float("-inf")  # for short-hand notation
when the given size is 5, the mask will look like this:

tensor = torch.Tensor([
    [0.0, nin, nin, nin, nin],
    [0.0, 0.0, nin, nin, nin],
    [0.0, 0.0, 0.0, nin, nin],
    [0.0, 0.0, 0.0, 0.0, nin],
    [0.0, 0.0, 0.0, 0.0, 0.0],
])

this prevents the model from looking ahead in the sequence,
"""

import torch


def generate_square_subsequent_mask_1(
    size: int, *, device: torch.device | None = None
) -> torch.Tensor:
    """
    Generate a square subsequent mask for a transformer model.
    Args:
        size (int): The size of the mask (number of tokens).
        device (torch.device, optional): The device to create the mask on. Defaults to "cpu".
    Returns:
        torch.Tensor: A square mask of shape (size, size) with the upper triangular part masked out.
    The mask is filled with 0.0 for valid positions and -inf for masked positions
    """
    mask = (torch.triu(torch.ones((size, size), device=device)) == 1).transpose(0, 1)
    mask = (
        mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, 0.0)
    )
    return mask


def generate_square_subsequent_mask_2(
    size: int, *, device: torch.device | None = None
) -> torch.Tensor:
    """
    Generate a square subsequent mask for a transformer model.
    Args:
        size (int): The size of the mask (number of tokens).
        device (torch.device, optional): The device to create the mask on. Defaults to "cpu".
    Returns:
        torch.Tensor: A square mask of shape (size, size) with the upper triangular part masked out.
    The mask is filled with 0.0 for valid positions and -inf for masked positions
    """
    mask = torch.nn.Transformer.generate_square_subsequent_mask(size, device=device)
    return mask
