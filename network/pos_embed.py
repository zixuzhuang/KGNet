import torch


@torch.no_grad()
def pos_encoding(position, d_model, num_view=3, min_freq=1e-4):
    """
    Computes 3D sinusoidal positional encoding for a given position in 3D space.

    Args:
    - position (torch.Tensor): a tensor of shape (..., 3) where 3 represents the 3D coordinates (x, y, z).
    - d_model (int): the number of dimensions of the embeddings or the model.
    - min_freq (float): the minimum frequency for the sinusoidal waves.

    Returns:
    - torch.Tensor: a tensor of shape (..., d_model) containing the positional encodings.
    """
    # Ensure position is a tensor
    position = torch.as_tensor(position, dtype=torch.float32)
    device = position.device
    # Create a tensor of dimension indices scaled by 2 (sin, cos)
    dim = d_model // (num_view * 2)
    dims = torch.arange(dim, dtype=torch.float32).to(device)
    # Calculate frequencies for the positional encoding
    freqs = min_freq ** (2 * dims / d_model)

    # Apply sinusoidal function separately to each dimension x, y, z
    pos_emb = []
    for i in range(num_view):  # Process each dimension separately
        pos = position[..., i : i + 1]  # Extract the i-th dimension
        sin_enc = torch.sin(pos * freqs)
        cos_enc = torch.cos(pos * freqs)
        enc = torch.cat((sin_enc, cos_enc), dim=-1)  # Concatenate sin and cos
        pos_emb.append(enc)

    # Concatenate all encoding parts along the last dimension
    pos_emb = torch.cat(pos_emb, dim=-1)

    # Ensure the output dimension matches d_model
    if pos_emb.shape[-1] < d_model:
        padding_size = d_model - pos_emb.shape[-1]
        padding_emb = torch.zeros(*pos_emb.shape[:-1], padding_size, device=device)
        pos_emb = torch.cat([pos_emb, padding_emb], dim=-1)
    elif pos_emb.shape[-1] > d_model:
        pos_emb = pos_emb[..., :d_model]

    return pos_emb
