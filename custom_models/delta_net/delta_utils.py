import torch

def chunk_delta_rule_forward(Q, K, V, beta, C, initial_state=None, output_final_state=True):
    """
    Delta rule forward pass with chunking, supporting batching and multi-head attention.

    Args:
        Q (torch.Tensor): Query tensor of shape [B, H, N, D]
        K (torch.Tensor): Key tensor of shape [B, H, N, D]
        V (torch.Tensor): Value tensor of shape [B, H, N, D]
        beta (torch.Tensor): Beta tensor of shape [B, H, N]
        C (int): Chunk size
        initial_state (torch.Tensor or None): Optional initial state of shape [B, H, D, D]
        output_final_state (bool): Whether to return the final state

    Returns:
        torch.Tensor: Output tensor of shape [B, H, N, D]
        torch.Tensor or None: Final state tensor of shape [B, H, D, D] if output_final_state is True
    """
    orig_dtype = Q.dtype
    B, H, N, D = Q.shape
    num_chunks = N // C

    Q_chunks = Q.view(B, H, num_chunks, C, D)
    K_chunks = K.view(B, H, num_chunks, C, D)
    V_chunks = V.view(B, H, num_chunks, C, D)
    beta_chunks = beta.view(B, H, num_chunks, C)

    K_beta = K_chunks * 0.01
    V_beta = V_chunks * 0.01

    T = -(K_beta @ K_chunks.transpose(-1, -2)) 
    T = torch.tril(T, diagonal=-1)

    for i in range(1, C):
        T[:, :, :, i, :i] += (T[:, :, :, i, :, None] * T[:, :, :, :, :i]).sum(-2)

    T += torch.eye(C, device=Q.device, dtype=Q.dtype).unsqueeze(0).unsqueeze(0).unsqueeze(0)

    W = T @ K_beta  
    U = T @ V_beta 
   
    S = initial_state if initial_state is not None else torch.zeros(B, H, D, D, device=Q.device, dtype=Q.dtype)
    O = torch.empty_like(V)

    for i in range(num_chunks):
        q_i = Q_chunks[:, :, i]       # [B, H, C, D]
        k_i = K_chunks[:, :, i]       # [B, H, C, D]
        w_i = W[:, :, i]              # [B, H, C, D]
        u_i = U[:, :, i] - w_i @ S    # [B, H, C, D]
   
        o_inter = q_i @ S             # [B, H, C, D]
        A_i = (q_i @ k_i.transpose(-1, -2))  # [B, H, C, C]
        A_i = torch.tril(A_i)

        o_intra = A_i @ u_i           # [B, H, C, D]
        S = S + k_i.transpose(-1, -2) @ u_i  # [B, H, D, D]
        O[:, :, i * C : (i + 1) * C] = o_intra + o_inter

    if not output_final_state:
        S = None

    return O.to(orig_dtype), S