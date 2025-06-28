import torch

def chunk_delta_rule_forward(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    beta: torch.Tensor,
    C: int,
    initial_state: torch.Tensor = None,
    output_final_state: bool = True,
):
    """
    Delta-rule forward pass with chunking.
    Q, K, V: [B, H, N, D]; beta: [B, H, N]; returns O: [B, H, N, D], state: [B, H, D, D]
    """
    B, H, N, D = Q.shape
    orig_dtype = Q.dtype
    num_chunks = N // C

    # reshape into chunks [B, H, num_chunks, C, D]
    Qc = Q.view(B, H, num_chunks, C, D)
    Kc = K.view(B, H, num_chunks, C, D)
    Vc = V.view(B, H, num_chunks, C, D)
    beta_c = beta.view(B, H, num_chunks, C)

    # simple beta-weighted K, V (replace with learned rule)
    Kb = Kc * beta_c.unsqueeze(-1)
    Vb = Vc * beta_c.unsqueeze(-1)

    # build T via lower-triangular operations (stub)
    T = -(Kb @ Kc.transpose(-1, -2))
    T = torch.tril(T, diagonal=-1)
    for i in range(1, C):
        T[:, :, :, i, :i] += (T[:, :, :, i, :, None] * T[:, :, :, :, :i]).sum(-2)
    T = T + torch.eye(C, device=Q.device, dtype=Q.dtype).view(1, 1, 1, C, C)

    # compute W and U
    W = T @ Kb
    U = T @ Vb

    # initial state S and output O
    S = initial_state if initial_state is not None else torch.zeros(B, H, D, D, device=Q.device, dtype=Q.dtype)
    O = torch.empty_like(V)

    # chunk-wise accumulation
    for i in range(num_chunks):
        q_i = Qc[:, :, i]        # [B,H,C,D]
        k_i = Kc[:, :, i]
        w_i = W[:, :, i]
        u_i = U[:, :, i] - w_i @ S
        o_inter = q_i @ S
        A_i = torch.tril(q_i @ k_i.transpose(-1, -2))
        o_intra = A_i @ u_i
        S = S + k_i.transpose(-1, -2) @ u_i
        O[:, :, i*C:(i+1)*C] = o_inter + o_intra

    if not output_final_state:
        S = None

    return O.to(orig_dtype), S