
def flash_fwd_torch(ctx,Q,K,V,is_causal=False):
    N, d = Q.shape
    Br,Bc = max(16, d), max(16, d) #Q块的大小，K/V块的大小
    Tr = (N + Br - 1) // Br #Q分成多少块
    Tc = (N + Bc - 1) // Bc #K/V分成多少块
    #output
    O = torch.empty_like(Q)
    L = torch.empty(N, device=Q.device, dtype=Q.dtype)
    scale = 1 / sqrt(d)

    for i in range(Tr): # Q外循环
        q_start = i * Br
        q_end = min((i+1) * Br, N)

        br = q_end - q_start #当前Q块 有多少个token

        #修正变量
        mi = torch.full((br,), float('-inf'), device=Q.device, dtype=Q.dtype)#累积到当前位置的最大值
        li = torch.zeros((br,) , device=Q.device, dtype=Q.dtype)#累积到当前位置的分母
        oi = torch.zeros((br, d) , device=Q.device, dtype=Q.dtype)

        qi = Q[q_start: q_end] #当前query 块
        for j in range(Tc): #K/V外循环
            k_start = j * Bc
            k_end = min((j+1) * Bc, N)

            bc = k_end - k_start #当前K块 有多少个token

            kj = K[k_start: k_end] # 当前key 块
            vj = V[k_start: k_end] # 当前value 块

            sij = torch.matmul(qi,kj.T) * scale #sij shape = [br , bc]
            m_cur = sij.max(dim=1).values
            m_new = torch.maximum(mi, m_cur) #m_new shape = [br]

            pij = torch.exp(sij - m_new[:, None])

            alter = torch.exp(mi - m_new)

            #l_new = exp(m_old - m_new) * l_old + sum(exp(S - m_new))
            li = alter * li + pij.sum(dim=1)

            oi = alter[:, None] * oi + torch.matmul(pij,vj)
            mi = m_new
        L[q_start:q_end] = mi + torch.log(li)
        O[q_start:q_end] = oi / li

    ctx.save_for_backward(Q,K,V,O,L)
    ctx.is_causal = is_causal
    return O