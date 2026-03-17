import torch
import triton
import triton.language as tl
from triton import cdiv
@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    Q_stride_b, Q_stride_row, Q_stride_col,
    K_stride_b, K_stride_row, K_stride_col,
    V_stride_b, V_stride_row, V_stride_col,
    O_stride_b, O_stride_row, O_stride_col,
    L_stride_b, L_stride_row,#1
    N_QUERIES, N_KEYS,
    scale,
    D:tl.constexpr,
    Q_TILE_SIZE:tl.constexpr,
    K_TILE_SIZE:tl.constexpr
):
    query_tile_index = tl.program_id(0) #当前query块 索引
    batch_index = tl.program_id(1)

    #当前 query块的指针/显存位置
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * Q_stride_b,
        shape = (N_QUERIES, D),
        strides = (Q_stride_row, Q_stride_col),
        offsets = (query_tile_index * Q_TILE_SIZE, 0), #行偏移，列不变
        block_shape = (Q_TILE_SIZE, D),
        order = (1, 0)
    )
    #当前 key块的指针/显存位置
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * K_stride_b,
        shape = (N_KEYS, D),
        strides = (K_stride_row, K_stride_col),
        offsets = (0, 0), #行列都不偏移
        block_shape = (K_TILE_SIZE, D),
        order = (1, 0)
    )
    #当前 value块的指针/显存位置
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * V_stride_b,
        shape = (N_KEYS, D),
        strides = (V_stride_row, V_stride_col),
        offsets = (0, 0), #行列都不偏移
        block_shape = (K_TILE_SIZE, D),
        order = (1, 0)
    )

    #当前 output块的指针/显存位置
    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * O_stride_b,
        shape = (N_QUERIES, D),
        strides = (O_stride_row, O_stride_col),
        offsets = (query_tile_index * Q_TILE_SIZE, 0), #行偏移，列不变
        block_shape = (Q_TILE_SIZE, D),
        order = (1, 0)
    )

    #当前块结果修正位置
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * L_stride_b,
        shape = (N_QUERIES,),
        strides = (L_stride_row, ),
        offsets = (query_tile_index * Q_TILE_SIZE, ), #行偏移
        block_shape = (Q_TILE_SIZE,),
        order = (0,)
    )

    query = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")

    output = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
    l = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)#softmax 分母
    m = tl.full((Q_TILE_SIZE,), value=-float('inf'), dtype=tl.float32)

    for i in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):

        key = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
        value = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")

        # key_T = tl.transpose(key)
        score = tl.dot(query, key.T, allow_tf32=False) * scale #Q_TILE_SIZE * K_TILE_SIZE

        key_range = tl.arange(0, K_TILE_SIZE) + i * K_TILE_SIZE
        mask = key_range[None, :] < N_KEYS  # [1, K_TILE_SIZE]
        score = tl.where(mask, score, float('-inf'))
        raw_max = tl.max(score, axis=1)
        m_new = tl.maximum(m, raw_max) #Q_TILE_SIZE
        p_unnormalized = tl.exp(score - m_new[:,None]) #Q_TILE_SIZE * K_TILE_SIZE
        alter = tl.exp(m - m_new) #Q_TILE_SIZE
        l = alter * l + tl.sum(p_unnormalized, axis=1) #修正 #Q_TILE_SIZE

        # output = output * alter[:, None] + tl.dot(p_unnormalized.to(value.dtype), value).to(tl.float32)
        output = output * alter[:, None]

        output = tl.dot(
            p_unnormalized.to(value.dtype),
            value,
            acc=output
        ).to(tl.float32)

        m = m_new
        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))
    

    tl.store(O_block_ptr, (output / l[:,None]).to(query.dtype), boundary_check=(0, 1))
    tl.store(L_block_ptr, m + tl.log(l), boundary_check=(0,))

class FlashAttentionFunc(torch.autograd.Function):
      
    def forward(ctx, q, k, v):
        #传入的shape 是 batch_size,num_heads,seq_len,head_dim
        b, h, n, d = q.shape
        q = q.reshape(b*h, n, d)
        k = k.reshape(b*h, n, d)
        v = v.reshape(b*h, n, d)

        
        O = torch.empty_like(q)
        L = torch.empty((b*h, n), device=q.device, dtype=torch.float32)

        #Q,K/V块大小
        Q_TILE_SIZE, K_TILE_SIZE = max(16, d), max(16, d) #Q块的大小，K/V块的大小
       
        scale = d ** -0.5

        flash_fwd_kernel[(cdiv(n, Q_TILE_SIZE), b*h)](
            q, k, v,
            O, L,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
            n, n,
            scale,
            d,
            Q_TILE_SIZE,
            K_TILE_SIZE
        )
        O = O.reshape(b, h, n, d).transpose(1,2).reshape(b, n, h*d)
        L = L.reshape(b, h, n).transpose(1,2) 
        return O

def flash_attention(q, k, v):
    return FlashAttentionFunc.apply(q, k, v)









        

            

            





                

