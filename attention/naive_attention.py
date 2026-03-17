import torch
import torch.nn as nn
from math import sqrt
from flash_attention_triton import flash_attention

def scaled_dot_product_attention(q,k,v):
    d = q.size(-1)
    scale = 1 / sqrt(d)

    score = torch.matmul(q, k.transpose(-2,-1)) * scale

    # #softmax
    # m = score.max(dim=-1, keepdim=True).values
    # scores_exp = torch.exp(score - m)
    # scores_sum = scores_exp.sum(dim=-1, keepdim=True)
    # p_weight = scores_exp / scores_sum
    # if torch.isnan(p_weight).any():
    #     print("Warning: NaN detected in p_weight")

    attention_weights = torch.softmax(score, dim=-1)
    o = torch.matmul(attention_weights,v)
    return o



class SelfAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert dim % num_heads == 0

        self.qkv = nn.Linear(dim, 3 * dim)
        # self.k = nn.Linear(dim, dim)
        # self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
      

    def forward(self,x, use_flash = False):
        r"""
        Args:
            x(Tensor): Shape [B, N, D]
        """
        b, n, _ = x.shape 
        h, d  = self.num_heads, self.head_dim
        # query = self.q(x).view(b, n, h, d).transpose(1, 2) #(b,h,n,d)
        # key   = self.k(x).view(b, n, h, d).transpose(1, 2)
        # value = self.v(x).view(b, n, h, d).transpose(1, 2)
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(b, n, h, d).transpose(1, 2)
        k = k.view(b, n, h, d).transpose(1, 2)
        v = v.view(b, n, h, d).transpose(1, 2)# (b,h,n,d)

        if use_flash:
            out = flash_attention(q, k, v)
            print('use_flash',out.shape)
        else:
            out = scaled_dot_product_attention(q, k, v) #(b,h,n,d)
            print('standard',out.shape)
            out = out.transpose(1, 2).reshape(b, n, h * d)

            print('standard reshape',out.shape)
            

        return self.o(out)


if __name__ == "__main__":

    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    
    dim = 1024
    num_heads = 16
    model = SelfAttention(dim, num_heads).cuda()
    x = torch.randn(1,768,dim).cuda() #
    
    # o = model(x, True)
    # print(o)
    # o2 = model(x, False)
    # print(o2)
    model.eval()
    
    # 计算两种方式的输出
    with torch.no_grad():
        o_flash = model(x, use_flash=True)
        print(o_flash)
        o_standard = model(x, use_flash=False)
        print(o_standard)
    print("Flash Attention输出 shape:", o_flash.shape)
    print("Standard Attention输出 shape:", o_standard.shape)

    rtol = 1e-05  # 相对容差
    atol = 1e-08  # 绝对容差


    if torch.allclose(o_flash, o_standard, rtol=rtol, atol=atol):
        print("两个张量在容差范围内相同")
    else:
        print("两个张量不同")




        



       




      
        
        