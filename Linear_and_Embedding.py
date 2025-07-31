import torch
import torch.nn as nn
import math
from einops import rearrange, einsum

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        factory_kwargs={"device":device,"dtype":dtype}
        weight = torch.empty(out_features, in_features, **factory_kwargs)
        self.W = torch.nn.Parameter(weight)

        std=math.sqrt(2/(in_features+out_features))
        torch.nn.init.trunc_normal_(self.W,mean=0,std=std,a=-3*std,b=3*std)

    def forward(self, x: torch.Tensor) -> torch.Tensor: # Apply the linear transformation to the input.
        return einsum(x,self.W, "... d, o d -> ... o")

class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        factory_kwargs={"device":device,"dtype":dtype}
        weight = torch.empty(num_embeddings,embedding_dim,  **factory_kwargs)
        self.W = torch.nn.Parameter(weight)
        torch.nn.init.trunc_normal_(self.W,mean=0,std=1,a=-3,b=3)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.W[token_ids]
