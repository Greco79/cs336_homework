import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import numpy.typing as npt

from collections.abc import Callable, Iterable
from typing import Optional
from einops import rearrange, einsum
from jaxtyping import Float, Int
from torch import Tensor


class Linear(nn.Module):
    """
    Given the weights of a Linear layer, compute the transformation of a batched input.

    Args:
        in_dim (int): The size of the input dimension
        out_dim (int): The size of the output dimension
        weights (Float[Tensor, "d_out d_in"]): The linear weights to use
        in_features (Float[Tensor, "... d_in"]): The output tensor to apply the function to
    
    Returns:
        Float[Tensor, "... d_out"]: The transformed output of your linear module.
    """
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
    """
    Given the weights of an Embedding layer, get the embeddings for a batch of token ids.

    Args:
        vocab_size (int): The number of embeddings in the vocabulary
        d_model (int): The size of the embedding dimension
        weights (Float[Tensor, "vocab_size d_model"]): The embedding vectors to fetch from
        token_ids (Int[Tensor, "..."]): The set of token ids to fetch from the Embedding layer
    
    Returns:
        Float[Tensor, "... d_model"]: Batch of embeddings returned by your Embedding layer.
    """
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        factory_kwargs={"device":device,"dtype":dtype}
        weight = torch.empty(num_embeddings,embedding_dim,  **factory_kwargs)
        self.W = torch.nn.Parameter(weight)
        torch.nn.init.trunc_normal_(self.W,mean=0,std=1,a=-3,b=3)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.W[token_ids]
    

class RMSNorm(nn.Module):
    """Given the weights of a RMSNorm affine transform,
    return the output of running RMSNorm on the input features.

    Args:
        d_model (int): The dimensionality of the RMSNorm input.
        eps: (float): A value added to the denominator for numerical stability.
        weights (Float[Tensor, "d_model"]): RMSNorm weights.
        in_features (Float[Tensor, "... d_model"]): Input features to run RMSNorm on. Can have arbitrary leading
            dimensions.

    Returns:
        Float[Tensor,"... d_model"]: Tensor of with the same shape as `in_features` with the output of running
        RMSNorm of the `in_features`.
    """
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        factory_kwargs={"device":device,"dtype":dtype}
        self.eps=eps
        self.scale=nn.Parameter(torch.ones(d_model, **factory_kwargs))


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        # Your code here performing RMSNorm
        rms=x.pow(2).mean(dim=-1,keepdim=True).add(self.eps).sqrt()
        x_norm=x/rms
        result = x_norm*self.scale
        # Return the result in the original dtype
        return result.to(in_dtype)

class SWIGLU(nn.Module):
    """Given the weights of a SwiGLU network, return
    the output of your implementation with these weights.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        d_ff (int): Dimensionality of the up-project happening internally to your swiglu.
        w1_weight (Float[Tensor, "d_ff d_model"]): Stored weights for W1
        w2_weight (Float[Tensor, "d_model d_ff"]): Stored weights for W2
        w3_weight (Float[Tensor, "d_ff d_model"]): Stored weights for W3
        in_features (Float[Tensor, "... d_model"]): Input embeddings to the feed-forward layer.

    Returns:
        Float[Tensor, "... d_model"]: Output embeddings of the same shape as the input embeddings.
    """
    def __init__(self, d_model: int, d_ff: int,device=None, dtype=None):
        super().__init__()
        self.d_model=d_model
        self.d_ff=d_ff

        factory_kwargs={"device":device,"dtype":dtype}
        w1=torch.empty(d_ff, d_model,  **factory_kwargs)
        w2=torch.empty(d_model, d_ff,  **factory_kwargs)
        w3=torch.empty(d_ff, d_model,  **factory_kwargs)
        self.w1=nn.Parameter(w1)
        self.w2=nn.Parameter(w2)
        self.w3=nn.Parameter(w3)

    
    def forward(self,x:torch.Tensor) -> torch.Tensor:
        tmp=einsum(x,self.w1, "... d, f d -> ... f")
        silu=tmp*torch.sigmoid(tmp)
        content=einsum(x,self.w3,"... d, f d -> ... f")
        gated=silu*content
        res=einsum(self.w2,gated,"d f, ... f -> ... d")
        return res

class ROPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.theta=theta
        self.d_k=d_k
        self.max_seq_len=max_seq_len

        self.rotary_dim=d_k-(d_k%2)
        half_dim=self.rotary_dim//2
        inv_freq = 1.0 / (self.theta ** (torch.arange(half_dim, dtype=torch.float32) / half_dim))
        pos=torch.arange(max_seq_len,dtype=torch.float32)
        freqs=torch.outer(pos,inv_freq)

        cos_cached=freqs.cos()
        sin_cached=freqs.sin()

        self.register_buffer("cos_cached",cos_cached,persistent=False)
        self.register_buffer("sin_cached",sin_cached,persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
            x: Tensor of shape (..., seq_len, d_k)
            token_positions: Tensor of shape (..., seq_len)
            Returns: Tensor of same shape as x
        """
        x1=x[...,:self.rotary_dim]
        x2=x[...,self.rotary_dim:]

        cos = self.cos_cached[token_positions]
        sin = self.sin_cached[token_positions]

        x1_even=x1[...,::2]
        x1_odd=x1[...,1::2]

        x1_rotated_even=x1_even*cos-x1_odd*sin
        x1_rotated_odd=x1_even*sin+x1_odd*cos

        x1_rotated=torch.stack([x1_rotated_even,x1_rotated_odd],dim=-1)
        x1_rotated=x1_rotated.flatten(-2)

        return torch.cat([x1_rotated,x2], dim=-1)

def soft_max(in_features, dim):
    shifted=in_features-in_features.max(dim=dim,keepdim=True).values
    exps=torch.exp(shifted)
    
    return exps/exps.sum(dim,keepdim=True)

def scaled_dot_product(Q,K,V, mask):
    """
        Q: Float[Tensor, " ... queries d_k"],
        K: Float[Tensor, " ... keys d_k"],
        V: Float[Tensor, " ... values d_v"],
        mask: Float[Tensor, " ... queries keys"] | None = None,
        -> Float[Tensor, " ... queries d_v"]
    """
    scores = Q @ K.transpose(-2, -1)
    d_k=Q.shape[-1]
    scores=scores/math.sqrt(d_k)
    if mask is not None:
        scores=scores.masked_fill(mask==False,float("-inf"))
    weights=torch.softmax(scores,dim=-1) # (queries, keys)
    return weights@V


class CausalMultiheadsSelfAttention(nn.Module):
    """
    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        q_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_v"]): Weights for the output projection
        in_features (Float[Tensor, "... sequence_length d_in"]): Tensor to run your implementation on.
    """
    def __init__(self, d_model: int, num_heads: int,device=None, dtype=None):
        super().__init__()
        self.d_model=d_model
        self.num_heads=num_heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.dk = self.dv = d_model//num_heads

    
    def forward(self,  q_proj_weight: torch.Tensor,k_proj_weight: torch.Tensor,v_proj_weight: torch.Tensor,o_proj_weight: torch.Tensor,in_features: torch.Tensor,max_sequence_len):
        *batch_shape, seq_len, d_in = in_features.shape
        q=F.linear(in_features, q_proj_weight)  # [..., seq_len, d_k]
        k=F.linear(in_features, k_proj_weight) 
        v=F.linear(in_features, v_proj_weight) 

        q=q.view(*batch_shape,seq_len,self.num_heads,self.dk).transpose(-2,-3)
        k=k.view(*batch_shape,seq_len,self.num_heads,self.dk).transpose(-2,-3)
        v=v.view(*batch_shape,seq_len,self.num_heads,self.dk).transpose(-2,-3)

        mask=torch.tril(torch.ones(max_sequence_len,max_sequence_len),diagonal=0).bool()
        causal_mask=mask.squeeze(0).squeeze(0)
        out=scaled_dot_product(q,k,v,causal_mask)
        out = out.transpose(1, 2).reshape(*batch_shape, seq_len, self.d_model) 
        out=F.linear(out,o_proj_weight)
        return out 

class CausalMultiheadsSelfAttentionWithRope(nn.Module):
    """
    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        theta (float): RoPE parameter.
        q_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_v"]): Weights for the output projection
        in_features (Float[Tensor, "... sequence_length d_in"]): Tensor to run your implementation on.
        token_positions (Int[Tensor, " ... sequence_length"] | None): Optional tensor with the positions of the tokens
    """
    def __init__(self, d_model: int, num_heads: int,theta: int ,max_sequence_len:int, device=None, dtype=None):
        super().__init__()
        self.d_model=d_model
        self.num_heads=num_heads
        self.theta=theta
        self.max_seq_len=max_sequence_len

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.dk = self.dv = d_model//num_heads

    
    def forward(self,  q_proj_weight: torch.Tensor,k_proj_weight: torch.Tensor,v_proj_weight: torch.Tensor,o_proj_weight: torch.Tensor,in_features: torch.Tensor, token_positions):
        rope=ROPE(self.theta,self.dk,self.max_seq_len)

        *batch_shape, seq_len, d_in = in_features.shape
        q=F.linear(in_features, q_proj_weight)  # [..., seq_len, d_k]
        k=F.linear(in_features, k_proj_weight) 
        v=F.linear(in_features, v_proj_weight) 

        q=q.view(*batch_shape,seq_len,self.num_heads,self.dk).transpose(-2,-3)# [...,num_heads, seq_len, d_k]
        k=k.view(*batch_shape,seq_len,self.num_heads,self.dk).transpose(-2,-3)
        v=v.view(*batch_shape,seq_len,self.num_heads,self.dk).transpose(-2,-3)

        q=rope(q,token_positions)
        k=rope(k,token_positions)

        mask=torch.tril(torch.ones(seq_len,seq_len),diagonal=0).bool()
        causal_mask=mask.unsqueeze(0).unsqueeze(0)
        out=scaled_dot_product(q,k,v,causal_mask)
        out = out.transpose(1, 2).reshape(*batch_shape, seq_len, self.d_model) 
        out=F.linear(out,o_proj_weight)
        return out 

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int,num_heads: int,d_ff: int,max_seq_len: int,theta: float):
        super().__init__()
        self.d_model=d_model
        self.num_heads=num_heads
        self.d_ff=d_ff
        self.max_seq_len=max_seq_len
        self.theta=theta

        self.norm1=RMSNorm(d_model)
        self.attn=CausalMultiheadsSelfAttentionWithRope(d_model,num_heads,theta,max_seq_len)
        self.norm2=RMSNorm(d_model)
        self.ffn=SWIGLU(d_model,d_ff)

    def forward(self,weights, in_features):
        *batch_shape, seq_len, d_in = in_features.shape
        
        ## 多头注意力
        with torch.no_grad():
            self.norm1.scale.copy_(weights["ln1.weight"])
        res_norm1=self.norm1(in_features)

        token_positions = torch.arange(seq_len, device=in_features.device)  # [seq_len]
        token_positions = token_positions.unsqueeze(0).expand(*batch_shape, seq_len)  # [batch_size, seq_len]
        
        res_attn=self.attn(weights["attn.q_proj.weight"],weights["attn.k_proj.weight"],weights["attn.v_proj.weight"],weights["attn.output_proj.weight"],res_norm1,token_positions)

        res_b1=in_features+res_attn
        
        ## FFN
        with torch.no_grad():
            self.norm2.scale.copy_(weights["ln2.weight"])
        res_norm2=self.norm2(res_b1)

        with torch.no_grad():
            self.ffn.w1.copy_(weights["ffn.w1.weight"])
            self.ffn.w2.copy_(weights["ffn.w2.weight"])
            self.ffn.w3.copy_(weights["ffn.w3.weight"])

        res_ffn=self.ffn(res_norm2)
        res=res_b1+res_ffn
        return res

class Transformer_Lm(nn.Module):
    def __init__(self, 
                 vocab_size: int,
                 context_length: int,
                 d_model: int,
                 num_layers: int,
                 num_heads: int,
                 d_ff: int,
                 rope_theta: float,):
        super().__init__()
        self.vocab_size=vocab_size
        self.context_length=context_length
        self.d_model=d_model
        self.num_layers=num_layers
        self.num_heads=num_heads
        self.d_ff=d_ff
        self.rope_theta=rope_theta

        
        self.embedding=Embedding(vocab_size,d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, context_length, rope_theta)
            for _ in range(num_layers)
        ])
        self.norm=RMSNorm(d_model)
        self.linear=Linear(d_model,vocab_size)
    
    def forward(self, weights: dict[str, Tensor], in_indices: Int[Tensor, " batch_size sequence_length"]):
        with torch.no_grad():
            self.embedding.W.copy_(weights["token_embeddings.weight"])
        emb=self.embedding(in_indices)
        res_block=emb
        for i,block in enumerate(self.blocks):
            layer_weights={
                key.replace(f"layers.{i}.",""):val
                for key, val in weights.items()
                if key.startswith(f"layers.{i}.")
            }
            res_block=block(layer_weights,res_block)

        with torch.no_grad():
            self.norm.scale.copy_(weights["ln_final.weight"])
        res_norm=self.norm(res_block)

        with torch.no_grad():
            self.linear.W.copy_(weights["lm_head.weight"])
        res_linear=self.linear(res_norm)

        res_lm=soft_max(res_linear,-1)

        return res_linear

def cross_entropy(inputs, targets):
    """Given a tensor of inputs and targets, compute the average cross-entropy
    loss across examples.

    Args:
        inputs (Float[Tensor, "batch_size vocab_size"]): inputs[i][j] is the
            unnormalized logit of jth class for the ith example.
        targets (Int[Tensor, "batch_size"]): Tensor of shape (batch_size,) with the index of the correct class.
            Each value must be between 0 and `num_classes - 1`.

    Returns:
        Float[Tensor, ""]: The average cross-entropy loss across examples.
    """
    max_logits=inputs.max(dim=-1,keepdim=True).values
    stable_logits=inputs-max_logits

    sum_logits=torch.log(torch.exp(stable_logits).sum(dim=-1))
    target_logits=torch.gather(stable_logits,dim=-1,index=targets.unsqueeze(-1)).squeeze(-1)
    loss=-target_logits+sum_logits

    return loss.mean()


class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"] # Get the learning rate.
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p] # Get state associated with p.
                t = state.get("t", 0) # Get iteration number from the state, or initial value.
                grad = p.grad.data # Get the gradient of loss with respect to p.
                p.data -= lr / math.sqrt(t + 1) * grad # Update weight tensor in-place.
                state["t"] = t + 1 # Increment iteration number.
        return loss

class AdamW(torch.optim.Optimizer):
    def __init__(self,params,lr=1e-3,betas=(0.9,0.999),eps=1e-8,weight_decay=0.01):
        defaults=dict(lr=lr,betas=betas,eps=eps,weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            beta1,beta2=group['betas']
            lr=group['lr']
            eps=group['eps']
            weight_decay=group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad=p.grad.data
                state=self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
            
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                state['step']+=1
                t=state['step']
                exp_avg.mul_(beta1).add_(grad,alpha=1-beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                bias1=1-beta1**t
                bias2=1-beta2**t
                corrected_lr=lr*math.sqrt(bias2)/bias1

                p.data.addcdiv_(exp_avg, exp_avg_sq.sqrt().add_(eps), value=-corrected_lr)

                p.data.add_(p.data, alpha=-lr * weight_decay)

                
        return
    
def learning_rate_schedule(it,max_learning_rate,min_learning_rate,warmup_iters,cosine_cycle_iters):
    if it<warmup_iters:
        return it*max_learning_rate/warmup_iters
    elif warmup_iters<=it<=cosine_cycle_iters:
        return min_learning_rate+0.5*(1+math.cos((it-warmup_iters)/(cosine_cycle_iters-warmup_iters)*math.pi))*(max_learning_rate-min_learning_rate)
    else:
        return min_learning_rate

def gradient_clipping(parameters,max_l2_norm):
    eps = 1e-6
    total_norm = 0.0

    # Step 1: Compute total gradient L2 norm
    for p in parameters:
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2

    total_norm = total_norm ** 0.5

    # Step 2: If norm too large, scale all gradients
    if total_norm > max_l2_norm:
        clip_coef = max_l2_norm / (total_norm + eps)
        for p in parameters:
            if p.grad is not None:
                p.grad.data.mul_(clip_coef)

def data_loading(dataset: npt.NDArray, batch_size: int, context_length: int, device: str):
    n=len(dataset)
    ix=np.random.randint(0,n-context_length,size=batch_size)
    inputs=[dataset[i:i+context_length] for i in ix]
    targets=[dataset[i+1:i+1+context_length] for i in ix]
    return (torch.tensor(inputs,dtype=torch.long,device=device),torch.tensor(targets,dtype=torch.long,device=device))


def save_checkpoint(model, optimizer, iteration, out):
# should dump all the state from the
# first three parameters into the file-like object out. You can use the state_dict method of both
# the model and the optimizer to get their relevant states and use torch.save(obj, out) to dump
# obj into out (PyTorch supports either a path or a file-like object here). A typical choice is to
# have obj be a dictionary, but you can use whatever format you want as long as you can load your
# checkpoint later.
# This function expects the following parameters:
# model: torch.nn.Module
# optimizer: torch.optim.Optimizer
# iteration: int
# out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]
    obj={
        "model":model.state_dict(),
        "optimizer":optimizer.state_dict(),
        "iteration":int(iteration),
    }
    torch.save(obj,out)


def load_checkpoint(src, model, optimizer):
# should load a checkpoint from src (path or filelike object), and then recover the model and optimizer states from that checkpoint. Your
# function should return the iteration number that was saved to the checkpoint. You can use
# torch.load(src) to recover what you saved in your save_checkpoint implementation, and the
# load_state_dict method in both the model and optimizers to return them to their previous
# states.
# This function expects the following parameters:
# src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]
# model: torch.nn.Module
# optimizer: torch.optim.Optimizer
    ckpt=torch.load(src,map_location="cpu")
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    return int(ckpt["iteration"])





        




            
        
        
        
            
        
        
