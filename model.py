import math
import struct
import inspect
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np
import jittor as jt
from jittor import nn
from jittor import Module

jt.flags.use_cuda = 1  # 启用 CUDA


@dataclass
class ModelArgs:
    dim: int = 512
    n_layers: int = 8
    n_heads: int = 8
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1
    multiple_of: int = 256
    norm_eps: float = 1e-5
    max_seq_len: int = 256
    dropout: float = 0.0


class RMSNorm(Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = jt.ones(dim)

    def _norm(self, x):
        return x * jt.rsqrt(x.pow(2).mean(-1, keepdims=True) + self.eps)

    def execute(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (jt.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = jt.arange(end) # Jittor 自动处理设备
    freqs = jt.outer(t, freqs).float()
    freqs_cos = jt.cos(freqs)
    freqs_sin = jt.sin(freqs)
    return freqs_cos, freqs_sin


def reshape_for_broadcast(freqs_cis: jt.Var, x: jt.Var):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1],
                               x.shape[-1]), f"Shape mismatch: {freqs_cis.shape} vs {(x.shape[1], x.shape[-1])}"
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(shape)


def apply_rotary_emb(
        xq: jt.Var,
        xk: jt.Var,
        freqs_cos: jt.Var,
        freqs_sin: jt.Var
) -> Tuple[jt.Var, jt.Var]:
    # 将输入张量重塑为复数表示
    xq_ = xq.float().reshape(xq.shape[:-1] + (-1, 2))
    xq_r, xq_i = xq_[..., 0], xq_[..., 1]
    xk_ = xk.float().reshape(xk.shape[:-1] + (-1, 2))
    xk_r, xk_i = xk_[..., 0], xk_[..., 1]

    # 调整旋转频率张量形状以便广播
    freqs_cos = reshape_for_broadcast(freqs_cos, xq_r)
    freqs_sin = reshape_for_broadcast(freqs_sin, xq_r)

    # 应用旋转
    xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin
    xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos
    xk_out_r = xk_r * freqs_cos - xk_i * freqs_sin
    xk_out_i = xk_r * freqs_sin + xk_i * freqs_cos

    # 合并结果
    xq_out = jt.stack([xq_out_r, xq_out_i], dim=-1).flatten(3)
    xk_out = jt.stack([xk_out_r, xk_out_i], dim=-1).flatten(3)

    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: jt.Var, n_rep: int) -> jt.Var:
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x.reindex_reduce(
            'add',
            [bs, slen, n_kv_heads * n_rep, head_dim],
            ['i0', 'i1', f'i2/{n_rep}', 'i3'],
            extras=[x.shape],
        )
    )


class Attention(Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        model_parallel_size = 1
        self.n_local_heads = args.n_heads // model_parallel_size
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads

        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.dropout = args.dropout

        # Jittor 目前不支持 Flash Attention，使用手动实现
        mask = jt.full((1, 1, args.max_seq_len, args.max_seq_len), float("-inf"))
        mask = jt.triu(mask, diagonal=1)
        self.mask = mask

    def execute(
            self,
            x: jt.Var,
            freqs_cos: jt.Var,
            freqs_sin: jt.Var,
    ):
        bsz, seqlen, _ = x.shape

        # 计算 Q, K, V
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        # 应用旋转位置编码
        xq, xk = apply_rotary_emb(xq, xk, freqs_cos, freqs_sin)

        # 重复 K 和 V 以匹配多头数量
        xk = repeat_kv(xk, self.n_rep)
        xv = repeat_kv(xv, self.n_rep)

        # 调整维度顺序以进行注意力计算
        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        # 手动实现注意力机制
        scores = jt.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
        mask = self.mask[:, :, :seqlen, :seqlen]
        scores = scores + mask
        scores = jt.nn.softmax(scores.float(), dim=-1).type_as(xq)
        scores = self.attn_dropout(scores)
        output = jt.matmul(scores, xv)  # (bs, n_local_heads, seqlen, head_dim)

        # 恢复原始维度顺序并拼接头部
        output = output.transpose(1, 2).reshape(bsz, seqlen, -1)

        # 最终投影
        output = self.wo(output)
        output = self.resid_dropout(output)
        return output


class FeedForward(Module):
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int, dropout: float):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def execute(self, x):
        return self.dropout(self.w2(nn.silu(self.w1(x)) * self.w3(x)))


class TransformerBlock(Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            dropout=args.dropout,
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def execute(self, x, freqs_cos, freqs_sin):
        h = x + self.attention(self.attention_norm(x), freqs_cos, freqs_sin)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class Transformer(Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)
        self.dropout = nn.Dropout(params.dropout)
        self.layers = nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))
        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = nn.Linear(params.dim, params.vocab_size, bias=False)

        # 权重绑定
        self.tok_embeddings.weight = self.output.weight

        # 预计算旋转位置编码
        freqs_cos, freqs_sin = precompute_freqs_cis(
            self.params.dim // self.params.n_heads,
            self.params.max_seq_len
        )
        self.freqs_cos = freqs_cos
        self.freqs_sin = freqs_sin

        # 初始化权重
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('w3.weight') or pn.endswith('wo.weight'):
                jt.init.gauss_(p, 0.0, 0.02 / math.sqrt(2 * params.n_layers))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            jt.init.gauss_(module.weight, 0.0, 0.02)
            if hasattr(module, 'bias') and module.bias is not None:
                jt.init.constant_(module.bias, 0.0)
        elif isinstance(module, nn.Embedding):
            jt.init.gauss_(module.weight, 0.0, 0.02)

    def execute(self, tokens: jt.Var, targets: Optional[jt.Var] = None) -> jt.Var:
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        h = self.dropout(h)

        freqs_cos = self.freqs_cos[:seqlen]
        freqs_sin = self.freqs_sin[:seqlen]

        for layer in self.layers:
            h = layer(h, freqs_cos, freqs_sin)
        h = self.norm(h)

        logits = self.output(h)

        return logits

    def configure_optimizers(self, weight_decay, learning_rate, betas):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        decay_params = [p for n, p in param_dict.items() if p.ndim >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.ndim < 2]

        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]

        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")

        # Jittor 使用 AdamW 优化器
        optimizer = jt.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        N = sum(p.numel() for p in self.parameters())
        cfg = self.params
        L, H, Q, T = cfg.n_layers, cfg.n_heads, cfg.dim // cfg.n_heads, cfg.max_seq_len
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        flops_achieved = flops_per_iter * (1.0 / dt)
        flops_promised = 312e12
        mfu = flops_achieved / flops_promised
        return mfu

    @jt.no_grad()
    def generate(self, idx, eos, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.shape[1] <= self.params.max_seq_len else idx[:, -self.params.max_seq_len:]
            logits = self(idx_cond)
            logits = logits[:, -1, :]

            if temperature == 0.0:
                _, idx_next = jt.topk(logits, k=1, dim=-1)
            else:
                logits = logits / temperature
                if top_k is not None:
                    v, _ = jt.topk(logits, min(top_k, logits.shape[-1]))
                    logits = jt.ternary(logits < v[:, -1:], -float('inf'), logits)
                probs = nn.softmax(logits, dim=-1)
                idx_next = jt.multinomial(probs, num_samples=1)

            idx = jt.concat([idx, idx_next], dim=1)
            if idx_next == eos:
                break

        return idx

    @jt.no_grad()
    def export(self, filepath='model.bin'):
        f = open(filepath, 'wb')

        def serialize(t):
            # Jittor 变量转为 numpy 数组
            d = t.numpy().astype(np.float32).flatten()
            b = struct.pack(f'{len(d)}f', *d)
            f.write(b)

        # 写入头信息
        hidden_dim = self.layers[0].feed_forward.w1.weight.shape[0]
        p = self.params
        n_kv_heads = p.n_heads if p.n_kv_heads is None else p.n_kv_heads
        header = struct.pack('iiiiiii', p.dim, hidden_dim, p.n_layers, p.n_heads,
                             n_kv_heads, p.vocab_size, p.max_seq_len)
        f.write(header)

        # 嵌入权重
        serialize(self.tok_embeddings.weight)

        # 各层权重
        for layer in self.layers:
            serialize(layer.attention_norm.weight)
        for layer in self.layers:
            serialize(layer.attention.wq.weight)
        for layer in self.layers:
            serialize(layer.attention.wk.weight)
        for layer in self.layers:
            serialize(layer.attention.wv.weight)
        for layer in self.layers:
            serialize(layer.attention.wo.weight)
        for layer in self.layers:
            serialize(layer.ffn_norm.weight)
        for layer in self.layers:
            serialize(layer.feed_forward.w1.weight)
        for layer in self.layers:
            serialize(layer.feed_forward.w2.weight)
        for layer in self.layers:
            serialize(layer.feed_forward.w3.weight)

        # 最终归一化
        serialize(self.norm.weight)

        # 位置编码
        serialize(self.freqs_cos[:p.max_seq_len])
        serialize(self.freqs_sin[:p.max_seq_len])

        f.close()
        print(f"wrote {filepath}")