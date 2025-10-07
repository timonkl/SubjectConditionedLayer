import math
import inspect

import torch
import torch.nn as nn
from torch.nn import functional as F


class LearningRateScheduler:
    """
    inspired by https://github.com/karpathy/nanoGPT/blob/master/train.py
    MIT License Copyright (c) 2022 Andrej Karpathy
    """

    def __init__(
        self, warmup_iters=150, learning_rate=3e-4, lr_decay_iters=1500, min_lr=3e-5
    ):
        self.warmup_iters = warmup_iters
        self.learning_rate = learning_rate
        self.lr_decay_iters = lr_decay_iters
        # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
        self.min_lr = min_lr

    def get_lr(self, iteration):
        # Epochs starts with 0
        iteration += 1
        # 1) linear warmup for warmup_iters steps
        if iteration < self.warmup_iters:
            return self.learning_rate * iteration / self.warmup_iters
        # 2) if it > lr_decay_iters, return min learning rate
        if iteration > self.lr_decay_iters:
            return self.min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (iteration - self.warmup_iters) / (
            self.lr_decay_iters - self.warmup_iters
        )
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
        return self.min_lr + coeff * (self.learning_rate - self.min_lr)

    def state_dict(self):
        state_dict = {
            "warmup_iters": self.warmup_iters,
            "learning_rate": self.learning_rate,
            "lr_decay_iters": self.lr_decay_iters,
            "min_lr": self.min_lr,
        }
        return state_dict

    def load_state_dict(self, state_dict):
        self.warmup_iters = state_dict["warmup_iters"]
        self.learning_rate = state_dict["learning_rate"]
        self.lr_decay_iters = state_dict["lr_decay_iters"]
        self.min_lr = state_dict["min_lr"]


class LayerNorm(nn.Module):
    """
    LayerNorm with optional bias
    https://arxiv.org/pdf/1607.06450.pdf
    """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x):
        return F.layer_norm(
            x,
            normalized_shape=self.weight.shape,
            weight=self.weight,
            bias=self.bias,
            eps=1e-5,
        )


class LoRAMHSA(nn.Module):
    """
    Multi-Head Self-Attention block
    """

    def __init__(
        self,
        d_model,
        n_head,
        bias,
        rank=8,
        alpha=1,
        num_adapters=10,
        dropout=0.0,
        flash_att=True,
    ):
        super().__init__()

        assert d_model % n_head == 0
        # key, query, value
        # self.attn = nn.Linear(d_model, 3 * d_model, bias=bias)
        self.attn = LoRALinearPerSubject(
            in_features=d_model,
            out_features=3 * d_model,
            rank=rank,
            alpha=alpha,
            num_adapters=num_adapters,
        )

        # self.query = nn.Linear(d_model, d_model, bias=bias)
        # self.key = nn.Linear(d_model, d_model, bias=bias)
        # self.value = nn.Linear(d_model, d_model, bias=bias)

        # self.proj = nn.Linear(d_model, d_model, bias=bias)
        self.proj = LoRALinearPerSubject(
            in_features=d_model,
            out_features=d_model,
            rank=rank,
            alpha=alpha,
            num_adapters=num_adapters,
        )

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.n_head = n_head
        self.d_model = d_model
        self.dropout = dropout
        self.flash_att = flash_att

    def forward(self, x, pos_embedding, subject_id):

        # batch size, sequence length, embedding dimensionality (d_model)
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch
        q, k, v = self.attn(x=x, subject_id=subject_id).split(self.d_model, dim=2)

        # q = self.query(x + pos_embedding)
        # k = self.key(x + pos_embedding)
        # v = self.value(x)

        q = q.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)

        if self.flash_att:
            y = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0,
                is_causal=False,
            )
        else:
            y = (
                self.attn_dropout(
                    F.softmax(
                        (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))),
                        dim=-1,
                    )
                )
                @ v
            )

        # re-assemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # output projection
        y = self.resid_dropout(self.proj(x=y, subject_id=subject_id))
        return y


#######################################################
class LoRALinearPerSubject(nn.Module):
    def __init__(self, in_features, out_features, rank=4, alpha=1.0, num_adapters=4):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.rank = rank
        self.alpha = alpha
        self.num_adapters = num_adapters

        # LoRA adapters as experts
        self.lora_A = nn.ModuleList(
            [nn.Linear(in_features, rank, bias=False) for _ in range(num_adapters)]
        )
        self.lora_B = nn.ModuleList(
            [nn.Linear(rank, out_features, bias=False) for _ in range(num_adapters)]
        )

        # Init B with zeros
        for a in self.lora_A:
            torch.nn.init.normal_(a.weight, mean=0.0, std=0.02)
        for b in self.lora_B:
            nn.init.zeros_(b.weight)
            # torch.nn.init.normal_(b.weight, mean=0.0, std=0.02)

    def forward(self, x, subject_id: torch.LongTensor):
        """
        x: batch size, sequence length, token dim
        subject_id: (batch_size,) ints in [0, num_subjects)
        """
        out = self.linear(x)

        # Apply correct adapter to each Subject
        lora_out = torch.zeros_like(out)
        for i in range(self.num_adapters):
            mask = subject_id == i
            if mask.any():
                lora_A_i = self.lora_A[i](x[mask])
                lora_B_i = self.lora_B[i](lora_A_i)
                lora_out[mask] = self.alpha / self.rank * lora_B_i

        return out + lora_out


class LoRAFeedForward(nn.Module):
    """
    Feed Forward block from Transformer
    """

    def __init__(
        self,
        d_model,
        dim_feedforward=None,
        rank=8,
        alpha=0,
        num_adapters=4,
        dropout=0.0,
    ):
        super().__init__()

        self.proj_in = LoRALinearPerSubject(
            in_features=d_model,
            out_features=dim_feedforward,
            rank=rank,
            alpha=alpha,
            num_adapters=num_adapters,
        )
        self.gelu = nn.GELU()
        self.proj = LoRALinearPerSubject(
            in_features=dim_feedforward,
            out_features=d_model,
            rank=rank,
            alpha=alpha,
            num_adapters=num_adapters,
        )
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)

    def forward(self, x, subject_id: torch.LongTensor):
        x = self.proj_in(x, subject_id)
        x = self.gelu(x)
        x = self.dropout1(x)
        x = self.proj(x, subject_id)
        x = self.dropout(x)
        return x


########################################################


class FeedForward(nn.Module):
    """
    Feed Forward block from Transformer
    """

    def __init__(self, d_model, dim_feedforward=None, dropout=0.0, bias=False):
        super().__init__()

        self.proj_in = nn.Linear(d_model, dim_feedforward, bias=bias)
        self.gelu = nn.GELU()
        self.proj = nn.Linear(dim_feedforward, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.proj_in(x)
        x = self.gelu(x)
        x = self.dropout1(x)
        x = self.proj(x)
        x = self.dropout(x)
        return x


class TransformerEncoderLayer(nn.Module):
    """
    Transformer model, based on 'Attention Is All You Need' -> https://arxiv.org/abs/1706.03762
    """

    def __init__(
        self,
        d_model,
        n_head,
        dropout=0.0,
        dim_feedforward=None,
        bias=False,
        alpha=1,
        rank=64,
        num_adapters=10,
        model_type="vanilla",
    ):
        super().__init__()

        if dim_feedforward is None:
            dim_feedforward = 4 * d_model
            print(
                "dim_feedforward is set to 4*d_model, the default in Vaswani et al. (Attention is all you need)"
            )

        self.layer_norm_att = LayerNorm(d_model, bias=bias)

        # self.mhsa = MHSA(d_model, n_head, bias, dropout=dropout, flash_att=True)
        self.mhsa = LoRAMHSA(
            d_model=d_model,
            n_head=n_head,
            bias=bias,
            rank=rank,
            alpha=alpha,
            num_adapters=num_adapters,
            dropout=0.0,
            flash_att=True,
        )

        self.layer_norm_ff = LayerNorm(d_model, bias=bias)

        self.model_type = model_type

        self.lora_FF = LoRAFeedForward(
            d_model=d_model,
            dim_feedforward=dim_feedforward,
            rank=rank,
            alpha=alpha,
            num_adapters=num_adapters,
            dropout=dropout,
        )

    def forward(self, x, subject_id, pos_embedding):
        topk_indices = None

        # Self-Attention
        x = x + self.mhsa(
            x=self.layer_norm_att(x), pos_embedding=pos_embedding, subject_id=subject_id
        )

        x = x + self.lora_FF(self.layer_norm_ff(x), subject_id)

        return x, topk_indices


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        n_blocks,
        d_model,
        n_head,
        dropout=0.0,
        bias=False,
        alpha=1,
        rank=64,
        num_adapters=10,
        model_type="vanilla",
    ):
        super().__init__()

        self.encoder_block = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    d_model=d_model,
                    n_head=n_head,
                    dropout=dropout,
                    dim_feedforward=None,
                    bias=bias,
                    alpha=alpha,
                    rank=rank,
                    num_adapters=num_adapters,
                    model_type=model_type,
                )
                for _ in range(n_blocks)
            ]
        )

        # GPT2 type of init -> Radford et al. 'Language Models are Unsupervised Multitask Learners'
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * n_blocks))

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, x, subject_id, pos_embedding):

        experts_chosen_list = []
        for block in self.encoder_block:
            x, experts_chosen = block(
                x, subject_id=subject_id, pos_embedding=pos_embedding
            )
            experts_chosen_list.append(experts_chosen)

        return x, experts_chosen_list


class PBT(nn.Module):
    def __init__(
        self,
        config,
        n_classes,
        num_embeddings,
        device,
        learnable_cls=False,
    ):
        super().__init__()

        self.model_type = config["model_type"]
        self.model_head_type = config["model_head_type"]

        self.linear_projection = LoRALinearPerSubject(
            in_features=config["d_input"],
            out_features=config["d_model"],
            rank=config["rank"],
            alpha=config["alpha"],
            num_adapters=config["num_adapters"],
        )

        if learnable_cls:
            self.cls_token = nn.Parameter(torch.randn(1, 1, config["d_model"]) * 0.002)
        else:
            self.cls_token = torch.full(
                size=(1, 1, config["d_model"]),
                fill_value=0,
                requires_grad=False,
                dtype=torch.float32,
                device=device,
            )

        # trainable parameters for the position embedding
        # lookup table that stores learnable positional embedding
        self.pos_embedding = nn.Embedding(
            num_embeddings=num_embeddings, embedding_dim=config["d_model"]
        )

        self.transformer_encoder = TransformerEncoder(
            n_blocks=config["num_transformer_blocks"],
            d_model=config["d_model"],
            n_head=config["num_heads"],
            dropout=config["dropout"],
            bias=config["bias_transformer"],
            alpha=config["alpha"],
            rank=config["rank"],
            num_adapters=config["num_adapters"],
            model_type=config["model_type"],
        )

        self.cls_head = LoRALinearPerSubject(
            in_features=config["d_model"],
            out_features=n_classes,
            rank=config["rank"],
            alpha=config["alpha"],
            num_adapters=config["num_adapters"],
        )

        # init all weights (linear_projection, cls_head )
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.002)

    def forward(self, x, pos, spatial_pos, temporal_pos, subject_id):
        # Linear Projection, Concatenate [CLS]-Token, add positional embedding
        # x = torch.cat((self.cls_token.expand(x.size(0), 1, -1), self.linear_projection(x)), dim=1)

        # Mask subject id => not subject adapter is activated!
        if self.model_type == "vanilla":
            subject_id.fill_(-100)

        x = self.linear_projection(x, subject_id)

        x += self.pos_embedding(pos)
        pos_embedding = 0

        # Transformer Encoder
        transformer_out, experts_chosen_list = self.transformer_encoder(
            x, subject_id=subject_id, pos_embedding=pos_embedding
        )

        if self.model_head_type == "vanilla":
            subject_id.fill_(-100)

        logits = self.cls_head(transformer_out[:, 0], subject_id)

        return (
            transformer_out,
            logits,
            experts_chosen_list,
        )

    def configure_optimizers(
        self, weight_decay, learning_rate, betas, device_type, weight_decay_cls_head=0.0
    ):
        """
        https://github.com/karpathy/nanoGPT/blob/master/model.py
        """

        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        cls_head = param_dict["cls_head.weight"]
        del param_dict["cls_head.weight"]

        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]

        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": [cls_head], "weight_decay": weight_decay_cls_head},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(
            f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
        )
        print(
            f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters"
        )
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas, **extra_args
        )
        print(f"using fused AdamW: {use_fused}")

        return optimizer
