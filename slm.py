# -*- coding: utf-8 -*-
import os
import subprocess
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import tiktoken
from tqdm.auto import tqdm
from contextlib import nullcontext
from dataclasses import dataclass
from datasets import Dataset
from huggingface_hub import login
from torch.optim.lr_scheduler import LinearLR, SequentialLR, CosineAnnealingLR

# --- Configuration Section ---
@dataclass
class GPTConfig:
    block_size: int
    vocab_size: int
    n_layer: int
    n_head: int
    n_embd: int
    dropout: float = 0.0
    bias: bool = True

# --- Model Architecture ---
class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.flash = hasattr(F, 'scaled_dot_product_attention')
        if not self.flash:
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                       .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        if self.flash:
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.attn_dropout.p if self.training else 0.0, is_causal=True)
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
    def forward(self, x):
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = LayerNorm(config.n_embd, config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln2 = LayerNorm(config.n_embd, config.bias)
        self.mlp = MLP(config)
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=LayerNorm(config.n_embd, config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            return logits, loss
        else:
            logits = self.lm_head(x[:, [-1], :])
            return logits, None

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# --- Helper Functions ---
def get_batch(split: str, batch_size: int, block_size: int, device: str):
    filename = 'train.bin' if split == 'train' else 'val.bin'
    data = np.memmap(filename, dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i+block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i+1:i+1+block_size].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)

def estimate_loss(model, eval_iters: int, batch_size: int, block_size: int, device: str):
    model.eval()
    losses = {}
    with torch.no_grad():
        for split in ['train', 'val']:
            losses[split] = 0
            for _ in range(eval_iters):
                X, Y = get_batch(split, batch_size, block_size, device)
                _, loss = model(X, Y)
                losses[split] += loss.item()
            losses[split] /= eval_iters
    model.train()
    return losses

# --- Main Execution ---
if __name__ == '__main__':
    # Dataset setup
    login(token="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")
    if not os.path.exists("ai-medical-chatbot"):
        subprocess.run(["git", "clone", "https://huggingface.co/datasets/ruslanmv/ai-medical-chatbot"], check=True)

    # Load and prepare data
    df = pd.read_parquet("ai-medical-chatbot/dialogues.parquet")
    ds = Dataset.from_pandas(df)
    
    # Tokenization setup
    enc = tiktoken.get_encoding("gpt2")
    split_ds = ds.train_test_split(test_size=0.1, seed=42)
    
    def combine_fields(ex):
        return {"text": f"<bos> {ex['Description']} {ex['Patient']} {ex['Doctor']} <eos>"}
    
    def process(ex):
        return {'ids': enc.encode_ordinary(ex['text']), 'len': len(ex['text'])}

    # Process datasets
    for name, dataset in [('train', split_ds['train']), ('val', split_ds['test'])]:
        dataset = dataset.map(combine_fields)
        dataset = dataset.map(process, remove_columns=['text'])
        total_tokens = sum(dataset['len'])
        arr = np.memmap(f'{name}.bin', dtype=np.uint16, mode='w+', shape=(total_tokens,))
        idx = 0
        for batch in dataset:
            tokens = np.array(batch['ids'], dtype=np.uint16)
            arr[idx:idx+len(tokens)] = tokens
            idx += len(tokens)
        arr.flush()

    # Training configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = GPTConfig(
        vocab_size=50257,
        block_size=128,
        n_layer=6,
        n_head=6,
        n_embd=384,
        dropout=0.1,
        bias=True
    )
    model = GPT(config).to(device)
    
    # Training parameters
    learning_rate = 1e-4
    max_iters = 20000
    warmup_steps = 1000
    min_lr = 5e-4
    eval_iters = 500
    batch_size = 32
    block_size = 128
    gradient_accumulation_steps = 32
    dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
    ptdtype = torch.bfloat16 if dtype == 'bfloat16' else torch.float16
    ctx = nullcontext() if device == 'cpu' else torch.amp.autocast(device_type=device, dtype=ptdtype)
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))
    
    # Initialize optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.95), weight_decay=0.1, eps=1e-9)
    scheduler_warmup = LinearLR(optimizer, total_iters=warmup_steps)
    scheduler_decay = CosineAnnealingLR(optimizer, T_max=max_iters - warmup_steps, eta_min=min_lr)
    scheduler = SequentialLR(optimizer, schedulers=[scheduler_warmup, scheduler_decay], milestones=[warmup_steps])
    
    # Training loop
    best_val_loss = float('inf')
    for epoch in tqdm(range(max_iters)):
        if epoch % eval_iters == 0 and epoch > 0:
            losses = estimate_loss(model, eval_iters, batch_size, block_size, device)
            print(f"Epoch {epoch}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            print(f"Current learning rate: {optimizer.param_groups[0]['lr']:.5f}")
            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                torch.save(model.state_dict(), "best_model_params.pt")
        
        # Training step
        X, Y = get_batch("train", batch_size, block_size, device)
        with ctx:
            _, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps
            scaler.scale(loss).backward()
        
        if (epoch + 1) % gradient_accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()