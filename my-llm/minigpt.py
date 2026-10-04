"""
MiniGPT — a working language model built from scratch in PyTorch.
Run:  python minigpt.py
It trains on a small text corpus and then generates text.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)

# ---------------- 1. Tokenizer (character-level) ----------------
class CharTokenizer:
    def __init__(self, text):
        chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.itos = {i: c for i, c in enumerate(chars)}
        self.vocab_size = len(chars)

    def encode(self, s):
        return [self.stoi[c] for c in s]

    def decode(self, ids):
        return "".join(self.itos[i] for i in ids)

# ---------------- 2. Model blocks ----------------
class CausalSelfAttention(nn.Module):
    """Multi-head self-attention: each token looks at previous tokens only."""
    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        self.n_head = n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd)
        self.proj = nn.Linear(n_embd, n_embd)
        mask = torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size)
        self.register_buffer("mask", mask)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)

class Block(nn.Module):
    """One transformer layer: attention + MLP with residual connections."""
    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))   # residual connection
        x = x + self.mlp(self.ln2(x))    # residual connection
        return x

class MiniGPT(nn.Module):
    def __init__(self, vocab_size, n_embd=128, n_head=4, n_layer=4, block_size=128):
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.blocks = nn.ModuleList([Block(n_embd, n_head, block_size) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=200, temperature=0.8, top_k=20):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, 1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx

# ---------------- 3. Training + Test ----------------
def train():
    # Small training corpus (later: replace with any .txt file)
    text = ("the quick brown fox jumps over the lazy dog. " * 20
            + "hello world this is a small language model. " * 20
            + "artificial intelligence learns patterns in data. " * 20)

    tok = CharTokenizer(text)
    data = torch.tensor(tok.encode(text), dtype=torch.long)

    block_size = 64
    def get_batch(batch_size=32):
        ix = torch.randint(len(data) - block_size - 1, (batch_size,))
        x = torch.stack([data[i:i + block_size] for i in ix])
        y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
        return x, y

    model = MiniGPT(tok.vocab_size, n_embd=128, n_head=4, n_layer=4, block_size=block_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    print(f"Vocab size: {tok.vocab_size} | Params: {sum(p.numel() for p in model.parameters()):,}")
    losses = []
    for step in range(500):
        x, y = get_batch()
        _, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if step % 100 == 0:
            print(f"step {step:4d} | loss {loss.item():.4f}")

    # ---- TEST CASES ----
    assert losses[-1] < losses[0], "Model should learn (loss must decrease)"
    assert losses[-1] < 1.5, f"Loss too high: {losses[-1]}"

    # Generate text from a prompt
    prompt = torch.tensor([tok.encode("the ")])
    out = model.generate(prompt, max_new_tokens=120)
    print("\n--- Generated text ---")
    print(tok.decode(out[0].tolist()))
    print("\nAll tests passed - you just trained your first LLM!")

if __name__ == "__main__":
    train()
