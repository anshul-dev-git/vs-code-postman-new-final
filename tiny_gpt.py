import os
import requests
import torch
import torch.nn as nn
from torch.nn import functional as F
from src import get_dense_attention, get_sliding_window_mask, get_bigbird_mask

# 1. inline dataset loading 
if not os.path.exists('input.txt'):
    print("grabbing tinyshakespeare...")
    try:
        r = requests.get('https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt')
        with open('input.txt', 'w', encoding='utf-8') as f:
            f.write(r.text)
    except Exception as e:
        print(f"network died, can't download data: {e}")
        exit(1)

with open('input.txt', 'r', encoding='utf-8') as f:
    raw_text = f.read()

uniq_chars = sorted(list(set(raw_text)))
v_size = len(uniq_chars)

# c2i map
c2i = {ch: i for i, ch in enumerate(uniq_chars)}
full_data = torch.tensor([c2i[c] for c in raw_text], dtype=torch.long)
# dropping val split entirely because we only need train loss for 1.6
train_data = full_data[:int(0.9 * len(full_data))] 

# 2. condensed network defs
class CustomAttnWrap(nn.Module):
    def __init__(self, emb, heads):
        super().__init__()
        self.heads = heads
        self.h_dim = emb // heads
        self.qkv = nn.Linear(emb, emb * 3)
        self.proj = nn.Linear(emb, emb)

    def forward(self, x, mask=None):
        b, t, c = x.shape
        qkv = self.qkv(x).reshape(b, t, 3, self.heads, self.h_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        
        # using the manual function
        out, _ = get_dense_attention(q, k, v, mask=mask) 
        # fix tensor shape layout before returning
        out = out.transpose(1, 2).contiguous().view(b, t, c)
        return self.proj(out)
class SmolBlock(nn.Module):
    def __init__(self, emb, heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(emb)
        self.attn = CustomAttnWrap(emb, heads)
        self.ln2 = nn.LayerNorm(emb)
        self.mlp = nn.Sequential(
            nn.Linear(emb, 4 * emb),
            nn.ReLU(),
            nn.Linear(4 * emb, emb)
        )

    def forward(self, x, mask=None):
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.mlp(self.ln2(x))
        return x

class SmolGPT(nn.Module):
    # pass params properly so the class is actually reusable
    def __init__(self, vocab_dim, embed_dim, num_heads, num_layers):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_dim, embed_dim)
        # keeping seq_len fixed to 256 since that's our training chunk size
        self.pos_emb = nn.Embedding(256, embed_dim) 
        
        self.blocks = nn.ModuleList([SmolBlock(embed_dim, num_heads) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_dim)

    def forward(self, idx, mask=None):
        b, t = idx.shape
        pos = torch.arange(0, t, dtype=torch.long, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        
        for block in self.blocks:
            x = block(x, mask)
            
        return self.head(self.ln_f(x))
def hacky_train_loop(net, train_data, mask, name, dev):
    print(f"\n--- hammering the {name} variant ---")
    net.to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
    
    last_loss = 0
    # sprint is ending soon, just doing 300 steps to prove the loss diff
    for step in range(300):
        # random chunk
        ix = torch.randint(len(train_data) - 256, (32,))
        x = torch.stack([train_data[i:i+256] for i in ix]).to(dev)
        y = torch.stack([train_data[i+1:i+256+1] for i in ix]).to(dev)
        out = net(x, mask)
        b, t, c = out.shape
        loss = F.cross_entropy(out.view(b*t, c), y.view(b*t))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 50 == 0:
            print(f"step {step} -> loss {loss.item():.4f}")
        last_loss = loss.item()    
    return last_loss
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"running on: {device}")

    # pre-compute masks for a 256 context window
    m_sliding = get_sliding_window_mask(256, 32).to(device)
    m_bigbird = get_bigbird_mask(256, 32, num_global=4, random_prob=0.05).to(device)

    # deliverable 1.6 requires a 2-layer network
    emb_dim = 128
    n_heads = 4
    n_layers = 2

    print("\n[SPAWNING NETWORKS]")
    loss_d = hacky_train_loop(SmolGPT(v_size, emb_dim, n_heads, n_layers), train_data, None, "dense_ref", device)
    loss_s = hacky_train_loop(SmolGPT(v_size, emb_dim, n_heads, n_layers), train_data, m_sliding, "sliding_window", device)
    loss_b = hacky_train_loop(SmolGPT(v_size, emb_dim, n_heads, n_layers), train_data, m_bigbird, "bigbird", device)
    print(f"\n[FINAL DELIVERABLE 1.6 LOSS]")
    print(f"dense:   {loss_d:.4f}")
    print(f"bigbird: {loss_b:.4f}")
    print(f"sliding: {loss_s:.4f}")