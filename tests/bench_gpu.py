"""P1 Step 7: RTX 4060 算力实测校准（docs/05 §3 估算 → 实测）。

基准设计：2层Transformer编码器 d_model=64（~20万参数，P3 深度模型的典型规格）
- 训练：50 epoch，batch=256，seq_len=60，10 特征（合成数据，测算力而非数据）
- 推理：batch=256 单次 forward × 20 次取均值
- 对照：同模型 CPU（2线程）计时
线程纪律：OMP=2，GPU 训练 dataloader=0。
"""
import time

import torch
import torch.nn as nn

torch.set_num_threads(2)
torch.manual_seed(42)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
N, SEQ, FEAT, BATCH, EPOCHS = 20000, 60, 10, 256, 50


class TinyTS(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(FEAT, 64)
        layer = nn.TransformerEncoderLayer(d_model=64, nhead=4, dim_feedforward=128, batch_first=True, dropout=0.1)
        self.enc = nn.TransformerEncoder(layer, num_layers=2)
        self.head = nn.Linear(64, 1)

    def forward(self, x):
        return self.head(self.enc(self.proj(x))[:, -1]).squeeze(-1)


def bench(device: str):
    model = TinyTS().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    lossf = nn.MSELoss()
    X = torch.randn(N, SEQ, FEAT, device=device)
    y = torch.randn(N, device=device)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    model.train()
    for ep in range(EPOCHS):
        for i in range(0, N, BATCH):
            xb, yb = X[i:i + BATCH], y[i:i + BATCH]
            opt.zero_grad()
            loss = lossf(model(xb), yb)
            loss.backward()
            opt.step()
    train_s = time.time() - t0
    if device == "cuda":
        peak = torch.cuda.max_memory_allocated() / 1e9
        torch.cuda.synchronize()
    else:
        peak = 0.0
    # 推理
    model.eval()
    with torch.no_grad():
        torch.cuda.synchronize() if device == "cuda" else None
        t1 = time.time()
        for _ in range(20):
            model(X[:BATCH])
        torch.cuda.synchronize() if device == "cuda" else None
        infer_ms = (time.time() - t1) / 20 * 1000
    n_par = sum(p.numel() for p in model.parameters())
    return train_s, infer_ms, peak, n_par


print(f"设备: {torch.cuda.get_device_name(0) if DEV == 'cuda' else 'CPU'}")
for dev in (["cuda", "cpu"] if DEV == "cuda" else ["cpu"]):
    ts, ims, peak, npar = bench(dev)
    print(f"[{dev.upper()}] 参数量={npar:,} | 训练50epoch={ts:.1f}s | 单批推理={ims:.1f}ms | 峰值显存={peak:.2f}GB")
print("CALIB_DONE")
