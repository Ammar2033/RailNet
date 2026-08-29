"""Train in the rail basis (ADR 0001, the parallel bet).

Post-hoc, a normally-trained transformer's route-id map has ~12 bits/weight of
entropy and no rail locality (`structured_routing_probe.py`) — so options A/A'
are dead. The one way back to a storage win: **train** the weights so they are
naturally cheap in the rail basis.

This is the minimal test. A tiny char-LM, three ways:
  1. dense           - normal Linear layers (reference loss)
  2. rail            - every Linear is W = ternary(g) @ rails, straight-through
  3. rail + entropy  - same, plus a penalty on per-output-row rail-usage entropy

Then measure, for the rail models: can it fit the task, and what is the
route-map cost (distinct routes, bits/weight at min width) vs the 16 bits a
post-hoc compile needs.

    pip install -e ".[hf]"      # for torch
    python research/train_rail_basis.py --steps 1500
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
TEXT = (
    "railnet replaces dense runtime weight arrays with shared primitive rails plus topology "
    "plus routing while preserving the exact same mathematical information. the rail is a "
    "programmable shared numeric primitive. the same fabric can execute different models by "
    "reprogramming rail values and routing. correctness first then compact then fast then "
    "physical. do not throw away the weight change how it is represented routed and shared. "
) * 6


def _ste_ternary(g, max_terms: int | None):
    """Trained-ternary coeff in {-1,0,+1}, straight-through. With ``max_terms``,
    keep only the top-|g| terms per weight (RailNet's route budget)."""
    import torch

    t = torch.tanh(g)
    delta = 0.7 * t.abs().mean(dim=(0, 1), keepdim=True)
    keep = t.abs() > delta
    if max_terms is not None and t.shape[-1] > max_terms:
        thr = t.abs().topk(max_terms, dim=-1).values[..., -1:]
        keep = keep & (t.abs() >= thr)
    hard = torch.sign(t) * keep.float()
    return hard + (t - t.detach())  # STE


def main() -> int:
    import torch
    import torch.nn.functional as F
    from torch import nn

    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--n-rails", type=int, default=24)
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--block", type=int, default=48)
    ap.add_argument("--entropy-weight", type=float, default=3e-3)
    ap.add_argument("--l1", type=float, default=1e-4)
    ap.add_argument("--max-terms", type=int, default=4)
    args = ap.parse_args()
    torch.manual_seed(0)

    chars = sorted(set(TEXT))
    stoi = {c: i for i, c in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in TEXT], dtype=torch.long)
    V, B, T = len(chars), 24, args.block

    def batch():
        ix = torch.randint(0, len(data) - T - 1, (B,))
        x = torch.stack([data[i : i + T] for i in ix])
        y = torch.stack([data[i + 1 : i + 1 + T] for i in ix])
        return x, y

    class RailLinear(nn.Module):
        def __init__(self, i, o):
            super().__init__()
            k = 1.0 / math.sqrt(i)
            self.rails = nn.Parameter(torch.linspace(-k, k, args.n_rails))
            self.g = nn.Parameter(torch.randn(o, i, args.n_rails) * 0.8)
            self.bias = nn.Parameter(torch.zeros(o))
            self.scale = nn.Parameter(torch.ones(o, 1))

        def _coeff(self):
            return _ste_ternary(self.g, args.max_terms)  # (o, i, n_rails) in {-1,0,+1}

        def weight(self):
            return self.scale * (self._coeff() @ self.rails)

        def forward(self, x):
            return x @ self.weight().T + self.bias

        def entropy_penalty(self):
            use = (self._coeff().abs()).sum(dim=1)  # (o, n_rails) per-row rail usage
            p = use / use.sum(dim=1, keepdim=True).clamp_min(1e-6)
            return -(p * (p + 1e-9).log()).sum(dim=1).mean()

        def route_stats(self):
            pat = self._coeff().reshape(-1, args.n_rails).detach()
            rows = [tuple(int(v) for v in r) for r in pat.tolist()]
            terms = float(pat.abs().sum(dim=1).mean())
            return len(set(rows)), terms

    class TinyLM(nn.Module):
        def __init__(self, rail: bool):
            super().__init__()
            d = args.d_model
            self.tok = nn.Embedding(V, d)
            self.pos = nn.Embedding(T, d)
            Lin = RailLinear if rail else nn.Linear
            self.q, self.k, self.v, self.o = (Lin(d, d) for _ in range(4))
            self.f1, self.f2 = Lin(d, 2 * d), Lin(2 * d, d)
            self.head = nn.Linear(d, V)
            self.rail = rail

        def forward(self, x):
            d = args.d_model
            h = self.tok(x) + self.pos(torch.arange(x.size(1)))
            q, k, v = self.q(h), self.k(h), self.v(h)
            att = (q @ k.transpose(-2, -1)) / math.sqrt(d)
            mask = torch.triu(torch.ones(x.size(1), x.size(1)), 1).bool()
            att = att.masked_fill(mask, float("-inf")).softmax(-1)
            h = h + self.o(att @ v)
            h = h + self.f2(F.gelu(self.f1(h)))
            return self.head(h)

        def rail_layers(self):
            return [m for m in self.modules() if isinstance(m, RailLinear)]

    def train(rail: bool, entropy_w: float):
        m = TinyLM(rail)
        opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
        for step in range(args.steps):
            x, y = batch()
            logits = m(x)
            loss = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1))
            if rail:
                ent = sum(lyr.entropy_penalty() for lyr in m.rail_layers())
                l1 = sum(lyr.g.abs().mean() for lyr in m.rail_layers())
                loss = loss + entropy_w * ent + args.l1 * l1
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            x, y = batch()
            val = F.cross_entropy(m(x).reshape(-1, V), y.reshape(-1)).item()
        report = {"val_loss": round(val, 3)}
        if rail:
            n_params = sum(lyr.g.shape[0] * lyr.g.shape[1] for lyr in m.rail_layers())
            n_routes = sum(lyr.route_stats()[0] for lyr in m.rail_layers())
            terms = float(np.mean([lyr.route_stats()[1] for lyr in m.rail_layers()]))
            bits_per_weight = math.ceil(math.log2(max(2, n_routes)))
            report |= {
                "distinct_routes": int(n_routes),
                "avg_terms": round(terms, 2),
                "route_map_bits_per_weight": bits_per_weight,
                "vs_posthoc_16bit": round(bits_per_weight / 16, 3),
                "n_rail_params": int(n_params),
            }
        return report

    out = {
        "dense": train(rail=False, entropy_w=0.0),
        "rail": train(rail=True, entropy_w=0.0),
        "rail_entropy": train(rail=True, entropy_w=args.entropy_weight),
    }
    out["reads_as"] = (
        "rail val_loss near dense => rail basis is trainable. "
        "route_map_bits_per_weight << 16 (esp. with entropy) => training CAN buy the "
        "storage win that post-hoc compilation cannot."
    )
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "train_rail_basis.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
