"""Profile paged_decode_b64_bs16_d128_gqa4 for nsys/ncu."""
import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))
    sys.path.insert(0, str(_ROOT))

import flag_gems
from benchmark.test_flash_attn_varlen_fa3_func import _pr4494_all_workloads, _pr4494_make_input

CASE_NAME = "paged_decode_b64_bs16_d128_gqa4"
MODE = sys.argv[1] if len(sys.argv) > 1 else "gluon"  # gluon | fa3
N_REPS = int(sys.argv[2]) if len(sys.argv) > 2 else 20

workloads = _pr4494_all_workloads()
wl = next(w for w in workloads if w.shape.name == CASE_NAME)
idx = workloads.index(wl)

inp = _pr4494_make_input(wl, torch.float16, "cuda", seed=2026 + idx)
args, kwargs = inp[:-1], inp[-1]
use_gluon = MODE == "gluon"
kwargs = {**kwargs, "fa_version": 3, "use_gluon": use_gluon}

# Warmup
for _ in range(3):
    flag_gems.ops.flash_attn_varlen_func(*args, **kwargs)
torch.cuda.synchronize()

for _ in range(N_REPS):
    flag_gems.ops.flash_attn_varlen_func(*args, **kwargs)
torch.cuda.synchronize()
print(f"done {CASE_NAME} mode={MODE} use_gluon={use_gluon} reps={N_REPS}")