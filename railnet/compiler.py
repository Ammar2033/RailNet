"""Lossless basis compilation (delegates to proven pipeline).

Default fabric: 96 rails / 4 terms (validated on Layer-0).
Escalation ladder for rare failures: 96 -> 128 -> 192.
"""
import time

from . import bf16 as B
from . import safetensors_reader as SR

RN = B.RN

# Fast meet-in-the-middle exhaustive evaluator (from the
# global-basis experiment) speeds every repair compile.
def _load_fast():
    import importlib.util
    from pathlib import Path
    p = Path(__file__).resolve().parent.parent / (
        "11_global_layer0_shared_basis.py"
    )
    spec = importlib.util.spec_from_file_location(
        "rn_global_fast", str(p)
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


FAST = _load_fast()          # import also monkeypatches RN
compile_exact = FAST.FAST_COMPILE if hasattr(FAST, "FAST_COMPILE") else (
    RN.compile_exact_routes_exhaustive
)

RN.MAX_ITERS = 4              # init+repairs carry the work


DEFAULT_RAILS = 96
TERMS = 4
LADDER = [96, 128, 192]


def analyze(raw):
    return RN.analyze_unique_values(raw)


def initialize(vals, bits, counts, rails):
    return RN.initialize_rails(vals, bits, counts, rails)


def learn(vals, bits, counts, rails, terms=TERMS):
    return RN.learn_basis(vals, bits, counts, rails, terms)


def exact_count(bits_arr, rails_arr, terms=TERMS):
    table = compile_exact(bits_arr, rails_arr, terms)
    cov = sum(1 for b in bits_arr if int(b) in table)
    return cov, table


def compile_tensor_lossless(raw, name="", log=None):
    """
    Returns dict:
      status PASS|FAILED, rails, terms, table, rails_arr,
      exact, unique, attempts[]
    Ascending ladder; stops at first lossless config.
    """
    bits, counts, vals = analyze(raw)
    n_uniq = len(bits)
    attempts = []

    for rc in LADDER:
        t0 = time.perf_counter()
        learned = learn(vals, bits, counts, rc)
        secs = time.perf_counter() - t0
        cov, table = exact_count(bits, learned["rails"])
        ok = cov == n_uniq
        attempts.append({
            "rails": rc,
            "exact": int(cov),
            "unique": int(n_uniq),
            "lossless": bool(ok),
            "seconds": round(secs, 1),
        })
        if log:
            log(f"[{name}] rails={rc} exact={cov}/{n_uniq}"
                f" ({'LOSSLESS' if ok else ''}) [{secs:.0f}s]")

        if ok:
            return {
                "status": "PASS",
                "rails": rc,
                "terms": TERMS,
                "table": table,
                "rails_arr": learned["rails"],
                "exact": int(cov),
                "unique": int(n_uniq),
                "attempts": attempts,
            }

    return {
        "status": "FAILED",
        "attempts": attempts,
        "unique": int(n_uniq),
    }
