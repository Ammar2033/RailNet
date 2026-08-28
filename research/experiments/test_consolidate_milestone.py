import json
from pathlib import Path

COMPILED = Path("compiled/layer0")
RESULTS = Path("results")

SHORT = ["q_proj", "k_proj", "v_proj", "o_proj"]

SHAPE_PARAMS = {
    "q_proj": 1179648,
    "k_proj": 294912,
    "v_proj": 294912,
    "o_proj": 1179648,
}

results = []

for s in SHORT:
    art = json.load(open(COMPILED / f"{s}_lossless.json"))
    v = art["validation"]
    results.append({
        "tensor": art["tensor"],
        "short": s,
        "shape": art["shape"],
        "parameters": art["parameters"],
        "unique_values": v["unique_values"],
        "exact_unique": v["exact_unique"],
        "rails": art["rail_count"],
        "terms": art["max_terms"],
        "pass": (
            v["weight_reconstruction"] == "PASS"
            and v["full_tensor_routed"]
            and v["math_oracle_exact"]
        ),
        "artifact_file": str(COMPILED / f"{s}_lossless.json"),
        "artifact_bytes": (COMPILED / f"{s}_lossless.json").stat().st_size,
    })

total_params = sum(r["parameters"] for r in results)
total_uniq = sum(r["unique_values"] for r in results)

milestone = {
    "milestone": "attention_layer0_lossless",
    "timestamp": __import__("time").strftime("%Y-%m-%dT%H:%M:%S"),
    "seed": 42,
    "results": results,
    "totals": {
        "tensors": len(results),
        "parameters": total_params,
        "unique_values": total_uniq,
        "all_pass": all(r["pass"] for r in results),
    },
    "verdict": "PASS" if all(r["pass"] for r in results) else "INCOMPLETE",
}

out = RESULTS / "milestone_attention.json"
json.dump(milestone, open(out, "w"), indent=2)
print(f"Saved {out}")
for r in results:
    print(f"  {r['short']:8s} rails={r['rails']:3d} terms={r['terms']} "
          f"exact={r['exact_unique']}/{r['unique_values']} pass={r['pass']}")
