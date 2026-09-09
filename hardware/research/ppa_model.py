"""Quantitative PPA (Power, Performance, Area), Silicon Sizing, and Fleet TCO Model:
RailNet (Reprogrammable ReRAM / Path B) vs Taalas (Fixed Metal ROM) vs SRAM.

Usage:
    python hardware/research/ppa_model.py

Outputs:
    results/taalas_vs_railnet_ppa.json
    Terminal report with comparative silicon metrics.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Semiconductor & PDK Technology Parameters (16nm / 22nm Class)
# ---------------------------------------------------------------------------
TECHNOLOGIES = {
    "taalas_rom": {
        "name": "Taalas Fixed ROM (Metal Etch)",
        "evidence_level": "THEORETICAL (Taalas Whitepaper Claim)",
        "reprogrammable": False,
        "storage_density_mbit_mm2": 120.0,  # 15.0 MB/mm2
        "read_energy_pj_per_bit": 0.08,      # 0.64 pJ/byte
        "compute_area_per_mac_um2": 450.0,   # Standard dense BF16 MAC
        "mask_nre_per_model_usd": 8_000_000, # Tapeout mask set per model
        "wafer_cost_usd": 6_000,             # 300mm wafer
    },
    "railnet_reram_academic": {
        "name": "RailNet ReRAM (Academic 3D Crossbar)",
        "evidence_level": "THEORETICAL (Academic Paper Projection)",
        "reprogrammable": True,
        "storage_density_mbit_mm2": 80.0,   # Modeled Estimate: 10.0 MB/mm2 (Academic/3D Crossbar)
        "read_energy_pj_per_bit": 0.20,      # Modeled estimate: 1.60 pJ/byte
        "compute_area_per_mac_um2": 60.0,
        "mask_nre_per_model_usd": 0,
        "base_mask_nre_usd": 8_500_000,
        "wafer_cost_usd": 6_500,
    },
    "railnet_reram_commercial": {
        "name": "RailNet ReRAM (Commercial 22nm Silicon)",
        "evidence_level": "SILICON-MEASURED (Foundry 22nm eReRAM IP)",
        "reprogrammable": True,
        "storage_density_mbit_mm2": 20.0,   # Silicon Measured: 2.5 MB/mm2 (Commercial 22nm eReRAM TSMC/UMC with decoders/pumps)
        "array_efficiency_pct": 48.0,       # Array/Macro efficiency (Sense amps, charge pumps, decoders)
        "read_energy_pj_per_bit": 0.55,      # Silicon measured: 4.40 pJ/byte (including SA & bitline parasitics)
        "compute_area_per_mac_um2": 60.0,
        "mask_nre_per_model_usd": 0,
        "base_mask_nre_usd": 8_500_000,
        "wafer_cost_usd": 6_800,
    },
    "mram_baseline": {
        "name": "STT-MRAM Baseline",
        "evidence_level": "SILICON-MEASURED (Foundry eMRAM IP)",
        "reprogrammable": True,
        "storage_density_mbit_mm2": 25.0,   # 3.125 MB/mm2
        "read_energy_pj_per_bit": 0.35,
        "compute_area_per_mac_um2": 60.0,
        "mask_nre_per_model_usd": 0,
        "base_mask_nre_usd": 9_000_000,
        "wafer_cost_usd": 7_200,
    },
    "sram_baseline": {
        "name": "Dense SRAM Baseline",
        "evidence_level": "SILICON-MEASURED (Foundry 16/22nm SRAM)",
        "reprogrammable": True,
        "storage_density_mbit_mm2": 2.5,    # 0.31 MB/mm2
        "read_energy_pj_per_bit": 0.10,
        "compute_area_per_mac_um2": 450.0,
        "mask_nre_per_model_usd": 0,
        "base_mask_nre_usd": 8_000_000,
        "wafer_cost_usd": 6_000,
    },
}

# ---------------------------------------------------------------------------
# Evaluated LLM Models
# ---------------------------------------------------------------------------
MODELS = {
    "gemma3_1b": {
        "name": "Gemma3 1B / Llama 1B Class",
        "params_billion": 1.0,
        "weights_bytes": 1.25 * 1024**3,   # 1.25 GB (~10 Gbit)
        "weights_bits": 1.25 * 1024**3 * 8,
        "active_macs_per_token": 2.0e9,    # ~2 MAC ops per parameter
        "sram_workspace_mm2": 15.0,        # KV-cache, activations, router tables
        "monolithic_feasible": True,
    },
    "llama_3b": {
        "name": "Llama-3.2 3B Class",
        "params_billion": 3.0,
        "weights_bytes": 3.75 * 1024**3,   # 3.75 GB (~30 Gbit)
        "weights_bits": 3.75 * 1024**3 * 8,
        "active_macs_per_token": 6.0e9,
        "sram_workspace_mm2": 35.0,
        "monolithic_feasible": True,
    },
    "llama_8b": {
        "name": "Llama-3.1 8B Class",
        "params_billion": 8.0,
        "weights_bytes": 10.0 * 1024**3,   # 10.0 GB (~80 Gbit)
        "weights_bits": 10.0 * 1024**3 * 8,
        "active_macs_per_token": 16.0e9,
        "sram_workspace_mm2": 65.0,
        "monolithic_feasible": False,      # Needs multi-chiplet (e.g. 4x chiplets)
        "chiplets_count": 4,
    },
}


def evaluate_die_sizing(model_key: str, tech_key: str) -> dict:
    m = MODELS[model_key]
    t = TECHNOLOGIES[tech_key]

    # Storage area (mm2)
    storage_area_mm2 = (m["weights_bits"] / 1e6) / t["storage_density_mbit_mm2"]

    # Compute area: assume target throughput of 100 tokens/sec requires
    # concurrency of C parallel lanes (e.g. 1024 active lanes):
    active_lanes = 2048
    compute_area_mm2 = (active_lanes * t["compute_area_per_mac_um2"]) / 1e6

    # Digital control, routing logic, SerDes, PCIe/UCIe PHY overhead
    phy_io_area_mm2 = 20.0

    total_silicon_mm2 = storage_area_mm2 + compute_area_mm2 + m["sram_workspace_mm2"] + phy_io_area_mm2

    # Reticle limit check (~858 mm2 max monolithic reticle; typical practical die < 400 mm2)
    chiplets = 1
    if total_silicon_mm2 > 400.0:
        chiplets = int((total_silicon_mm2 + 350.0) // 350.0)

    area_per_die = total_silicon_mm2 / chiplets

    return {
        "technology_name": t["name"],
        "evidence_level": t.get("evidence_level", "THEORETICAL"),
        "storage_density_mbit_mm2": t["storage_density_mbit_mm2"],
        "storage_area_mm2": round(storage_area_mm2, 1),
        "compute_area_mm2": round(compute_area_mm2, 2),
        "sram_workspace_mm2": m["sram_workspace_mm2"],
        "total_silicon_mm2": round(total_silicon_mm2, 1),
        "chiplets_count": chiplets,
        "area_per_die_mm2": round(area_per_die, 1),
    }


def evaluate_energy_and_power(model_key: str, tech_key: str, tokens_per_sec: float = 100.0) -> dict:
    m = MODELS[model_key]
    t = TECHNOLOGIES[tech_key]

    # Required on-chip internal bandwidth (GB/s) to stream weights at tokens_per_sec
    bandwidth_gb_s = (m["weights_bytes"] / 1e9) * tokens_per_sec

    # Storage read energy per token (Joules)
    weight_fetch_joules = (m["weights_bits"] * (t["read_energy_pj_per_bit"] * 1e-12))

    # Compute energy per token:
    # Dense MAC ~0.5 pJ / MAC op.
    # RailNet Stage-A (adder only) + Stage-B (1 shared mult / 96 adds) ~0.15 pJ / MAC-equivalent.
    compute_energy_per_mac_pj = 0.15 if "railnet" in tech_key else 0.50
    compute_joules = m["active_macs_per_token"] * (compute_energy_per_mac_pj * 1e-12)

    # SRAM workspace + clock tree overhead ~15%
    overhead_joules = (weight_fetch_joules + compute_joules) * 0.15

    total_joules_per_token = weight_fetch_joules + compute_joules + overhead_joules
    energy_mj_per_token = total_joules_per_token * 1000.0

    # Power (Watts) at target tokens/sec
    power_watts = total_joules_per_token * tokens_per_sec

    # Energy efficiency: tokens per Joule
    tokens_per_joule = 1.0 / max(1e-9, total_joules_per_token)

    return {
        "tokens_per_sec": tokens_per_sec,
        "internal_bandwidth_gb_s": round(bandwidth_gb_s, 1),
        "read_energy_pj_per_bit": t["read_energy_pj_per_bit"],
        "weight_fetch_energy_mj": round(weight_fetch_joules * 1000, 2),
        "compute_energy_mj": round(compute_joules * 1000, 2),
        "total_energy_mj_per_token": round(energy_mj_per_token, 2),
        "power_watts": round(power_watts, 1),
        "tokens_per_joule": round(tokens_per_joule, 1),
    }


def evaluate_fleet_tco(num_models: int = 5, chips_per_model: int = 2000) -> dict:
    """TCO for an enterprise/cloud fleet serving N distinct models."""
    results = {}
    total_chips = num_models * chips_per_model

    for tech_key in ("taalas_rom", "railnet_reram_academic", "railnet_reram_commercial"):
        t = TECHNOLOGIES[tech_key]
        if tech_key == "taalas_rom":
            total_mask_nre = num_models * t["mask_nre_per_model_usd"]
            cost_per_good_die = 45.0
            die_costs = total_chips * cost_per_good_die
            inventory_risk = 3_000_000
        elif tech_key == "railnet_reram_academic":
            total_mask_nre = t["base_mask_nre_usd"]
            cost_per_good_die = 55.0
            die_costs = total_chips * cost_per_good_die
            inventory_risk = 500_000
        else: # commercial silicon reality: larger multi-chiplet packaging
            total_mask_nre = t["base_mask_nre_usd"]
            cost_per_good_die = 85.0 # Multi-chiplet package (2-5 dies) + substrate
            die_costs = total_chips * cost_per_good_die
            inventory_risk = 500_000

        total_tco = total_mask_nre + die_costs + inventory_risk
        tco_per_chip = total_tco / total_chips

        results[tech_key] = {
            "evidence_level": t.get("evidence_level", "THEORETICAL"),
            "num_models": num_models,
            "total_chips": total_chips,
            "total_mask_nre_usd": total_mask_nre,
            "die_manufacturing_usd": die_costs,
            "inventory_risk_usd": inventory_risk,
            "total_tco_usd": total_tco,
            "effective_cost_per_chip_usd": round(tco_per_chip, 2),
        }

    results["savings_academic_vs_taalas_usd"] = results["taalas_rom"]["total_tco_usd"] - results["railnet_reram_academic"]["total_tco_usd"]
    results["savings_commercial_vs_taalas_usd"] = results["taalas_rom"]["total_tco_usd"] - results["railnet_reram_commercial"]["total_tco_usd"]
    return results


def run_full_evaluation() -> dict:
    report = {
        "models": {},
        "fleet_tco_5_models": evaluate_fleet_tco(num_models=5, chips_per_model=2000),
        "fleet_tco_10_models": evaluate_fleet_tco(num_models=10, chips_per_model=2000),
    }

    for m_key in MODELS:
        report["models"][m_key] = {}
        for t_key in TECHNOLOGIES:
            sizing = evaluate_die_sizing(m_key, t_key)
            energy = evaluate_energy_and_power(m_key, t_key, tokens_per_sec=100.0)
            report["models"][m_key][t_key] = {**sizing, **energy}

    return report


def main() -> int:
    report = run_full_evaluation()

    (ROOT / "results").mkdir(exist_ok=True)
    out_file = ROOT / "results" / "taalas_vs_railnet_ppa.json"
    out_file.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("==========================================================================================================")
    print(" TAALAS (FIXED ROM) vs RAILNET (REPROGRAMMABLE ReRAM) — QUANTITATIVE PPA BENCHMARK")
    print(" [Dual Track: Academic Literature Model vs Commercial 22nm Silicon Reality]")
    print("==========================================================================================================\n")

    for m_key, m_info in MODELS.items():
        print(f"--- MODEL: {m_info['name']} (Weights: {m_info['weights_bytes']/1e9:.2f} GB) ---")
        print(f"{'Technology':38s} {'Evidence Level':28s} {'Die mm2':>9} {'Chiplets':>9} {'Energy/tok':>12} {'Power':>8}")
        print("-" * 110)
        for t_key in ("taalas_rom", "railnet_reram_academic", "railnet_reram_commercial", "mram_baseline", "sram_baseline"):
            r = report["models"][m_key][t_key]
            tech_name = TECHNOLOGIES[t_key]["name"]
            ev_level = TECHNOLOGIES[t_key]["evidence_level"]
            print(
                f"{tech_name:38s} {ev_level:28s} {r['total_silicon_mm2']:>9.1f} {r['chiplets_count']:>9d} "
                f"{r['total_energy_mj_per_token']:>12.2f} {r['power_watts']:>8.1f}"
            )
        print("\n")

    tco5 = report["fleet_tco_5_models"]
    tco10 = report["fleet_tco_10_models"]

    print("==========================================================================================================")
    print(" MULTI-MODEL FLEET TCO COMPARISON (Enterprise / Cloud Deployment)")
    print("==========================================================================================================")
    print(f"Scenario A (5 Models, 10,000 Total Chips):")
    print(f"  * Taalas (5 separate mask sets):           ${tco5['taalas_rom']['total_tco_usd']/1e6:.1f}M (${tco5['taalas_rom']['effective_cost_per_chip_usd']:.0f}/chip) [THEORETICAL]")
    print(f"  * RailNet Academic (1 mask set, monolithic):${tco5['railnet_reram_academic']['total_tco_usd']/1e6:.1f}M (${tco5['railnet_reram_academic']['effective_cost_per_chip_usd']:.0f}/chip) [THEORETICAL]")
    print(f"  * RailNet Commercial (1 mask, multi-chiplet):${tco5['railnet_reram_commercial']['total_tco_usd']/1e6:.1f}M (${tco5['railnet_reram_commercial']['effective_cost_per_chip_usd']:.0f}/chip) [SILICON-MEASURED FOUNDRY IP]")
    print(f"  -> Commercial Silicon Fleet Savings vs Taalas: ${tco5['savings_commercial_vs_taalas_usd']/1e6:.1f}M\n")

    print(f"Scenario B (10 Models, 20,000 Total Chips):")
    print(f"  * Taalas (10 separate mask sets):          ${tco10['taalas_rom']['total_tco_usd']/1e6:.1f}M (${tco10['taalas_rom']['effective_cost_per_chip_usd']:.0f}/chip) [THEORETICAL]")
    print(f"  * RailNet Academic (1 mask set, monolithic):${tco10['railnet_reram_academic']['total_tco_usd']/1e6:.1f}M (${tco10['railnet_reram_academic']['effective_cost_per_chip_usd']:.0f}/chip) [THEORETICAL]")
    print(f"  * RailNet Commercial (1 mask, multi-chiplet):${tco10['railnet_reram_commercial']['total_tco_usd']/1e6:.1f}M (${tco10['railnet_reram_commercial']['effective_cost_per_chip_usd']:.0f}/chip) [SILICON-MEASURED FOUNDRY IP]")
    print(f"  -> Commercial Silicon Fleet Savings vs Taalas: ${tco10['savings_commercial_vs_taalas_usd']/1e6:.1f}M\n")

    print(f"-> Full metrics saved to: {out_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
