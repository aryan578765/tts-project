"""
TTS Cost Calculator
===================
Calculates the marginal cost per generated hour of audio based on:
  - RTF (Real-Time Factor): how long it takes to generate 1 second of audio
  - GPU hourly cost

Compares against a commercial baseline and outputs savings multiplier.

Usage:
    python cost_calculator.py --rtf 0.05 --gpu L4
    python cost_calculator.py --rtf 0.05 --gpu custom --gpu-cost 0.50
    python cost_calculator.py --rtf 0.05 --gpu all
    python cost_calculator.py --benchmark-csv benchmark_results.csv --gpu L4
"""

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# GPU pricing (RunPod Community Cloud, approximate $/hr as of 2025)
# ---------------------------------------------------------------------------
GPU_CONFIGS: dict[str, float] = {
    "L4":   0.69,
    "A40":  1.22,
    "A100": 2.72,
    "H100": 4.49,
    "L40S": 1.59,
    "4090": 0.74,
}

# Commercial TTS baseline cost per hour of generated audio
COMMERCIAL_BASELINE_PER_HOUR = 1.28  # $/hr (e.g., typical cloud TTS API)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
@dataclass
class CostEstimate:
    gpu_name: str
    gpu_hourly_cost: float
    rtf: float
    cost_per_audio_hour: float
    commercial_baseline: float
    savings_multiplier: float
    cost_per_audio_minute: float
    cost_per_1000_chars: float  # estimated


def calculate_cost(
    rtf: float,
    gpu_name: str,
    gpu_hourly_cost: float,
    commercial_baseline: float = COMMERCIAL_BASELINE_PER_HOUR,
) -> CostEstimate:
    """
    Calculate marginal cost per hour of generated audio.

    RTF = generation_time / audio_duration
    If RTF = 0.05, generating 1 hour of audio takes 0.05 hours of GPU time.

    cost_per_audio_hour = RTF * gpu_hourly_cost
    savings_multiplier  = commercial_baseline / cost_per_audio_hour
    """
    cost_per_audio_hour = rtf * gpu_hourly_cost
    savings_multiplier = (
        commercial_baseline / cost_per_audio_hour if cost_per_audio_hour > 0 else float("inf")
    )
    cost_per_audio_minute = cost_per_audio_hour / 60
    # Rough estimate: ~150 words/min speaking, ~5 chars/word = ~750 chars/min
    cost_per_1000_chars = cost_per_audio_minute * (1000 / 750)

    return CostEstimate(
        gpu_name=gpu_name,
        gpu_hourly_cost=gpu_hourly_cost,
        rtf=rtf,
        cost_per_audio_hour=round(cost_per_audio_hour, 4),
        commercial_baseline=commercial_baseline,
        savings_multiplier=round(savings_multiplier, 2),
        cost_per_audio_minute=round(cost_per_audio_minute, 6),
        cost_per_1000_chars=round(cost_per_1000_chars, 6),
    )


def load_rtf_from_benchmark_csv(csv_path: str) -> Optional[float]:
    """Load mean RTF from a benchmark summary CSV file."""
    path = Path(csv_path)
    if not path.exists():
        print(f"[ERROR] Benchmark CSV not found: {path}", file=sys.stderr)
        return None

    rtfs = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "mean_rtf" in row:
                try:
                    rtfs.append(float(row["mean_rtf"]))
                except ValueError:
                    continue

    if not rtfs:
        print("[ERROR] No RTF values found in CSV.", file=sys.stderr)
        return None

    import statistics
    return statistics.mean(rtfs)


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------
def print_cost_report(estimates: list[CostEstimate], model_name: str = "Model"):
    """Print a formatted cost comparison report."""
    print(f"\n{'=' * 72}")
    print(f"TTS COST ANALYSIS: {model_name}")
    print(f"{'=' * 72}")
    print(f"RTF (Real-Time Factor): {estimates[0].rtf}")
    print(f"  → Generating 1 hour of audio takes {estimates[0].rtf * 60:.1f} minutes of GPU time")
    print(f"  → Generating 1 hour of audio takes {estimates[0].rtf:.4f} GPU-hours")
    print(f"\nCommercial baseline: ${COMMERCIAL_BASELINE_PER_HOUR:.2f}/hr of audio")
    print(f"\n{'-' * 72}")
    print(
        f"{'GPU':<8} {'$/hr GPU':>10} {'$/hr Audio':>12} {'$/min Audio':>12} "
        f"{'$/1K chars':>12} {'Savings':>10}"
    )
    print(
        f"{'---':<8} {'--------':>10} {'----------':>12} {'----------':>12} "
        f"{'----------':>12} {'-------':>10}"
    )

    for est in estimates:
        savings_str = f"{est.savings_multiplier:.1f}x"
        if est.savings_multiplier >= 100:
            savings_str = f"{est.savings_multiplier:.0f}x"

        print(
            f"{est.gpu_name:<8} ${est.gpu_hourly_cost:>9.2f} "
            f"${est.cost_per_audio_hour:>11.4f} "
            f"${est.cost_per_audio_minute:>11.6f} "
            f"${est.cost_per_1000_chars:>11.6f} "
            f"{savings_str:>10}"
        )

    print(f"\n{'-' * 72}")

    # Best option
    best = min(estimates, key=lambda e: e.cost_per_audio_hour)
    print(f"\n✓ Best value: {best.gpu_name} at ${best.cost_per_audio_hour:.4f}/hr of audio")
    print(f"  That's {best.savings_multiplier:.1f}x cheaper than commercial TTS (${COMMERCIAL_BASELINE_PER_HOUR:.2f}/hr)")

    # Volume example
    print(f"\n--- Volume Examples (using {best.gpu_name}) ---")
    for hours in [1, 10, 100, 1000]:
        self_hosted = hours * best.cost_per_audio_hour
        commercial = hours * COMMERCIAL_BASELINE_PER_HOUR
        saved = commercial - self_hosted
        print(
            f"  {hours:>5} hrs audio:  Self-hosted ${self_hosted:>8.2f}  |  "
            f"Commercial ${commercial:>8.2f}  |  Saved ${saved:>8.2f}"
        )


def write_cost_csv(estimates: list[CostEstimate], output_path: str):
    """Write cost estimates to CSV."""
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "gpu_name", "gpu_hourly_cost", "rtf",
            "cost_per_audio_hour", "cost_per_audio_minute", "cost_per_1000_chars",
            "commercial_baseline", "savings_multiplier",
        ])
        for est in estimates:
            writer.writerow([
                est.gpu_name, est.gpu_hourly_cost, est.rtf,
                est.cost_per_audio_hour, est.cost_per_audio_minute, est.cost_per_1000_chars,
                est.commercial_baseline, est.savings_multiplier,
            ])
    print(f"\nCost estimates written to: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="TTS Cost Calculator — Compare self-hosted vs commercial TTS costs"
    )

    # RTF source (either direct or from benchmark CSV)
    rtf_group = parser.add_mutually_exclusive_group(required=True)
    rtf_group.add_argument(
        "--rtf", type=float, help="Real-Time Factor (e.g., 0.05 means 20x real-time)"
    )
    rtf_group.add_argument(
        "--benchmark-csv", type=str,
        help="Path to benchmark summary CSV to extract mean RTF"
    )

    # GPU config
    parser.add_argument(
        "--gpu", required=True,
        help=f"GPU type: {', '.join(GPU_CONFIGS.keys())}, 'all', or 'custom'"
    )
    parser.add_argument(
        "--gpu-cost", type=float, default=None,
        help="Custom GPU hourly cost (required when --gpu custom)"
    )
    parser.add_argument(
        "--baseline", type=float, default=COMMERCIAL_BASELINE_PER_HOUR,
        help=f"Commercial baseline $/hr of audio (default: {COMMERCIAL_BASELINE_PER_HOUR})"
    )
    parser.add_argument(
        "--model", type=str, default="Model",
        help="Model name for the report header"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Optional CSV output path for cost estimates"
    )

    args = parser.parse_args()

    # Resolve RTF
    rtf = args.rtf
    if rtf is None:
        rtf = load_rtf_from_benchmark_csv(args.benchmark_csv)
        if rtf is None:
            sys.exit(1)
        print(f"Loaded mean RTF from benchmark: {rtf:.4f}")

    if rtf <= 0:
        print("[ERROR] RTF must be positive.", file=sys.stderr)
        sys.exit(1)

    # Resolve GPU configs
    gpu_arg = args.gpu.upper()
    if gpu_arg == "ALL":
        gpu_list = list(GPU_CONFIGS.items())
    elif gpu_arg == "CUSTOM":
        if args.gpu_cost is None:
            print("[ERROR] --gpu-cost is required when --gpu custom", file=sys.stderr)
            sys.exit(1)
        gpu_list = [("Custom", args.gpu_cost)]
    elif gpu_arg in GPU_CONFIGS:
        gpu_list = [(gpu_arg, GPU_CONFIGS[gpu_arg])]
    else:
        print(
            f"[ERROR] Unknown GPU '{args.gpu}'. "
            f"Choose from: {', '.join(GPU_CONFIGS.keys())}, 'all', or 'custom'",
            file=sys.stderr,
        )
        sys.exit(1)

    # Calculate
    estimates = [
        calculate_cost(rtf, gpu_name, gpu_cost, args.baseline)
        for gpu_name, gpu_cost in gpu_list
    ]

    print_cost_report(estimates, model_name=args.model)

    if args.output:
        write_cost_csv(estimates, args.output)


if __name__ == "__main__":
    main()
