"""
TTS Benchmark Runner
====================
Measures Real-Time Factor (RTF) and GPU VRAM usage for TTS model endpoints.
Runs multiple iterations for statistical significance and outputs results as CSV.

Usage:
    python run_benchmark.py --model kokoro --endpoint http://localhost:8000
    python run_benchmark.py --model kokoro --endpoint http://localhost:8000 --iterations 10 --output results.csv
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import requests

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
TEST_TEXTS_PATH = PROJECT_ROOT / "test_texts" / "test_texts.json"
DEFAULT_OUTPUT = SCRIPT_DIR / "benchmark_results.csv"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class BenchmarkResult:
    model: str
    test_id: str
    text_title: str
    text_length_chars: int
    iteration: int
    generation_time_s: float
    audio_duration_s: float
    rtf: float
    vram_mb: Optional[float] = None


@dataclass
class BenchmarkSummary:
    model: str
    test_id: str
    text_title: str
    mean_rtf: float
    std_rtf: float
    min_rtf: float
    max_rtf: float
    mean_gen_time_s: float
    mean_audio_dur_s: float
    mean_vram_mb: Optional[float] = None


# ---------------------------------------------------------------------------
# GPU helpers
# ---------------------------------------------------------------------------
def get_vram_usage_mb() -> Optional[float]:
    """Query nvidia-smi for current GPU VRAM usage in MB."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            # Take the first GPU if multiple are present
            line = result.stdout.strip().split("\n")[0]
            return float(line.strip())
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        pass
    return None


# ---------------------------------------------------------------------------
# Audio duration estimation
# ---------------------------------------------------------------------------
def get_audio_duration_from_wav(audio_bytes: bytes) -> float:
    """Estimate audio duration from raw WAV bytes using the header."""
    import struct

    if len(audio_bytes) < 44:
        return 0.0
    try:
        # Standard WAV: bytes 24-27 = sample rate, bytes 34-35 = bits per sample
        # bytes 40-43 = data chunk size
        # We'll parse more robustly by scanning for 'data' sub-chunk
        sample_rate = struct.unpack_from("<I", audio_bytes, 24)[0]
        bits_per_sample = struct.unpack_from("<H", audio_bytes, 34)[0]
        num_channels = struct.unpack_from("<H", audio_bytes, 22)[0]

        # Find the 'data' chunk
        idx = 12  # skip RIFF header
        while idx < len(audio_bytes) - 8:
            chunk_id = audio_bytes[idx : idx + 4]
            chunk_size = struct.unpack_from("<I", audio_bytes, idx + 4)[0]
            if chunk_id == b"data":
                bytes_per_sample = bits_per_sample // 8
                if sample_rate > 0 and bytes_per_sample > 0 and num_channels > 0:
                    num_samples = chunk_size // (bytes_per_sample * num_channels)
                    return num_samples / sample_rate
                break
            idx += 8 + chunk_size

    except (struct.error, ZeroDivisionError):
        pass

    return 0.0


def get_audio_duration_from_response(response: requests.Response) -> float:
    """Try to extract audio duration from response. Supports WAV and JSON with duration field."""
    content_type = response.headers.get("Content-Type", "")

    # If JSON response contains a duration field
    if "application/json" in content_type:
        try:
            data = response.json()
            if "audio_duration" in data:
                return float(data["audio_duration"])
            if "duration" in data:
                return float(data["duration"])
        except (json.JSONDecodeError, ValueError):
            pass

    # Try parsing as WAV
    duration = get_audio_duration_from_wav(response.content)
    if duration > 0:
        return duration

    # Fallback: estimate from content length assuming 24kHz 16-bit mono WAV
    content_length = len(response.content)
    if content_length > 44:
        estimated_samples = (content_length - 44) // 2  # 16-bit = 2 bytes
        return estimated_samples / 24000.0

    return 0.0


# ---------------------------------------------------------------------------
# TTS request
# ---------------------------------------------------------------------------
def send_tts_request(
    endpoint: str, text: str, timeout: int = 120
) -> tuple[float, float, Optional[float]]:
    """
    Send a TTS request and return (generation_time_s, audio_duration_s, vram_mb).

    Tries common payload formats:
      1. {"text": "...", "stream": false}
      2. {"input": {"text": "..."}}   (RunPod serverless)
    """
    payload = {"text": text, "stream": False}

    vram_before = get_vram_usage_mb()

    start = time.perf_counter()
    try:
        resp = requests.post(endpoint, json=payload, timeout=timeout)
        elapsed = time.perf_counter() - start

        # If the first format fails, try RunPod format
        if resp.status_code >= 400:
            payload_runpod = {"input": {"text": text}}
            start = time.perf_counter()
            resp = requests.post(endpoint, json=payload_runpod, timeout=timeout)
            elapsed = time.perf_counter() - start
    except requests.RequestException as exc:
        print(f"  [ERROR] Request failed: {exc}", file=sys.stderr)
        return (0.0, 0.0, None)

    vram_after = get_vram_usage_mb()
    vram_peak = max(vram_before or 0, vram_after or 0) if (vram_before or vram_after) else None

    if resp.status_code != 200:
        print(f"  [ERROR] HTTP {resp.status_code}: {resp.text[:200]}", file=sys.stderr)
        return (elapsed, 0.0, vram_peak)

    audio_duration = get_audio_duration_from_response(resp)
    return (elapsed, audio_duration, vram_peak)


# ---------------------------------------------------------------------------
# Benchmark loop
# ---------------------------------------------------------------------------
def run_benchmark(
    model: str,
    endpoint: str,
    iterations: int = 5,
    warmup: int = 1,
    timeout: int = 120,
) -> list[BenchmarkResult]:
    """Run the full benchmark across all test texts."""

    # Load test texts
    if not TEST_TEXTS_PATH.exists():
        print(f"[ERROR] Test texts not found: {TEST_TEXTS_PATH}", file=sys.stderr)
        sys.exit(1)

    with open(TEST_TEXTS_PATH, "r", encoding="utf-8") as f:
        test_texts = json.load(f)

    print(f"{'=' * 60}")
    print(f"TTS Benchmark: {model}")
    print(f"Endpoint:      {endpoint}")
    print(f"Iterations:    {iterations} (+{warmup} warmup)")
    print(f"Test texts:    {len(test_texts)}")
    print(f"{'=' * 60}\n")

    # Warmup
    if warmup > 0:
        print(f"[Warmup] Running {warmup} warmup request(s)...")
        for i in range(warmup):
            send_tts_request(endpoint, "This is a warmup request.", timeout=timeout)
        print("[Warmup] Done.\n")

    results: list[BenchmarkResult] = []

    for entry in test_texts:
        test_id = entry["id"]
        title = entry["title"]
        text = entry["text"]

        print(f"--- {test_id}: {title} ({len(text)} chars) ---")

        for i in range(1, iterations + 1):
            gen_time, audio_dur, vram = send_tts_request(endpoint, text, timeout=timeout)
            rtf = gen_time / audio_dur if audio_dur > 0 else float("inf")

            result = BenchmarkResult(
                model=model,
                test_id=test_id,
                text_title=title,
                text_length_chars=len(text),
                iteration=i,
                generation_time_s=round(gen_time, 4),
                audio_duration_s=round(audio_dur, 4),
                rtf=round(rtf, 4),
                vram_mb=round(vram, 1) if vram is not None else None,
            )
            results.append(result)

            vram_str = f"{vram:.0f} MB" if vram else "N/A"
            print(
                f"  iter {i}/{iterations}: "
                f"gen={gen_time:.3f}s  audio={audio_dur:.3f}s  "
                f"RTF={rtf:.4f}  VRAM={vram_str}"
            )

        print()

    return results


def compute_summaries(results: list[BenchmarkResult]) -> list[BenchmarkSummary]:
    """Aggregate per-text results into summaries with mean/std/min/max."""
    from collections import defaultdict
    import statistics

    grouped: dict[str, list[BenchmarkResult]] = defaultdict(list)
    for r in results:
        grouped[r.test_id].append(r)

    summaries = []
    for test_id, runs in sorted(grouped.items()):
        rtfs = [r.rtf for r in runs if r.rtf != float("inf")]
        gen_times = [r.generation_time_s for r in runs]
        audio_durs = [r.audio_duration_s for r in runs if r.audio_duration_s > 0]
        vrams = [r.vram_mb for r in runs if r.vram_mb is not None]

        summaries.append(
            BenchmarkSummary(
                model=runs[0].model,
                test_id=test_id,
                text_title=runs[0].text_title,
                mean_rtf=round(statistics.mean(rtfs), 4) if rtfs else 0.0,
                std_rtf=round(statistics.stdev(rtfs), 4) if len(rtfs) > 1 else 0.0,
                min_rtf=round(min(rtfs), 4) if rtfs else 0.0,
                max_rtf=round(max(rtfs), 4) if rtfs else 0.0,
                mean_gen_time_s=round(statistics.mean(gen_times), 4) if gen_times else 0.0,
                mean_audio_dur_s=round(statistics.mean(audio_durs), 4) if audio_durs else 0.0,
                mean_vram_mb=round(statistics.mean(vrams), 1) if vrams else None,
            )
        )

    return summaries


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def write_results_csv(
    results: list[BenchmarkResult],
    summaries: list[BenchmarkSummary],
    output_path: Path,
):
    """Write raw results and summary to CSV files."""
    # Raw results
    raw_path = output_path.with_suffix(".raw.csv")
    with open(raw_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "model", "test_id", "text_title", "text_length_chars",
            "iteration", "generation_time_s", "audio_duration_s", "rtf", "vram_mb",
        ])
        for r in results:
            writer.writerow([
                r.model, r.test_id, r.text_title, r.text_length_chars,
                r.iteration, r.generation_time_s, r.audio_duration_s, r.rtf,
                r.vram_mb if r.vram_mb is not None else "",
            ])
    print(f"Raw results written to: {raw_path}")

    # Summary
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "model", "test_id", "text_title",
            "mean_rtf", "std_rtf", "min_rtf", "max_rtf",
            "mean_gen_time_s", "mean_audio_dur_s", "mean_vram_mb",
        ])
        for s in summaries:
            writer.writerow([
                s.model, s.test_id, s.text_title,
                s.mean_rtf, s.std_rtf, s.min_rtf, s.max_rtf,
                s.mean_gen_time_s, s.mean_audio_dur_s,
                s.mean_vram_mb if s.mean_vram_mb is not None else "",
            ])
    print(f"Summary written to:     {output_path}")


def print_summary_table(summaries: list[BenchmarkSummary]):
    """Print a nicely formatted summary table."""
    print(f"\n{'=' * 80}")
    print(f"BENCHMARK SUMMARY: {summaries[0].model if summaries else 'N/A'}")
    print(f"{'=' * 80}")
    print(f"{'Test ID':<10} {'Title':<35} {'Mean RTF':>10} {'Std':>8} {'VRAM MB':>10}")
    print(f"{'-' * 10} {'-' * 35} {'-' * 10} {'-' * 8} {'-' * 10}")

    for s in summaries:
        vram_str = f"{s.mean_vram_mb:.0f}" if s.mean_vram_mb else "N/A"
        print(
            f"{s.test_id:<10} {s.text_title[:35]:<35} "
            f"{s.mean_rtf:>10.4f} {s.std_rtf:>8.4f} {vram_str:>10}"
        )

    # Overall
    all_rtfs = [s.mean_rtf for s in summaries if s.mean_rtf > 0]
    if all_rtfs:
        import statistics
        overall_mean = statistics.mean(all_rtfs)
        print(f"\n{'Overall Mean RTF:':<46} {overall_mean:>10.4f}")
        print(f"{'Interpretation:':<46} ", end="")
        if overall_mean < 0.1:
            print("Excellent (>10x real-time)")
        elif overall_mean < 0.5:
            print("Good (>2x real-time)")
        elif overall_mean < 1.0:
            print("Acceptable (faster than real-time)")
        else:
            print("Slow (slower than real-time)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="TTS Benchmark Runner — Measures RTF and VRAM for TTS endpoints"
    )
    parser.add_argument(
        "--model", required=True, help="Model name (e.g., kokoro, cosyvoice, styletts2)"
    )
    parser.add_argument(
        "--endpoint", required=True, help="TTS endpoint URL (e.g., http://localhost:8000/generate)"
    )
    parser.add_argument(
        "--iterations", type=int, default=5, help="Number of iterations per text (default: 5)"
    )
    parser.add_argument(
        "--warmup", type=int, default=1, help="Number of warmup requests (default: 1)"
    )
    parser.add_argument(
        "--timeout", type=int, default=120, help="Request timeout in seconds (default: 120)"
    )
    parser.add_argument(
        "--output", type=str, default=str(DEFAULT_OUTPUT),
        help=f"Output CSV path (default: {DEFAULT_OUTPUT})"
    )

    args = parser.parse_args()

    results = run_benchmark(
        model=args.model,
        endpoint=args.endpoint,
        iterations=args.iterations,
        warmup=args.warmup,
        timeout=args.timeout,
    )

    if not results:
        print("[ERROR] No results collected.", file=sys.stderr)
        sys.exit(1)

    summaries = compute_summaries(results)
    print_summary_table(summaries)
    write_results_csv(results, summaries, Path(args.output))


if __name__ == "__main__":
    main()
