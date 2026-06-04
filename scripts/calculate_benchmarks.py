import json
import csv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
KOKORO_LOG = PROJECT_ROOT / "audio_output" / "kokoro" / "generation_log.json"
COSYVOICE_LOG = PROJECT_ROOT / "audio_output" / "cosyvoice" / "generation_log.json"
OUTPUT_CSV = PROJECT_ROOT / "benchmark" / "benchmark_results.csv"

def load_log(log_path):
    if not log_path.exists():
        print(f"Log not found: {log_path}")
        return None
    with open(log_path, "r", encoding="utf-8") as f:
        return json.load(f)

def main():
    kokoro_data = load_log(KOKORO_LOG)
    cosyvoice_data = load_log(COSYVOICE_LOG)
    
    if not kokoro_data or not cosyvoice_data:
        print("Missing logs. Make sure both models have run.")
        return

    # Prepare CSV data
    rows = []
    # Header
    rows.append([
        "test_id", "title", "text_length_chars",
        "kokoro_gen_time_s", "kokoro_audio_dur_s", "kokoro_rtf",
        "cosyvoice_gen_time_s", "cosyvoice_audio_dur_s", "cosyvoice_rtf"
    ])

    kokoro_entries = {e["test_id"]: e for e in kokoro_data["entries"]}
    cosyvoice_entries = {e["test_id"]: e for e in cosyvoice_data["entries"]}

    all_test_ids = sorted(list(set(kokoro_entries.keys()) | set(cosyvoice_entries.keys())))

    for tid in all_test_ids:
        k = kokoro_entries.get(tid, {})
        c = cosyvoice_entries.get(tid, {})

        title = k.get("title") or c.get("title", "Unknown")
        chars = k.get("text_length_chars") or c.get("text_length_chars", 0)

        k_gen = k.get("generation_time_s", 0)
        k_dur = k.get("duration_seconds", 0)
        k_rtf = k_gen / k_dur if k_dur > 0 else 0

        c_gen = c.get("generation_time_s", 0)
        c_dur = c.get("duration_seconds", 0)
        c_rtf = c_gen / c_dur if c_dur > 0 else 0

        rows.append([
            tid, title, chars,
            round(k_gen, 4), round(k_dur, 4), round(k_rtf, 4),
            round(c_gen, 4), round(c_dur, 4), round(c_rtf, 4)
        ])

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    print(f"Benchmark results written to {OUTPUT_CSV}")

    # Print summary
    k_rtfs = [r[5] for r in rows[1:] if r[5] > 0]
    c_rtfs = [r[8] for r in rows[1:] if r[8] > 0]

    k_mean_rtf = sum(k_rtfs) / len(k_rtfs) if k_rtfs else 0
    c_mean_rtf = sum(c_rtfs) / len(c_rtfs) if c_rtfs else 0

    print("\n" + "="*50)
    print("BENCHMARK SUMMARY")
    print("="*50)
    print(f"Kokoro v1.0   Mean RTF: {k_mean_rtf:.4f} (approx {1/k_mean_rtf:.1f}x speed)")
    print(f"CosyVoice 2.0 Mean RTF: {c_mean_rtf:.4f} (approx {1/c_mean_rtf:.1f}x speed)")
    print("="*50)

if __name__ == "__main__":
    main()
