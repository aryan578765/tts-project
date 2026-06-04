"""
SSML Pause Support Tester
==========================
Tests whether a TTS endpoint respects SSML <break> tags by generating audio
with and without pauses, then comparing the results.

Usage:
    python test_ssml.py --endpoint http://localhost:8000
    python test_ssml.py --endpoint http://localhost:8000 --model kokoro
"""

import argparse
import base64
import json
import struct
import sys
import time
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
AUDIO_OUTPUT_DIR = PROJECT_ROOT / "audio_output"


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------
SSML_TEST_CASES = [
    {
        "id": "ssml_break_10ms",
        "description": "10ms break between words",
        "text_plain": "This is a test of pause insertion.",
        "text_ssml": 'This is a <break time="10ms"/> test of pause insertion.',
    },
    {
        "id": "ssml_break_250ms",
        "description": "250ms break (noticeable pause)",
        "text_plain": "The system is processing your request now.",
        "text_ssml": 'The system is processing <break time="250ms"/> your request now.',
    },
    {
        "id": "ssml_break_500ms",
        "description": "500ms break (dramatic pause)",
        "text_plain": "I have one thing to say. Listen carefully.",
        "text_ssml": 'I have one thing to say. <break time="500ms"/> Listen carefully.',
    },
    {
        "id": "ssml_break_1s",
        "description": "1 second break (long pause)",
        "text_plain": "First point. Second point.",
        "text_ssml": 'First point. <break time="1s"/> Second point.',
    },
    {
        "id": "ssml_multiple_breaks",
        "description": "Multiple breaks in one sentence",
        "text_plain": "Step one, step two, step three, done.",
        "text_ssml": (
            'Step one, <break time="200ms"/> step two, <break time="200ms"/> '
            'step three, <break time="500ms"/> done.'
        ),
    },
    {
        "id": "ssml_break_emphasis",
        "description": "Break combined with emphasis context",
        "text_plain": "This is not acceptable. We need to fix it.",
        "text_ssml": 'This is not acceptable. <break time="300ms"/> We need to fix it.',
    },
]


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------
def get_wav_duration(audio_bytes: bytes) -> float:
    """Parse WAV header to get audio duration in seconds."""
    if len(audio_bytes) < 44:
        return 0.0
    try:
        sample_rate = struct.unpack_from("<I", audio_bytes, 24)[0]
        bits_per_sample = struct.unpack_from("<H", audio_bytes, 34)[0]
        num_channels = struct.unpack_from("<H", audio_bytes, 22)[0]

        # Find data chunk
        idx = 12
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


def estimate_duration(audio_bytes: bytes) -> float:
    """Estimate audio duration, trying WAV header first, then fallback."""
    dur = get_wav_duration(audio_bytes)
    if dur > 0:
        return dur
    # Fallback: assume 24kHz 16-bit mono
    if len(audio_bytes) > 44:
        return (len(audio_bytes) - 44) / (24000 * 2)
    return 0.0


# ---------------------------------------------------------------------------
# RunPod / TTS request helpers
# ---------------------------------------------------------------------------
RUNPOD_BASE_URL = "https://api.runpod.ai/v2"


def get_payload(model: str, text: str, is_ssml: bool) -> dict:
    """Build payload for Kokoro or CosyVoice."""
    if model == "kokoro":
        return {
            "input": {
                "text": text,
                "voice": "af_heart",
                "speed": 1.0,
                "lang_code": "a",
                "ssml": is_ssml,
            }
        }
    elif model == "cosyvoice":
        return {
            "input": {
                "text": text,
                "voice": "英文女",
                "speed": 1.0,
                "language": "en",
                "mode": "sft",
            }
        }
    else:
        return {
            "input": {
                "text": text,
            }
        }


def submit_job(endpoint_id: str, api_key: str, payload: dict, timeout: int = 300) -> dict:
    """Submit job to RunPod serverless and wait for completion."""
    url = f"{RUNPOD_BASE_URL}/{endpoint_id}/runsync"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
    except requests.RequestException as exc:
        return {"error": f"Request failed: {exc}"}

    if resp.status_code == 401:
        return {"error": "Unauthorized – check your RunPod API key"}
    if resp.status_code == 404:
        return {"error": f"Endpoint {endpoint_id} not found"}
    if resp.status_code >= 400:
        return {"error": f"HTTP {resp.status_code}: {resp.text[:200]}"}

    try:
        data = resp.json()
    except json.JSONDecodeError:
        return {"error": "Invalid JSON response"}

    status = data.get("status", "")

    if status == "COMPLETED":
        return data

    job_id = data.get("id")
    if not job_id:
        return {"error": f"No job ID in response: {data}"}

    if status in ("IN_QUEUE", "IN_PROGRESS"):
        print(f"  [INFO] Job {job_id} is {status}. Polling...")
        return poll_job(endpoint_id, api_key, job_id, timeout)

    if status == "FAILED":
        error = data.get("error", "Unknown error")
        return {"error": f"Job failed: {error}", "raw": data}

    return {"error": f"Unknown status '{status}': {data}"}


def poll_job(endpoint_id: str, api_key: str, job_id: str, timeout: int = 300) -> dict:
    """Poll RunPod job status."""
    url = f"{RUNPOD_BASE_URL}/{endpoint_id}/status/{job_id}"
    headers = {"Authorization": f"Bearer {api_key}"}

    start = time.time()
    poll_interval = 2.0

    while time.time() - start < timeout:
        time.sleep(poll_interval)
        try:
            resp = requests.get(url, headers=headers, timeout=30)
            data = resp.json()
        except Exception as exc:
            print(f"  [WARN] Poll error: {exc}")
            continue

        status = data.get("status", "")

        if status == "COMPLETED":
            return data
        elif status == "FAILED":
            error = data.get("error", "Unknown error")
            return {"error": f"Job failed: {error}", "raw": data}
        elif status in ("IN_QUEUE", "IN_PROGRESS"):
            poll_interval = min(poll_interval * 1.2, 10.0)
        else:
            return {"error": f"Unknown status '{status}'"}

    return {"error": f"Job timed out after {timeout}s"}


def extract_audio(result: dict) -> bytes | None:
    """Extract audio bytes from RunPod job result."""
    output = result.get("output", result)
    if isinstance(output, dict):
        if "error" in output:
            print(f"  [ERROR] Handler error: {output['error']}", file=sys.stderr)
            return None
        for key in ("audio_base64", "audio", "audio_data"):
            if key in output and isinstance(output[key], str):
                try:
                    return base64.b64decode(output[key])
                except Exception:
                    continue
    return None


# ---------------------------------------------------------------------------
# Main test logic
# ---------------------------------------------------------------------------
def run_ssml_tests(model: str, endpoint_id: str, api_key: str, timeout: int = 300) -> list[dict]:
    """Run all SSML test cases and return results."""
    output_dir = AUDIO_OUTPUT_DIR / model / "ssml_tests"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'=' * 65}")
    print(f"SSML Pause Support Test: {model}")
    print(f"Endpoint ID: {endpoint_id}")
    print(f"Output:   {output_dir}")
    print(f"{'=' * 65}\n")

    results = []

    for case in SSML_TEST_CASES:
        test_id = case["id"]
        desc = case["description"]

        print(f"--- {test_id}: {desc} ---")

        # Generate without SSML (plain text)
        print(f"  [plain]  Generating...")
        payload_plain = get_payload(model, case["text_plain"], is_ssml=False)
        start_plain = time.perf_counter()
        res_plain = submit_job(endpoint_id, api_key, payload_plain, timeout)
        time_plain = time.perf_counter() - start_plain

        audio_plain = None
        if "error" not in res_plain:
            audio_plain = extract_audio(res_plain)
        else:
            print(f"  [plain]  ✗ Submission/Execution error: {res_plain['error']}")

        # Generate with SSML
        print(f"  [ssml]   Generating...")
        payload_ssml = get_payload(model, case["text_ssml"], is_ssml=True)
        start_ssml = time.perf_counter()
        res_ssml = submit_job(endpoint_id, api_key, payload_ssml, timeout)
        time_ssml = time.perf_counter() - start_ssml

        audio_ssml = None
        if "error" not in res_ssml:
            audio_ssml = extract_audio(res_ssml)
        else:
            print(f"  [ssml]   ✗ Submission/Execution error: {res_ssml['error']}")

        result = {
            "test_id": test_id,
            "description": desc,
            "text_plain": case["text_plain"],
            "text_ssml": case["text_ssml"],
            "plain_success": audio_plain is not None,
            "ssml_success": audio_ssml is not None,
            "plain_duration_s": 0.0,
            "ssml_duration_s": 0.0,
            "duration_difference_s": 0.0,
            "pause_respected": False,
            "plain_gen_time_s": round(time_plain, 4),
            "ssml_gen_time_s": round(time_ssml, 4),
        }

        if audio_plain:
            plain_path = output_dir / f"{test_id}_plain.wav"
            with open(plain_path, "wb") as f:
                f.write(audio_plain)
            plain_dur = estimate_duration(audio_plain)
            result["plain_duration_s"] = round(plain_dur, 4)
            print(f"  [plain]  ✓ Saved ({plain_dur:.3f}s, {len(audio_plain)/1024:.1f}KB)")
        else:
            print(f"  [plain]  ✗ Failed to generate audio")

        if audio_ssml:
            ssml_path = output_dir / f"{test_id}_ssml.wav"
            with open(ssml_path, "wb") as f:
                f.write(audio_ssml)
            ssml_dur = estimate_duration(audio_ssml)
            result["ssml_duration_s"] = round(ssml_dur, 4)
            print(f"  [ssml]   ✓ Saved ({ssml_dur:.3f}s, {len(audio_ssml)/1024:.1f}KB)")
        else:
            print(f"  [ssml]   ✗ Failed to generate audio")

        # Compare durations
        if audio_plain and audio_ssml:
            dur_diff = result["ssml_duration_s"] - result["plain_duration_s"]
            result["duration_difference_s"] = round(dur_diff, 4)

            expected_break_s = _parse_break_duration(case["text_ssml"])
            threshold = expected_break_s * 0.3  # at least 30% of expected break

            if dur_diff > threshold and threshold > 0:
                result["pause_respected"] = True
                print(f"  [result] ✓ PAUSE RESPECTED (diff: +{dur_diff:.3f}s, expected: ~{expected_break_s:.3f}s)")
            elif dur_diff > 0.01:
                print(f"  [result] ? POSSIBLE (diff: +{dur_diff:.3f}s, expected: ~{expected_break_s:.3f}s)")
            else:
                print(f"  [result] ✗ PAUSE NOT DETECTED (diff: {dur_diff:.3f}s)")

        results.append(result)
        print()

    return results


def _parse_break_duration(ssml_text: str) -> float:
    """Extract total break duration from SSML text in seconds."""
    import re

    total = 0.0
    for match in re.finditer(r'<break\s+time="([^"]+)"', ssml_text):
        time_str = match.group(1)
        if time_str.endswith("ms"):
            total += float(time_str[:-2]) / 1000
        elif time_str.endswith("s"):
            total += float(time_str[:-1])
    return total


def print_summary(results: list[dict]):
    """Print a summary table of SSML test results."""
    print(f"{'=' * 65}")
    print(f"SSML TEST SUMMARY")
    print(f"{'=' * 65}")
    print(f"{'Test':<25} {'Plain':>8} {'SSML':>8} {'Diff':>8} {'Pause?':>8}")
    print(f"{'-' * 25} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 8}")

    for r in results:
        pause_str = "✓ Yes" if r["pause_respected"] else "✗ No"
        if not r["plain_success"] or not r["ssml_success"]:
            pause_str = "N/A"

        print(
            f"{r['test_id']:<25} "
            f"{r['plain_duration_s']:>7.3f}s "
            f"{r['ssml_duration_s']:>7.3f}s "
            f"{r['duration_difference_s']:>+7.3f}s "
            f"{pause_str:>8}"
        )

    respected = sum(1 for r in results if r["pause_respected"])
    total = sum(1 for r in results if r["plain_success"] and r["ssml_success"])
    print(f"\nPauses respected: {respected}/{total}")

    if respected == 0:
        print("\n⚠ The model does not appear to support SSML <break> tags.")
        print("  This is expected for many open-source TTS models.")
    elif respected == total:
        print("\n✓ Full SSML <break> support confirmed!")
    else:
        print(f"\n⚠ Partial SSML support ({respected}/{total} tests passed).")


def main():
    parser = argparse.ArgumentParser(
        description="Test SSML pause support in TTS RunPod serverless endpoints"
    )
    parser.add_argument(
        "--model", required=True, help="Model name (kokoro or cosyvoice)"
    )
    parser.add_argument(
        "--endpoint-id", required=True, help="RunPod endpoint ID"
    )
    parser.add_argument(
        "--api-key", default=None, help="RunPod API key (or set RUNPOD_API_KEY env var)"
    )
    parser.add_argument(
        "--timeout", type=int, default=300, help="Request timeout in seconds (default: 300)"
    )

    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        print("[ERROR] RunPod API key required. Use --api-key or set RUNPOD_API_KEY env var", file=sys.stderr)
        sys.exit(1)

    results = run_ssml_tests(args.model, args.endpoint_id, api_key, args.timeout)
    print_summary(results)

    # Save results
    output_dir = AUDIO_OUTPUT_DIR / args.model / "ssml_tests"
    results_path = output_dir / "ssml_test_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model": args.model,
                "endpoint_id": args.endpoint_id,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "results": results,
            },
            f,
            indent=2,
        )
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
