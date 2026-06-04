"""
Generate Commercial Baselines Directly
======================================
Generates audio files for all 13 test texts using official Azure, GCP, or AWS Polly APIs.

Usage:
    python generate_commercial_baselines.py --api azure --key YOUR_AZURE_KEY --region eastus
    python generate_commercial_baselines.py --api google --key YOUR_GCP_API_KEY
    python generate_commercial_baselines.py --api aws --access-key ACCESS_KEY --secret-key SECRET_KEY
"""

import argparse
import base64
import json
import os
import struct
import sys
from pathlib import Path
import requests

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
TEST_TEXTS_PATH = PROJECT_ROOT / "test_texts" / "test_texts.json"
AUDIO_OUTPUT_DIR = PROJECT_ROOT / "audio_output"

def load_test_texts() -> list[dict]:
    if not TEST_TEXTS_PATH.exists():
        print(f"[ERROR] Test texts not found: {TEST_TEXTS_PATH}", file=sys.stderr)
        sys.exit(1)
    with open(TEST_TEXTS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def wrap_pcm_wav(pcm_bytes: bytes, sample_rate: int) -> bytes:
    """Wrap raw PCM bytes in a WAV header."""
    num_channels = 1
    bits_per_sample = 16
    byte_rate = sample_rate * num_channels * (bits_per_sample // 8)
    block_align = num_channels * (bits_per_sample // 8)
    data_size = len(pcm_bytes)
    chunk_size = 36 + data_size
    
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", chunk_size, b"WAVE", b"fmt ", 16, 1, num_channels,
        sample_rate, byte_rate, block_align, bits_per_sample, b"data", data_size
    )
    return header + pcm_bytes

# ---------------------------------------------------------------------------
# API Wrappers
# ---------------------------------------------------------------------------
def generate_azure(text: str, key: str, region: str, voice: str = "en-US-JennyNeural") -> bytes:
    url = f"https://{region}.tts.speech.microsoft.com/cognitiveservices/v1"
    headers = {
        "Ocp-Apim-Subscription-Key": key,
        "Content-Type": "application/ssml+xml",
        "X-Microsoft-OutputFormat": "riff-24khz-16bit-mono-pcm",
        "User-Agent": "EllaTTSClient"
    }
    ssml = f"<speak version='1.0' xml:lang='en-US'><voice name='{voice}'>{text}</voice></speak>"
    resp = requests.post(url, headers=headers, data=ssml.encode('utf-8'), timeout=60)
    if resp.status_code == 200:
        return resp.content
    raise Exception(f"Azure HTTP {resp.status_code}: {resp.text}")

def generate_google(text: str, api_key: str, voice: str = "en-US-Neural2-F") -> bytes:
    url = f"https://texttospeech.googleapis.com/v1/text:synthesize?key={api_key}"
    payload = {
        "input": {"text": text},
        "voice": {"languageCode": "en-US", "name": voice},
        "audioConfig": {"audioEncoding": "LINEAR16", "sampleRateHertz": 24000}
    }
    resp = requests.post(url, json=payload, timeout=60)
    if resp.status_code == 200:
        return base64.b64decode(resp.json()["audioContent"])
    raise Exception(f"Google HTTP {resp.status_code}: {resp.text}")

def generate_aws(text: str, access_key: str, secret_key: str, region: str = "us-east-1", voice: str = "Joanna") -> bytes:
    try:
        import boto3
    except ImportError:
        print("[ERROR] AWS client requires boto3. Please install it: pip install boto3", file=sys.stderr)
        sys.exit(1)
        
    polly = boto3.client(
        "polly",
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name=region
    )
    resp = polly.synthesize_speech(
        Text=text,
        OutputFormat="pcm",
        SampleRate="24000",
        VoiceId=voice,
        Engine="neural"
    )
    pcm_bytes = resp["AudioStream"].read()
    return wrap_pcm_wav(pcm_bytes, 24000)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Generate commercial baseline audios directly")
    parser.add_argument("--api", required=True, choices=["azure", "google", "aws"], help="Commercial API type")
    parser.add_argument("--key", help="API key (Required for Azure/Google)")
    parser.add_argument("--region", default="eastus", help="Region (Azure endpoint region or AWS region)")
    parser.add_argument("--access-key", help="AWS Access Key ID")
    parser.add_argument("--secret-key", help="AWS Secret Access Key")
    parser.add_argument("--voice", help="Override default neural voice ID")
    
    args = parser.parse_args()
    test_texts = load_test_texts()
    output_dir = AUDIO_OUTPUT_DIR / f"commercial_{args.api}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating commercial baseline for: {args.api.upper()}")
    print(f"Saving to: {output_dir}\n")
    
    for entry in test_texts:
        test_id = entry["id"]
        text = entry["text"]
        print(f"[{test_id}] Synthesizing...")
        
        try:
            if args.api == "azure":
                if not args.key:
                    raise ValueError("Azure API requires --key")
                voice = args.voice or "en-US-JennyNeural"
                audio = generate_azure(text, args.key, args.region, voice)
            elif args.api == "google":
                if not args.key:
                    raise ValueError("Google API requires --key")
                voice = args.voice or "en-US-Neural2-F"
                audio = generate_google(text, args.key, voice)
            elif args.api == "aws":
                if not args.access_key or not args.secret_key:
                    raise ValueError("AWS Polly requires --access-key and --secret-key")
                voice = args.voice or "Joanna"
                audio = generate_aws(text, args.access_key, args.secret_key, args.region, voice)
            
            output_path = output_dir / f"{test_id}.wav"
            with open(output_path, "wb") as f:
                f.write(audio)
            print(f"  ✓ Saved to {output_path}")
            
        except Exception as exc:
            print(f"  ✗ Failed: {exc}")
            
if __name__ == "__main__":
    main()
