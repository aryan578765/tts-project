# TTS Model Evaluation — Final Report

## 1. Executive Summary

This report evaluates two state-of-the-art open-source text-to-speech (TTS) models, **Kokoro v1.0 (82M parameters)** and **CosyVoice 2.0 (0.5B parameters)**, deployed on **RunPod serverless L4 GPU ($0.69/hr)** infrastructure. The evaluation compares them against a typical **commercial cloud TTS baseline ($1.28 per generated hour)** and the client's existing **StyleTTS 2** setup.

### Key Takeaways
- **Kokoro v1.0** is the **most cost-effective** option. With a mean Real-Time Factor (RTF) of **0.2319** (4.3x real-time speed), it generates audio at **$0.1600 per hour** on an L4 GPU, representing an **8.0x cost reduction** over commercial APIs. It has native, robust support for SSML `<break>` pauses via our custom serverless handler.
- **CosyVoice 2.0** provides the **highest quality and broadest features**, including zero-shot voice cloning and instruct-controlled emotion/timbre. However, it runs slower with a mean RTF of **1.1249** (0.9x real-time speed), costing **$0.7762 per audio hour** on L4 (a **1.6x cost reduction** over commercial). It does **not** natively support standard SSML pause control.

---

## 2. Head-to-Head Comparison

| Metric / Feature | Kokoro v1.0 | CosyVoice 2.0 | StyleTTS 2 (Baseline) | Commercial APIs |
| :--- | :---: | :---: | :---: | :---: |
| **Model Size** | 82M | 500M | ~150M | Proprietary |
| **Quality (Naturalness)** | Very High (Elo 1058) | Very High | High | Human-like |
| **Mean RTF (L4 GPU)** | **0.2319** (4.3x speed) | **1.1249** (0.9x speed) | **0.1500** (6.7x speed) | N/A (API) |
| **Cost per Audio Hour** | **$0.1600** | **$0.7762** | **$0.1035** | **$1.2800** |
| **Savings vs Commercial** | **8.0x cheaper** | **1.6x cheaper** | **12.4x cheaper** | Baseline |
| **SSML Pause Support** | Yes (`<break>` tag) | No (Spells out tags) | Partial | Full |
| **Multilingual Support** | Limited (4/15 OK) | Robust (9/15 OK) | English-only (Base) | Full |
| **Voice Cloning** | No | Yes (3s ref) | No | Zero-Shot / Custom |
| **GPU VRAM footprint** | ~2–4 GB | ~6–8 GB | ~4 GB | N/A |

---

## 3. Deep-Dive Findings

### A. SSML & Micro-Pause Support
- **Kokoro v1.0**: The custom handler successfully parses standard SSML `<break time="..."/>` tags and inserts silent intervals dynamically. Evaluated at 10ms, 250ms, 500ms, and 1s, the model respected every pause duration with 100% precision.
- **CosyVoice 2.0**: The model has no native understanding of SSML tags. When tags are included in the input, the text frontend stumbles, resulting in either literal reading of tags or excessive silent delays (e.g., adding 9.6s of silence for a 10ms break). Pause control in CosyVoice must be handled via natural language instructions (Instruct mode) rather than XML.

### B. Multilingual Capabilities
- **Kokoro v1.0**: Officially supports 7 languages (EN, ES, FR, IT, JA, PT, ZH). However, during serverless evaluation:
  - Hugging Face voice asset loading issues caused failures for French (`ff_sixtine.pt` 404) and Portuguese (`pf_daniele.pt` 404).
  - Japanese failed due to missing MeCab/unidic dictionaries in the default docker workspace.
  - Succeeded in English, Spanish, Italian, and Chinese.
- **CosyVoice 2.0**: Succeeded in all **9 officially supported languages** (`en, es, fr, de, it, ja, zh, ko, ru`) using native cross-lingual voice synthesis. Successfully rejected unsupported languages (Portuguese, Hindi, Arabic, Turkish, Dutch, Polish) with explicit validation errors.

### C. Cost & Performance Feasibility
Evaluating marginal costs per generated hour of audio on RunPod serverless L4 GPUs ($0.69/hr):
- **Kokoro**: High throughput leads to massive savings. At **$0.16/audio hour**, generating 1,000 hours of speech costs **$160.00** (saving **$1,120.00** vs. commercial).
- **CosyVoice**: Heavy computational overhead leads to higher cost. At **$0.7762/audio hour**, generating 1,000 hours of speech costs **$776.20** (saving **$503.80**).

---

## 4. Final Recommendations

1. **Deploy Kokoro v1.0 for English & Spanish standard TTS**:
   - If the main requirement is natural English speech with precise pause/ssml control at the absolute lowest cost, Kokoro v1.0 on L4 is the clear winner.
2. **Deploy CosyVoice 2.0 for Multilingual & Voice Cloning**:
   - If zero-shot voice cloning, emotional control (instruct mode), or languages like French, German, Japanese, Korean, and Russian are required, deploy CosyVoice 2.0. Note that it will cost ~4.8x more to run than Kokoro.
3. **Address Container Gaps**:
   - **Kokoro**: Update the Dockerfile to include MeCab/unidic system dependencies to enable Japanese, and resolve voice naming mappings on Hugging Face.
   - **Hebrew/Arabic**: Since neither model supports Hebrew or Arabic, the client must use commercial APIs for these languages or proceed with fine-tuning a custom StyleTTS 2 model.
