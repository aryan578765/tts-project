# TTS Model Evaluation & Verification Walkthrough

This walkthrough documents the successful execution and verification of the TTS evaluation pipeline comparing **Kokoro v1.0** and **CosyVoice 2.0** on RunPod serverless L4 GPU infrastructure.

---

## 1. SSML Pause Support Testing

We updated `scripts/test_ssml.py` to communicate directly with RunPod endpoints (using Bearer tokens and polling) and ran tests for both models.

### Kokoro v1.0 Verification
- **Endpoint ID**: `3z59kpduf0vkil`
- **Result**: **100% Passed (6/6 tests)**
- **Findings**: Kokoro's custom handler parses `<break>` tags and programmatically inserts silent intervals. It correctly respected all pause lengths (from 10ms up to 1.0s) and multiple breaks in single utterances.

### CosyVoice 2.0 Verification
- **Endpoint ID**: `x3yzb0ve3mgx82`
- **Result**: **6/6 marked as passed by duration difference, but with issues**
- **Findings**: While the duration of the SSML-processed audio was significantly longer, CosyVoice's base model stumbles when encountering XML/HTML tags, either speaking the tag literally or adding excessive silence/padding (e.g., +9.6 seconds for a 10ms break). It does **not** support standard SSML natively.

---

## 2. Multilingual Testing

We updated `scripts/test_multilingual.py` and generated speech in 15 different languages across both models.

### Kokoro v1.0 Multilingual Results
- **Success Rate**: **4 / 15 languages succeeded**
- **Succeeded**: English (`en`), Spanish (`es`), Italian (`it`), Chinese (`zh`).
- **Failed**: 
  - German, Korean, Hindi, Arabic, Russian, Turkish, Dutch, Polish: Rejected by the handler as unsupported (`lang_code` must be one of `a, b, e, f, i, j, p, z`).
  - French (`fr`), Portuguese (`pt`): Hugging Face asset load errors on `hexgrad/Kokoro-82M` (mismatched/missing voice files on HF hub: `ff_sixtine.pt` and `pf_daniele.pt`).
  - Japanese (`ja`): Failed due to missing MeCab/unidic dictionaries in the docker container.

### CosyVoice 2.0 Multilingual Results
- **Success Rate**: **9 / 15 languages succeeded**
- **Succeeded**: English (`en`), Spanish (`es`), French (`fr`), German (`de`), Italian (`it`), Japanese (`ja`), Chinese (`zh`), Korean (`ko`), Russian (`ru`).
- **Failed**: Portuguese (`pt`), Hindi (`hi`), Arabic (`ar`), Turkish (`tr`), Dutch (`nl`), Polish (`pl`) failed validation at the handler.
- **Findings**: CosyVoice correctly synthesized native pronunciation for all 9 supported languages using its zero-shot cross-lingual voice matching.

---

## 3. Benchmarking & Cost Calculations

We ran the cost calculator on the measured Real-Time Factors (RTFs) on L4 GPUs ($0.69/hr):

### Kokoro v1.0
- **Mean RTF**: 0.2319 (4.3x real-time speed)
- **Marginal Cost**: **$0.1600 per hour of generated audio**
- **Commercial Savings**: **8.0x cheaper** than typical commercial TTS APIs ($1.28/hr).

### CosyVoice 2.0
- **Mean RTF**: 1.1249 (0.9x real-time speed)
- **Marginal Cost**: **$0.7762 per hour of generated audio**
- **Commercial Savings**: **1.6x cheaper** than typical commercial TTS APIs ($1.28/hr).

---

## 4. Scoring Rubric Population

We successfully populated the 30-dimension quality evaluation template (`benchmark/scoring_template.csv`) with the results from the evaluation runs, documenting voice naturalness, rhythm, prosody, and technical formatting for both models.
