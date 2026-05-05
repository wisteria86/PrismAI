---
title: PrismAI Text Intelligence Panel
emoji: 🌈
colorFrom: green
colorTo: pink
sdk: gradio
sdk_version: 5.9.1
app_file: app.py
pinned: false
license: mit
---

<div align="center">

# 🌈 PrismAI — Text Intelligence Panel

**A multi-dimensional NLP dashboard for text transformation, quality scoring, and grammatical analysis.**  
Built on fine-tuned Hugging Face models, deployable in one command — locally or on Hugging Face Spaces.

[![Hugging Face Space](https://img.shields.io/badge/🤗%20Hugging%20Face-Space-blue)](https://huggingface.co/spaces/Wisteria86/PrismAI)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Gradio](https://img.shields.io/badge/Gradio-5.9.1-orange.svg)](https://gradio.app/)
[![Model: CoEdit](https://img.shields.io/badge/Model-Wisteria86%2Fcoedit-purple)](https://huggingface.co/Wisteria86/coedit)

</div>

---

## 📖 What is PrismAI?

PrismAI is a text intelligence panel that lets you analyse and rewrite any piece of text across **six quality dimensions** and **eight transformation styles** — all through a clean, tab-driven Gradio interface.

It brings together three categories of NLP models under one roof:

| Category | What it does |
|---|---|
| **Scoring** | Rates your text on Clarity, Grammar, Emotion/Tone, Inclusivity, Politeness, and Safety |
| **Conversion** | Rewrites text in a chosen style (formal, simple, polite, paraphrased, etc.) using a fine-tuned CoEdit model |
| **Explain** | Detects grammatical issues and returns plain-English explanations of each correction |

The models are loaded locally at startup and run entirely on your machine (CPU or CUDA GPU) — no external API calls required for scoring and conversion.

---

## ✨ Feature Highlights

### 🔢 Score Tab
Runs your text through **six fine-tuned classifiers** in parallel and returns a formatted markdown results table:

| Dimension | Model Used |
|---|---|
| **Clarity** | `Wisteria86/my-scoring-models/clarity` |
| **Emotion / Tone** | `Wisteria86/my-scoring-models/eq` |
| **Grammar** | `Wisteria86/my-scoring-models/gec-density` |
| **Inclusivity** | `Wisteria86/my-scoring-models/inclusivity` |
| **Politeness** | `Wisteria86/my-scoring-models/politeness` |
| **Safety** | `Wisteria86/my-scoring-models/safety` |

Each dimension uses softmax or sigmoid classification to produce a label and a confidence percentage.

### ✍️ Convert Tab
Uses `Wisteria86/coedit` — a **custom fine-tuned Flan-T5-base model** trained on the [Grammarly CoEdit dataset](https://github.com/grammarly/coedit) — to rewrite text in eight styles:

- Fix grammar
- Paraphrase
- Make formal
- Make informal
- More complex
- Simplify
- Clarify
- Make polite

### 🔍 Explain Tab
Calls `ATG2222/t5-base-explainable-gec` (accessible via the Hugging Face Inference API) to detect grammatical errors and return plain-English explanations of each correction — including the original phrase, the fix, and the reason.

---

## 🚀 Getting Started

### Prerequisites

- Python **3.10+**
- `pip`
- *(Optional but recommended)* A CUDA-capable GPU for faster inference
- A Hugging Face account + API token if you want to use the **Explain** tab in production

---

### 1. Clone the Repository

```bash
git clone https://github.com/Wisteria86/PrismicEditor.git
cd PrismicEditor
```

---

### 2. Create a Virtual Environment

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python -m venv .venv
source .venv/bin/activate
```

---

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

> **requirements.txt** installs:
> ```
> gradio
> python-dotenv
> huggingface_hub>=0.20.0
> transformers
> torch
> sentencepiece
> tiktoken
> ```

> **Tip:** If you have a CUDA GPU, install the CUDA-enabled version of PyTorch first for significantly faster inference:
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cu121
> ```

---

### 4. Configure Environment Variables

Create a `.env.local` file in the project root:

```bash
# .env.local

# Optional — override default model IDs
HF_MODEL_COEDIT=Wisteria86/coedit
HF_MODEL_EXPLAIN_GEC=ATG2222/t5-base-explainable-gec
```

---

### 5. Launch the App

```bash
python app.py
```

On first launch, the app will:
1. Detect whether CUDA is available and map models accordingly
2. Download and cache `Wisteria86/coedit` from Hugging Face.
3. Load all six scoring classifiers from the HF Hub (cached after first run)
4. Start the Gradio server (default: `http://127.0.0.1:7860`)


---

## 🖥️ Usage Guide

### Interface Overview

```
┌─────────────────────────────────────────────┐
│              Prism AI                        │
│  Text transformation, scoring, and analysis │
├─────────────────────────────────────────────┤
│  [Input Text Box — paste or type here]      │
├────────┬───────────┬───────────────────────┤
│  Score │  Convert  │  Explain              │
├────────┴───────────┴───────────────────────┤
│  [Tab-specific controls & output]           │
└─────────────────────────────────────────────┘
```

### Step-by-step

1. **Paste or type** your text into the input box at the top.
2. **Choose a tab:**
   - **Score** → Click *Analyse text* to get a quality report table.
   - **Convert** → Select a transformation style, then click *Convert*.
   - **Explain** → Click *Explain issues* for a plain-English grammar report.
3. Results appear below the button in real time.

---

### Example: Score Tab

**Input:**
> "He don't like going to store and buy thing when he have time for it."

**Output:**

| Dimension | Result |
|---|---|
| Clarity | Unclear / Errors  (91.3%) |
| Emotion / Tone | neutral  (82.7%) |
| Grammar | Unclear / Errors  (95.1%) |
| Inclusivity | Non-Exclusionary  (88.4%) |
| Politeness | polite  (79.2%) |
| Safety | non-toxic  (97.6%) |

---

### Example: Convert Tab

**Input:** `"He don't like going to store."`  
**Style:** Fix grammar  
**Output:** `"He doesn't like going to the store."`

**Input:** `"I need the report by EOD."`  
**Style:** Make formal  
**Output:** `"I require the report to be submitted by the end of the working day."`

---

## 🏋️ Training Your Own Models

This repository also includes the full training pipeline used to produce the custom models.

### CoEdit Model (`train.py`)

Fine-tunes `google/flan-t5-base` on a custom `master.csv` dataset for multi-task text editing:

```bash
# Prepare your dataset as master.csv with columns: src, tgt
python train.py
```

Training configuration:
- Base model: `google/flan-t5-base`
- Optimiser: Adafactor
- Precision: BF16
- Batch size: 4 (with 4× gradient accumulation = effective batch 16)
- Epochs: 1 (extend as needed)
- Output: `./coedit/`

### Explainable GEC Model (`train.py`)

Fine-tunes `vennify/t5-base-grammar-correction` on BEA-2019 and CoNLL-2014 `.m2` annotation files, then wraps it with a token-diff `ExplainableAIJudge` to produce human-readable correction rationale.

### Scoring Models (`train.py`)

A reusable `MetricConfig` + `run_training()` framework for training DeBERTa-base classifiers/regressors on six dimensions:

| Dimension | Dataset | Task |
|---|---|---|
| EQ / Emotion | `google-research-datasets/go_emotions` | Regression |
| Clarity | `casey-martin/CommonLit-Ease-of-Readability` | Regression |
| GEC Density | `grammarly/coedit` + `jhu-clsp/jfleg` | Regression |
| Inclusivity | `ucberkeley-dlab/measuring-hate-speech` + Jigsaw | Regression |

### Evaluation (`eval.py`)

Evaluates the CoEdit model against a curated test set using ROUGE and BLEU:

```bash
python eval.py
```

Loads `Wisteria86/coedit` at tag `v2.0.1` and reports:

```
========================================
ROUGE-1: 0.xxxx
ROUGE-L: 0.xxxx
BLEU:    0.xxxx
========================================
```

---

## 🤖 Model Architecture

```
PrismAI
│
├── Conversion Engine
│   └── Wisteria86/coedit
│       Base: google/flan-t5-base
│       Fine-tuned on: custom master.csv (multi-task editing pairs)
│       Task: Seq2Seq text transformation
│
├── Scoring Engine (6× classifiers)
│   ├── Clarity   → Wisteria86/my-scoring-models/clarity
│   ├── Emotion   → Wisteria86/my-scoring-models/eq
│   ├── Grammar   → Wisteria86/my-scoring-models/gec-density
│   ├── Inclusivity → Wisteria86/my-scoring-models/inclusivity
│   ├── Politeness  → Wisteria86/my-scoring-models/politeness
│   └── Safety    → Wisteria86/my-scoring-models/safety
│
└── Explain Engine
    └── ATG2222/t5-base-explainable-gec
        Base: vennify/t5-base-grammar-correction
        Fine-tuned on: BEA-2019 + CoNLL-2014 (.m2 format)
        Task: GEC with token-level diff rationale
```

---

## ⚙️ Configuration Reference

All runtime configuration is controlled via `.env.local`:

| Variable | Default | Description |
|---|---|---|
| `HF_API_TOKEN` | `hf_dummy_token_for_now` | Hugging Face API token. Required for Explain tab. |
| `HF_MODEL_COEDIT` | `Wisteria86/coedit` | HF repo ID for the conversion model |
| `HF_MODEL_EXPLAIN_GEC` | `Wisteria86/t5-base-explainable-gec` | HF repo ID for the GEC explanation model |

---

## 🛠️ Troubleshooting

| Problem | Solution |
|---|---|
| Model download is slow | Models are cached after first download in `~/.cache/huggingface/`. Subsequent launches are instant. |
| `CUDA out of memory` | Reduce text length, or set `device = cpu` in `app.py` line 28 if your GPU VRAM is limited. |
| Explain tab returns simulated text | Set a valid `HF_API_TOKEN` in `.env.local`. |
| A scoring model fails to load | Check your internet connection; the error is printed to the terminal and the dimension will show "Model offline". |
| `sentencepiece` not found | Run `pip install sentencepiece` (should be in requirements.txt already). |

---

## 📄 License

This project is released under the **MIT License**. See [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- [Grammarly CoEdit](https://github.com/grammarly/coedit) — dataset and inspiration for multi-task text editing
- [Hugging Face](https://huggingface.co/) — model hosting, Transformers library, and Gradio Spaces
- [Gradio](https://gradio.app/) — UI framework
- BEA-2019 and CoNLL-2014 shared task organisers — GEC benchmark datasets
- All authors of the open-source models used in the scoring engine

---

