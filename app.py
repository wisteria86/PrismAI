import os
import time
import json
import torch
import gradio as gr
import concurrent.futures

from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification
from huggingface_hub import InferenceClient

load_dotenv(".env.local")

HF_API_TOKEN = os.environ.get("HF_API_TOKEN", "hf_dummy_token_for_now")
HF_MODEL_COEDIT = os.environ.get("HF_MODEL_COEDIT", "Wisteria86/coedit")
HF_MODEL_EXPLAIN_GEC = os.environ.get("HF_MODEL_EXPLAIN_GEC", "Wisteria86/t5-base-explainable-gec")

HF_MODEL_SCORING_CR = "Wisteria86/my-scoring-models/clarity"
HF_MODEL_SCORING_EQ = "Wisteria86/my-scoring-models/eq"
HF_MODEL_SCORING_GEC = "Wisteria86/my-scoring-models/gec-density"
HF_MODEL_SCORING_IN = "Wisteria86/my-scoring-models/inclusivity"
HF_MODEL_SCORING_PL = "Wisteria86/my-scoring-models/politeness"
HF_MODEL_SCORING_SF = "Wisteria86/my-scoring-models/safety"

actual_token = None if HF_API_TOKEN == "hf_dummy_token_for_now" else HF_API_TOKEN
client = InferenceClient(token=actual_token)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"System mapped to run on: {device}")

print(f"\nLoading local conversion model {HF_MODEL_COEDIT}...")
try:
    coedit_tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_COEDIT)
    coedit_model = AutoModelForSeq2SeqLM.from_pretrained(HF_MODEL_COEDIT).to(device)
    print(f"✅ Successfully loaded {HF_MODEL_COEDIT}")
except Exception as e:
    print(f"❌ Failed to load CoEdit model: {e}")
    coedit_tokenizer = None
    coedit_model = None

print("\nLoading local scoring models...")

scoring_model_paths = {
    "Clarity": HF_MODEL_SCORING_CR,
    "EQ": HF_MODEL_SCORING_EQ,
    "GEC": HF_MODEL_SCORING_GEC,
    "Inclusivity": HF_MODEL_SCORING_IN,
    "Politeness": HF_MODEL_SCORING_PL,
    "Safety": HF_MODEL_SCORING_SF,
}

scoring_tokenizers = {}
scoring_models = {}
tok = AutoTokenizer.from_pretrained("roberta-base")

for name, path in scoring_model_paths.items():
    print(f"Loading {name} ({path})...")
    try:
        parts = path.split('/')
        if len(parts) >= 3 and not os.path.exists(path):
            repo_id = f"{parts[0]}/{parts[1]}"
            subfolder = "/".join(parts[2:])
            mod = AutoModelForSequenceClassification.from_pretrained(repo_id, subfolder=subfolder).to(device)
        else:
            mod = AutoModelForSequenceClassification.from_pretrained(path).to(device)

        scoring_tokenizers[name] = tok
        scoring_models[name] = mod
        print(f"  ✅ {name} loaded.")
    except Exception as e:
        print(f"  ❌ Failed to load {name}: {e}")
        scoring_tokenizers[name] = None
        scoring_models[name] = None

print("\nAll local models processed! Ready to launch PrismAi.")


def call_hf_api(model_id, payload):
    if not HF_API_TOKEN or HF_API_TOKEN == "hf_dummy_token_for_now":
        time.sleep(1.5)
        return {"dummy": True, "model": model_id}
    try:
        response = client.post(json=payload, model=model_id)
        return json.loads(response.decode("utf-8"))
    except Exception as e:
        return {"error": str(e), "model": model_id}


def run_score(text):
    if not text.strip():
        return "Please enter some text to score."

    results = {}
    for name, model in scoring_models.items():
        tokenizer = scoring_tokenizers.get(name)
        if model is None or tokenizer is None:
            results[name] = "Model offline / failed to load."
            continue
        try:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
            with torch.no_grad():
                outputs = model(**inputs)

            logits = outputs.logits[0]

            if name in ["Safety", "Inclusivity"]:
                probs = torch.sigmoid(logits)
                top_val, top_idx = torch.max(probs, dim=0)
                label = model.config.id2label.get(top_idx.item(), f"Category_{top_idx.item()}")
                results[name] = f"{label}  ({top_val.item() * 100:.1f}%)"
            else:
                probs = torch.nn.functional.softmax(logits, dim=0)
                top_val, top_idx = torch.max(probs, dim=0)
                label = model.config.id2label.get(top_idx.item(), f"Label_{top_idx.item()}")
                if label == "LABEL_1" and name in ["Clarity", "GEC"]:
                    label = "Clear / Correct"
                if label == "LABEL_0" and name in ["Clarity", "GEC"]:
                    label = "Unclear / Errors"
                results[name] = f"{label}  ({top_val.item() * 100:.1f}%)"

        except Exception as e:
            results[name] = f"Error: {str(e)}"

    lines = ["| Dimension | Result |", "|---|---|"]
    label_map = {
        "Clarity": "Clarity",
        "EQ": "Emotion / Tone",
        "GEC": "Grammar",
        "Inclusivity": "Inclusivity",
        "Politeness": "Politeness",
        "Safety": "Safety",
    }
    for key, display in label_map.items():
        lines.append(f"| {display} | {results.get(key, '—')} |")
    return "\n".join(lines)


def run_convert(text, action):
    if not text.strip():
        return "Please enter some text to convert."

    if coedit_model is None or coedit_tokenizer is None:
        return "Error: conversion model failed to load. Check terminal logs."

    prompt_prefix = {
        "Fix grammar": "Fix grammar: ",
        "Paraphrase": "Paraphrase: ",
        "Make formal": "Make formal: ",
        "Make informal": "Make informal: ",
        "More complex": "Make more complex: ",
        "Simplify": "Make simpler: ",
        "Clarify": "Make clearer: ",
        "Make polite": "Make It Polite:",
    }.get(action, "")

    inputs = f"{prompt_prefix}{text}"

    try:
        input_ids = coedit_tokenizer(inputs, return_tensors="pt").input_ids.to(device)
        outputs = coedit_model.generate(input_ids, max_length=256)
        result = coedit_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return result
    except Exception as e:
        return f"Conversion error: {str(e)}"


def run_explain(text):
    if not text.strip():
        return "Please enter some text to analyse."

    if HF_API_TOKEN == "hf_dummy_token_for_now":
        time.sleep(1.5)
        return (
            f"[Simulated] 2 grammatical issues found in your text.\n\n"
            f"1. Subject–verb agreement error near the start of the sentence.\n"
            f"2. Inconsistent tense usage — consider keeping everything in past tense.\n\n"
            f"Original: '{text[:60]}{'...' if len(text) > 60 else ''}'"
        )

    data = call_hf_api(HF_MODEL_EXPLAIN_GEC, {"inputs": text})

    if isinstance(data, dict) and "error" in data:
        return f"Error accessing explanation model: {data['error']}"
    if isinstance(data, list) and len(data) > 0:
        return data[0].get("generated_text", str(data))
    return str(data)


custom_css = """
.gradio-container {
    font-family: system-ui, -apple-system, sans-serif !important;
    max-width: 780px !important;
    margin: 0 auto !important;
}

.app-header {
    margin-bottom: 1.5rem;
}
.app-header h1 {
    font-size: 22px !important;
    font-weight: 500 !important;
    letter-spacing: -0.3px;
    margin-bottom: 4px !important;
}
.app-header p {
    font-size: 14px;
    opacity: 0.6;
}

.input-area textarea {
    font-size: 15px !important;
    line-height: 1.65 !important;
    border-radius: 10px !important;
    resize: vertical !important;
}

.tab-nav button[role="tab"] {
    font-size: 14px !important;
    font-weight: 400 !important;
    padding: 8px 20px !important;
    border-radius: 0 !important;
    border-bottom: 2px solid transparent !important;
    background: none !important;
    transition: border-color 0.15s, color 0.15s !important;
}

.tab-nav button[role="tab"].selected {
    font-weight: 500 !important;
    border-bottom: 2px solid currentColor !important;
}

.tab-hint {
    font-size: 13px;
    opacity: 0.55;
    margin-bottom: 1rem;
    line-height: 1.5;
}

.convert-radio label {
    border-radius: 20px !important;
    padding: 5px 14px !important;
    font-size: 13px !important;
    cursor: pointer !important;
    transition: background 0.15s !important;
}

.run-btn button {
    font-size: 14px !important;
    font-weight: 500 !important;
    border-radius: 8px !important;
    padding: 8px 20px !important;
    min-width: 120px !important;
}

.output-text textarea,
.output-md {
    font-size: 14px !important;
    line-height: 1.7 !important;
    border-radius: 8px !important;
}

.score-output table {
    width: 100%;
    border-collapse: collapse;
    font-size: 14px;
}
.score-output th {
    text-align: left;
    font-weight: 500;
    padding: 8px 12px;
    border-bottom: 1px solid rgba(128,128,128,0.2);
}
.score-output td {
    padding: 8px 12px;
    border-bottom: 1px solid rgba(128,128,128,0.08);
}
"""

ACTION_CHOICES = [
    "Fix grammar",
    "Paraphrase",
    "Make formal",
    "Make informal",
    "More complex",
    "Simplify",
    "Clarify",
    "Make polite",
]

with gr.Blocks(theme=gr.themes.Soft(), css=custom_css, title="Prism AI") as demo:

    gr.HTML("""
        <div class="app-header">
            <h1>Prism AI</h1>
            <p>Text transformation, scoring, and grammatical analysis</p>
        </div>
    """)

    text_input = gr.Textbox(
        lines=6,
        placeholder="Paste or type your text here…",
        label="Input text",
        elem_classes=["input-area"],
    )

    with gr.Tabs(elem_classes=["tab-nav"]):

        with gr.Tab("Score"):
            gr.HTML('<p class="tab-hint">Evaluate your text across clarity, grammar, emotion, inclusivity, politeness, and safety.</p>')
            score_btn = gr.Button("Analyse text", variant="primary", elem_classes=["run-btn"])
            score_output = gr.Markdown(
                label="Results",
                elem_classes=["score-output"],
            )
            score_btn.click(fn=run_score, inputs=text_input, outputs=score_output)

        with gr.Tab("Convert"):
            gr.HTML('<p class="tab-hint">Choose a transformation style and rewrite your text.</p>')
            action_radio = gr.Radio(
                choices=ACTION_CHOICES,
                label="Transformation",
                value="Fix grammar",
                elem_classes=["convert-radio"],
            )
            convert_btn = gr.Button("Convert", variant="primary", elem_classes=["run-btn"])
            convert_output = gr.Textbox(
                label="Converted text",
                lines=5,
                interactive=False,
                placeholder="Converted text will appear here…",
                elem_classes=["output-text"],
            )
            convert_btn.click(fn=run_convert, inputs=[text_input, action_radio], outputs=convert_output)

        with gr.Tab("Explain"):
            gr.HTML('<p class="tab-hint">Detect grammatical issues and get plain-English explanations of suggested corrections.</p>')
            explain_btn = gr.Button("Explain issues", variant="primary", elem_classes=["run-btn"])
            explain_output = gr.Textbox(
                label="Analysis",
                lines=6,
                interactive=False,
                placeholder="Grammatical analysis will appear here…",
                elem_classes=["output-text"],
            )
            explain_btn.click(fn=run_explain, inputs=text_input, outputs=explain_output)

if __name__ == "__main__":
    demo.launch()