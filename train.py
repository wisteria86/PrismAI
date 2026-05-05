
import os
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)

from spellchecker import SpellChecker
import re
import os
from collections import defaultdict
import tarfile
import os
import glob
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from torch.utils.data import Dataset
import difflib
import random
import math


os.makedirs('/content/data', exist_ok=True)


# || COEDIT MODEL TRAINING ||

MODEL_NAME = "google/flan-t5-base"
OUTPUT_DIR = "./coedit"
MAX_INPUT_LENGTH = 128
MAX_TARGET_LENGTH = 128

print("Loading and Tokenizing Dataset...")

raw_data = pd.read_csv('./master.csv')
full_dataset = Dataset.from_pandas(raw_data)
dataset = full_dataset.train_test_split(test_size=0.1)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def preprocess(examples):
    model_inputs = tokenizer(examples["src"], max_length=MAX_INPUT_LENGTH, truncation=True, padding="max_length")
    labels = tokenizer(examples["tgt"], max_length=MAX_TARGET_LENGTH, truncation=True, padding="max_length")
    
    labels_with_ignore_index = []
    for label_sequence in labels["input_ids"]:
        labels_with_ignore_index.append([l if l != tokenizer.pad_token_id else -100 for l in label_sequence])
        
    model_inputs["labels"] = labels_with_ignore_index
    return model_inputs

print("Mapping dataset...")
tokenized_datasets = dataset.map(
    preprocess, 
    batched=True, 
    remove_columns=dataset["train"].column_names
)


print("Loading Model to GPU...")
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
model.config.use_cache = False

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    save_strategy="steps",             
    save_steps=1000,                   
    eval_strategy="steps",
    eval_steps=1000,
    save_total_limit=2,            
    logging_steps=5,                  
    learning_rate=2e-5,               
    
    fp16=False,    
    bf16=True,     
    
    per_device_train_batch_size=4,     
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,
    
    dataloader_num_workers=2,
    optim="adafactor",               
    num_train_epochs=1,                
    push_to_hub=False,
    logging_dir=f"{OUTPUT_DIR}/logs",
    report_to="none", 
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"], 
    eval_dataset=tokenized_datasets["test"],   
    tokenizer=tokenizer,
    data_collator=data_collator,
)

print("Starting GPU Training...")
trainer.train()

print(f"Saving to {OUTPUT_DIR}...")
trainer.save_model(OUTPUT_DIR) 
tokenizer.save_pretrained(OUTPUT_DIR)
print("Training Complete!")

# ||  EXPLAINABLE GEC TRAINING IMPLEMENTATION  ||


def extract_tar(path, extract_path):
    if os.path.exists(path):
        with tarfile.open(path, 'r:gz') as tar:
            tar.extractall(path=extract_path)

extract_tar('/content/wi+locness_v2.1.bea19.tar.gz', '/content/data/bea')
extract_tar('/content/conll14st-test-data.tar.gz', '/content/data/conll')

bea_m2 = glob.glob('/content/data/bea/**/*.m2', recursive=True)
conll_m2 = glob.glob('/content/data/conll/**/*.m2', recursive=True)
print(f"Datasets ready: {len(bea_m2)} BEA files, {len(conll_m2)} CoNLL files found.")

def parse_m2(file_path):
    data = []
    with open(file_path, 'r') as f:
        source = None
        for line in f:
            if line.startswith('S '): source = line[2:].strip()
            elif line.startswith('A '):
                parts = line[2:].split('|||')
                if len(parts) > 1 and parts[1] != 'noop':
                    data.append({'source': source, 'target': parts[2], 'type': parts[1]})
    return data

all_train_data = []
for f in (bea_m2 + conll_m2): all_train_data.extend(parse_m2(f))

MODEL_NAME = 'vennify/t5-base-grammar-correction'
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)

class GECDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        item = self.data[idx]
        inputs = self.tokenizer("gec: " + item['source'], max_length=self.max_len, padding='max_length', truncation=True, return_tensors="pt")
        labels = self.tokenizer(item['target'], max_length=self.max_len, padding='max_length', truncation=True, return_tensors="pt")
        return {'input_ids': inputs.input_ids.squeeze(), 'attention_mask': inputs.attention_mask.squeeze(), 'labels': labels.input_ids.squeeze()}

train_dataset = GECDataset(all_train_data, tokenizer)

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=100,
    save_total_limit=2,
    logging_dir='./logs',
    remove_unused_columns=False
)

train_dataset = GECDataset(all_train_data, tokenizer)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator
)

print(f"Starting training on {len(all_train_data)} samples...")
trainer.train()

error_map = {
    'R:VERB:TENSE': 'The timing of the action (verb tense) seems a bit off.',
    'R:VERB:SVA': "The subject and verb don't quite match up.",
    'R:SPELL': 'Looks like a small spelling typo here.',
    'R:PUNCT': 'Let\'s adjust the punctuation for better flow.',
    'M:PUNCT': 'Adding a little punctuation here would help.',
    'U:PUNCT': 'We can remove this extra punctuation.',
    'R:NOUN:NUM': 'Let\'s check if this should be singular or plural.',
    'R:PREP': 'A different preposition might fit better here.',
    'M:DET': 'Adding a word like "the" or "a" makes this clearer.',
    'R:OTHER': 'Just a quick polish for style and clarity.',
    'M:OTHER': 'Adding a missing word to complete the thought.',
    'U:OTHER': 'Removing an extra word to keep it concise.'
}

class ExplainableAIJudge:
    def __init__(self, model, tokenizer, error_map):
        self.model = model
        self.tokenizer = tokenizer
        self.error_map = error_map
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def analyze(self, text, task='gec'):
        prompts = {
            'gec': 'Fix grammatical errors in this sentence: ',
        }
        prefix = prompts.get(task, 'gec: ')

        input_ids = self.tokenizer(prefix + text, return_tensors='pt').input_ids.to(self.device)
        outputs = self.model.generate(input_ids, max_length=128)
        corrected = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        s_toks, c_toks = text.split(), corrected.split()
        matcher = difflib.SequenceMatcher(None, s_toks, c_toks)
        changes = []

        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'equal': continue

            original = ' '.join(s_toks[i1:i2])
            fixed = ' '.join(c_toks[j1:j2])

            if tag == 'replace':
                code = 'R:OTHER'
                if any(p in original or p in fixed for p in '.,!?;:'): code = 'R:PUNCT'
            elif tag == 'delete':
                code = 'U:OTHER'
                if any(p in original for p in '.,!?;:'): code = 'U:PUNCT'
            elif tag == 'insert':
                code = 'M:OTHER'
                if any(p in fixed for p in '.,!?;:'): code = 'M:PUNCT'

            changes.append({
                'original': original,
                'fixed': fixed,
                'explanation': self.error_map.get(code, 'A friendly refinement for clarity.')
            })
        return {'fixed': corrected, 'changes': changes}

judge = ExplainableAIJudge(model, tokenizer, error_map)

# Multi-Task Inference & Linguistic Rationale
test_cases = [
    ("Hey how are you", "gec"),
]

for text, task in test_cases:
    result = judge.analyze(text, task=task)
    print(f"--- Task: {task.upper()} ---")
    print(f"Input: {text}")
    print("\nLinguistic Explanation & Correction Rationale:")

    if not result['changes']:
        print("  - No significant linguistic adjustments required.")
    else:
        for c in result['changes']:
            # Combining error identification with correction rationale
            rationale = f"Correction of '{c['original']}' to '{c['fixed']}' due to: {c['explanation']}"
            print(f"  • {rationale}")
    print("\n" + "="*45 + "\n")

# testing the specific sentence and additional samples
additional_test_cases = [
    ("I are very goodest at the programming, it are my favorites thing.", "gec"),
    ("He don't like apple.", "gec"),
    ("The children is playing in park", "gec"),
    ("i live in london", "gec")
]

for text, task in additional_test_cases:
    result = judge.analyze(text, task=task)
    print(f"--- Task: {task.upper()} ---")
    print(f"Original: {text}")
    print(f"Corrected: {result['fixed']}")
    print("\nRecommendations:")

    if not result['changes']:
        print("  - Everything looks great!")
    else:
        for c in result['changes']:
            print(f"  • '{c['original']}' → '{c['fixed']}': {c['explanation']}")
    print("\n" + "="*50 + "\n")
    
    
# || SCORING MODELS ||

# BaseTrainer.py

import os
import json
import torch
import numpy as np
from dataclasses import dataclass
from typing import Optional, Callable

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
)
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, f1_score


@dataclass
class MetricConfig:
    metric_name:    str
    dataset_loader: Callable
    output_dir:     str                = "outputs"
    model_name:     str                = "microsoft/deberta-base"
    task_type:      str                = "regression"
    max_length:     int                = 128
    epochs:         int                = 4
    batch_size:     int                = 16
    lr:             float              = 2e-5
    weight_decay:   float              = 0.01
    eval_strategy:  str                = "epoch"
    fp16:           bool               = False
    bf16:           bool               = False
    seed:           int                = 42
    label_smoother: Optional[Callable] = None


def tokenize_dataset(dataset: Dataset, tokenizer, max_length: int, task_type: str) -> Dataset:
    def _tokenize(batch):
        if "text_pair" in batch:
            tokens = tokenizer(batch["text"], batch["text_pair"], truncation=True, max_length=max_length)
        else:
            tokens = tokenizer(batch["text"], truncation=True, max_length=max_length)
        if task_type == "classification":
            tokens["labels"] = [int(l) for l in batch["label"]]
        else:
            tokens["labels"] = [float(l) for l in batch["label"]]
        return tokens
    return dataset.map(
        _tokenize, batched=True,
        remove_columns=[c for c in ["text", "text_pair", "label"] if c in dataset.column_names]
    )


def compute_regression_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.squeeze()
    if np.isnan(preds).any():
        preds = np.where(np.isnan(preds), 0.5, preds)
    preds = np.clip(preds, 0.0, 1.0)
    mse  = mean_squared_error(labels, preds)
    mae  = mean_absolute_error(labels, preds)
    rmse = np.sqrt(mse)
    return {"mse": mse, "rmse": rmse, "mae": mae}


def compute_classification_metrics(eval_pred):
    logits, labels = eval_pred
    if np.isnan(logits).any():
        return {"accuracy": 0.0, "f1": 0.0, "rmse": 1.0}
    preds = np.argmax(logits, axis=-1)
    acc  = accuracy_score(labels, preds)
    f1   = f1_score(labels, preds, average="binary", zero_division=0)
    probs = torch.softmax(torch.tensor(logits, dtype=torch.float32), dim=-1)[:, 1].numpy()
    probs = np.clip(probs, 0.0, 1.0)
    rmse = np.sqrt(mean_squared_error(labels, probs))
    return {"accuracy": acc, "f1": f1, "rmse": rmse}


def patch_layernorm_names(model):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.LayerNorm):
            if hasattr(module, 'gamma'):
                module.weight = module.gamma
                del module.__dict__['gamma']
            if hasattr(module, 'beta'):
                module.bias = module.beta
                del module.__dict__['beta']
    print("    LayerNorm gamma/beta patched")


def run_training(config: MetricConfig):
    is_cls = config.task_type == "classification"

    print(f"\n{'='*60}")
    print(f"  Training : {config.metric_name.upper()} Scoring Model")
    print(f"  Task     : {config.task_type}")
    print(f"  Model    : {config.model_name}")
    print(f"  Device   : {'CUDA (' + torch.cuda.get_device_name(0) + ')' if torch.cuda.is_available() else 'CPU'}")
    print(f"{'='*60}\n")

    print("[1/5] Loading dataset...")
    full_dataset = config.dataset_loader()

    if config.label_smoother:
        full_dataset = full_dataset.map(lambda x: {"label": config.label_smoother(x["label"])})

    assert "text"  in full_dataset.column_names
    assert "label" in full_dataset.column_names

    split1   = full_dataset.train_test_split(test_size=0.2,  seed=config.seed)
    split2   = split1["test"].train_test_split(test_size=0.5, seed=config.seed)
    train_ds = split1["train"]
    val_ds   = split2["train"]
    test_ds  = split2["test"]
    print(f"    Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    print("[2/5] Tokenizing...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    train_ds  = tokenize_dataset(train_ds, tokenizer, config.max_length, config.task_type)
    val_ds    = tokenize_dataset(val_ds,   tokenizer, config.max_length, config.task_type)
    test_ds   = tokenize_dataset(test_ds,  tokenizer, config.max_length, config.task_type)

    print("[3/5] Loading model...")
    if is_cls:
        model = AutoModelForSequenceClassification.from_pretrained(
            config.model_name, num_labels=2, ignore_mismatched_sizes=True,
        )
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            config.model_name, num_labels=1, problem_type="regression",
            ignore_mismatched_sizes=True,
        )

    patch_layernorm_names(model)

    if hasattr(model, 'classifier') and hasattr(model.classifier, 'weight'):
        torch.nn.init.normal_(model.classifier.weight, mean=0.0, std=0.002)
        torch.nn.init.zeros_(model.classifier.bias)
        print("    Head initialised (std=0.002)")

    save_dir = os.path.join(config.output_dir, config.metric_name)
    os.makedirs(save_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir                  = save_dir,
        num_train_epochs            = config.epochs,
        per_device_train_batch_size = config.batch_size,
        per_device_eval_batch_size  = config.batch_size * 2,
        learning_rate               = config.lr,
        weight_decay                = config.weight_decay,
        eval_strategy               = config.eval_strategy,
        save_strategy               = "no",
        load_best_model_at_end      = False,
        gradient_accumulation_steps = 2,
        fp16                        = config.fp16,
        bf16                        = config.bf16,
        seed                        = config.seed,
        logging_steps               = 50,
        report_to                   = "none",
        dataloader_num_workers      = 0,
        lr_scheduler_type           = "linear",
        max_grad_norm               = 0.3,
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = Trainer(
        model           = model,
        args            = training_args,
        train_dataset   = train_ds,
        eval_dataset    = val_ds,
        compute_metrics = compute_classification_metrics if is_cls else compute_regression_metrics,
        data_collator   = data_collator,
    )

    print("[4/5] Training...")
    trainer.train()

    print("[5/5] Evaluating on test set...")
    test_results = trainer.evaluate(test_ds)
    print(f"\n  Test Results for [{config.metric_name}]:")
    for k, v in test_results.items():
        print(f"    {k}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")

    final_path = os.path.join(save_dir, "final_model")
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)

    with open(os.path.join(final_path, "scorer_meta.json"), "w") as f:
        json.dump({"task_type": config.task_type, "metric_name": config.metric_name}, f, indent=2)

    print(f"\n  Model saved → {final_path}")

    with open(os.path.join(save_dir, "test_metrics.json"), "w") as f:
        json.dump(test_results, f, indent=2)
    print(f"  Metrics saved → {save_dir}/test_metrics.json\n")

    print("\n  LIVE INFERENCE SANITY CHECK")
    sanity_sentences = [
        "i will kill you bitch",
        "Could you please review this when you get a chance? I appreciate your time.",
        "I'm going to make you regret this. Watch your back.",
        "She goes then returns oiehfikolsehfiosehfiosehefeisajhfoisehfback.",
        "I can't believe she's gone. I feel empty, scared, and completely lost.",
    ]
    trained_model = trainer.model
    trained_model.eval()
    for sent in sanity_sentences:
        enc = tokenizer(sent, return_tensors="pt", truncation=True, max_length=128)
        enc = {k: v.to(trained_model.device) for k, v in enc.items()}
        with torch.no_grad():
            val = trained_model(**enc).logits.squeeze(-1).item()
        print(f"    {val:.4f}  {sent[:60]}")

    return test_results


def load_eq_data():
    ds = load_dataset("google-research-datasets/go_emotions", "simplified")
    rows = []
    for split in ["train", "validation", "test"]:
        if split not in ds:
            continue
        for example in ds[split]:
            text   = example.get("text", "").strip()
            labels = example.get("labels", [])
            if not text:
                continue
            non_neutral_active = sum(1 for l in labels if l < 27)
            eq_score = min(non_neutral_active / 3.0, 1.0)
            rows.append({"text": text, "label": eq_score})
    return Dataset.from_list(rows)


if __name__ == "__main__":
    config = MetricConfig(
        metric_name    = "eq",
        dataset_loader = load_eq_data,
        output_dir     = "saved_models",
        epochs         = 4,
        batch_size     = 16,
        lr             = 2e-5,
        bf16           = False,
        fp16           = False,
        model_name     = "microsoft/deberta-base",
    )
    run_training(config)

# train_clarity.py


_TEXT_COLS  = ["Excerpt", "excerpt", "sentence", "text"]
_SCORE_COLS = ["BT_easiness", "BT Easiness", "bt_easiness"]


def _find_col(keys, candidates, label):
    for col in candidates:
        if col in keys:
            return col
    raise KeyError(f"Could not find {label} column. Available: {keys}. Tried: {candidates}")


def augment_text(text: str) -> str:
    sentences = text.split(". ")
    if len(sentences) > 2 and random.random() < 0.3:
        sentences = sentences[:-1]
    return ". ".join(sentences).strip()


def load_clarity_data():
    ds  = load_dataset("casey-martin/CommonLit-Ease-of-Readability")
    raw = ds["train"]
    cols      = raw.column_names
    text_col  = _find_col(cols, _TEXT_COLS,  "text/excerpt")
    score_col = _find_col(cols, _SCORE_COLS, "BT_easiness")

    def preprocess(example):
        return {
            "text":  str(example[text_col] or ""),
            "label": float(example[score_col]) if example[score_col] is not None else None,
        }

    original  = raw.map(preprocess, remove_columns=cols)
    original  = original.filter(lambda x: x["label"] is not None)
    augmented = original.map(lambda x: {"text": augment_text(x["text"]), "label": x["label"]})
    combined  = concatenate_datasets([original, augmented]).shuffle(seed=42)

    scores = combined["label"]
    min_s, max_s = min(scores), max(scores)
    combined = combined.map(lambda x: {"label": (x["label"] - min_s) / (max_s - min_s + 1e-8)})
    return combined


if __name__ == "__main__":
    config = MetricConfig(
        metric_name    = "clarity",
        dataset_loader = load_clarity_data,
        output_dir     = "saved_models",
        epochs         = 6,
        batch_size     = 16,
        lr             = 1e-5,
        bf16           = False,
        fp16           = False,
        model_name     = "microsoft/deberta-base",
    )
    run_training(config)
    

# GEC 



def char_edit_distance(s1: str, s2: str) -> int:
    m, n = len(s1), len(s2)
    if m == 0: return n
    if n == 0: return m
    prev = list(range(n + 1))
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        curr[0] = i
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                curr[j] = prev[j-1]
            else:
                curr[j] = 1 + min(prev[j], curr[j-1], prev[j-1])
        prev, curr = curr, [0] * (n + 1)
    return prev[n]


def normalized_char_edit(s1: str, s2: str) -> float:
    if not s1 and not s2:
        return 0.0
    raw = char_edit_distance(s1[:300], s2[:300]) / max(len(s1), len(s2), 1)
    return min(raw, 0.6) / 0.6


def safe_label(v: float) -> float:
    return float(max(0.0, min(1.0, v))) if math.isfinite(v) else 0.0


def load_coedit(cap: int) -> Dataset:
    raw  = load_dataset("grammarly/coedit", split="train")
    rows = []
    for ex in raw:
        src = str(ex.get("src", "")).strip()
        tgt = str(ex.get("tgt", "")).strip()
        if not src or not tgt:
            continue
        rows.append({"text": src, "label": safe_label(normalized_char_edit(src, tgt))})
    random.seed(42)
    random.shuffle(rows)
    if len(rows) > cap:
        rows = rows[:cap]
    return Dataset.from_list(rows)


def load_grammar_correction(cap: int) -> Dataset:
    raw  = load_dataset("agentlans/grammar-correction", split="train")
    rows = []
    for ex in raw:
        src = str(ex.get("input",  "")).strip()
        tgt = str(ex.get("output", "")).strip()
        if not src or not tgt or src == tgt:
            continue
        density = normalized_char_edit(src, tgt)
        if density < 0.02:
            continue
        rows.append({"text": src, "label": safe_label(density)})
        if len(rows) >= cap:
            break
    random.seed(42)
    random.shuffle(rows)
    return Dataset.from_list(rows)


def load_jfleg() -> Dataset:
    rows = []
    for split in ["validation", "test"]:
        raw = load_dataset("jhu-clsp/jfleg", split=split)
        for ex in raw:
            src         = str(ex.get("sentence", "")).strip()
            corrections = ex.get("corrections", [])
            if not src or not corrections:
                continue
            densities = [normalized_char_edit(src, str(c).strip()) for c in corrections if str(c).strip()]
            if not densities:
                continue
            rows.append({"text": src, "label": safe_label(float(np.mean(densities)))})
    return Dataset.from_list(rows).shuffle(seed=42)


def load_gec_density_data() -> Dataset:
    coedit = load_coedit(cap=80_000)
    jfleg  = load_jfleg()
    c4     = load_grammar_correction(cap=len(coedit))
    combined = concatenate_datasets([coedit, c4, jfleg]).shuffle(seed=42)
    return combined.filter(lambda x: x["label"] is not None and math.isfinite(x["label"]))


if __name__ == "__main__":
    config = MetricConfig(
        metric_name    = "gec_density",
        task_type      = "regression",
        dataset_loader = load_gec_density_data,
        output_dir     = "saved_models",
        epochs         = 4,
        batch_size     = 16,
        lr             = 1e-5,
        bf16           = False,
        fp16           = False,
        max_length     = 128,
        model_name     = "microsoft/deberta-base",
    )
    run_training(config)
    
# Inclusivity.py

import math
import numpy as np
import requests
import tarfile
import csv
import io
from datasets import load_dataset, Dataset, concatenate_datasets
from base_trainer import MetricConfig, run_training


def safe_label(value: float):
    if not math.isfinite(value):
        return None
    return max(0.0, min(1.0, value))


def load_ucberkeley() -> Dataset:
    raw = load_dataset("ucberkeley-dlab/measuring-hate-speech", "default", split="train")
    records = {}
    texts   = {}
    for ex in raw:
        cid   = str(ex.get("comment_id", ""))
        text  = str(ex.get("text", "")).strip()
        score = ex.get("hate_speech_score", None)
        if not cid or not text or score is None:
            continue
        try:
            score = float(score)
        except (ValueError, TypeError):
            continue
        if not math.isfinite(score):
            continue
        records.setdefault(cid, []).append(score)
        texts[cid] = text

    all_means = [float(np.mean(v)) for v in records.values()]
    s_min, s_max = min(all_means), max(all_means)
    s_range = s_max - s_min if s_max != s_min else 1.0

    rows = []
    for cid, score_list in records.items():
        mean   = float(np.mean(score_list))
        normed = (mean - s_min) / s_range
        lbl    = safe_label(1.0 - normed)
        if lbl is None:
            continue
        rows.append({"text": texts[cid][:512], "label": lbl})

    ds = Dataset.from_list(rows).shuffle(seed=42)
    print(f"  → UC Berkeley: {len(ds)} unique comments")
    return ds


def load_davidson(cap: int) -> Dataset:
    raw  = load_dataset("tdavidson/hate_speech_offensive", split="train")
    rows = []
    for ex in raw:
        text = str(ex.get("tweet", "")).strip()
        if not text:
            continue
        hate_c    = int(ex.get("hate_speech_count",        0) or 0)
        offensive = int(ex.get("offensive_language_count", 0) or 0)
        neither   = int(ex.get("neither_count",            0) or 0)
        total     = hate_c + offensive + neither
        if total == 0:
            continue
        offensiveness = (hate_c * 1.0 + offensive * 0.5) / total
        lbl = safe_label(1.0 - offensiveness)
        if lbl is None:
            continue
        rows.append({"text": text[:512], "label": lbl})
    ds = Dataset.from_list(rows).shuffle(seed=42)
    return ds.select(range(min(cap, len(ds))))


def load_jigsaw(cap: int) -> Dataset:
    raw  = load_dataset("Arsive/toxicity_classification_jigsaw", split="train")
    rows = []
    for ex in raw:
        text = str(ex.get("comment_text", "")).strip()
        if not text:
            continue
        id_hate = int(ex.get("identity_hate", 0) or 0)
        toxic   = int(ex.get("toxic",         0) or 0)
        insult  = int(ex.get("insult",        0) or 0)
        threat  = int(ex.get("threat",        0) or 0)
        excl    = min(1.0, id_hate * 0.5 + threat * 0.3 + toxic * 0.1 + insult * 0.1)
        lbl     = safe_label(1.0 - excl)
        if lbl is None:
            continue
        rows.append({"text": text[:512], "label": lbl})
    ds = Dataset.from_list(rows).shuffle(seed=42)
    return ds.select(range(min(cap, len(ds))))


def load_sbic(cap: int) -> Dataset:
    SBIC_URL = "https://homes.cs.washington.edu/~msap/social-bias-frames/SBIC.v2.tgz"
    print("  → Downloading Social Bias Frames...")
    try:
        r = requests.get(SBIC_URL, timeout=120, stream=True)
        r.raise_for_status()
    except Exception as e:
        print(f"  → SBIC download failed ({e}), skipping.")
        return Dataset.from_list([])

    rows = []
    with tarfile.open(fileobj=io.BytesIO(r.content), mode="r:gz") as tar:
        for member in tar.getmembers():
            if "SBIC.v2.trn.csv" not in member.name:
                continue
            f = tar.extractfile(member)
            if f is None:
                continue
            reader = csv.DictReader(io.TextIOWrapper(f, encoding="utf-8"))
            for record in reader:
                text = str(record.get("post", "")).strip()
                offensive_str = str(
                    record.get("offensiveScore", record.get("offensiveYN", ""))
                ).strip()
                if not text or not offensive_str:
                    continue
                try:
                    offensive = float(offensive_str)
                except ValueError:
                    continue
                lbl = safe_label(1.0 - offensive)
                if lbl is None:
                    continue
                rows.append({"text": text[:512], "label": lbl})

    ds = Dataset.from_list(rows).shuffle(seed=42)
    return ds.select(range(min(cap, len(ds))))


def load_civil_comments(cap: int) -> Dataset:
    raw  = load_dataset("google/civil_comments", split="train")
    raw  = raw.shuffle(seed=42).select(range(min(300_000, len(raw))))
    rows = []
    for ex in raw:
        text            = str(ex.get("text", "")).strip()
        if not text:
            continue
        toxicity        = float(ex.get("toxicity",        0.0) or 0.0)
        identity_attack = float(ex.get("identity_attack", 0.0) or 0.0)
        threat          = float(ex.get("threat",          0.0) or 0.0)
        insult          = float(ex.get("insult",          0.0) or 0.0)
        excl = min(1.0, identity_attack * 0.4 + threat * 0.3 + toxicity * 0.2 + insult * 0.1)
        lbl  = safe_label(1.0 - excl)
        if lbl is None:
            continue
        rows.append({"text": text[:512], "label": lbl})
        if len(rows) >= cap:
            break
    ds = Dataset.from_list(rows).shuffle(seed=42)
    print(f"  → civil_comments: {len(ds)} examples")
    return ds


def load_hatecheck(cap: int) -> Dataset:
    try:
        raw = load_dataset("Paul/hatecheck", split="test")
    except Exception as e:
        print(f"  → hatecheck failed ({e}), skipping.")
        return Dataset.from_list([])

    rows = []
    for ex in raw:
        text  = str(ex.get("test_case", "")).strip()
        label = str(ex.get("label_gold", "")).strip().lower()
        if not text or not label:
            continue
        if label == "hateful":
            lbl = 0.1
        elif label == "non-hateful":
            lbl = 0.9
        else:
            continue
        rows.append({"text": text[:512], "label": lbl})

    ds = Dataset.from_list(rows).shuffle(seed=42)
    return ds.select(range(min(cap, len(ds))))


def load_inclusivity_data() -> Dataset:
    ucb       = load_ucberkeley()
    cap       = len(ucb)
    davidson  = load_davidson(cap)
    jigsaw    = load_jigsaw(cap)
    sbic      = load_sbic(cap)
    civil     = load_civil_comments(cap)
    hatecheck = load_hatecheck(cap)

    combined = concatenate_datasets(
        [ucb, davidson, jigsaw, sbic, civil, hatecheck]
    ).shuffle(seed=42)

    combined = combined.filter(
        lambda x: x["label"] is not None and math.isfinite(x["label"])
    )

    low  = combined.filter(lambda x: x["label"] < 0.5)
    high = combined.filter(lambda x: x["label"] >= 0.5)
    print(f"  → Before balance — low: {len(low)}, high: {len(high)}")

    bal_cap  = min(len(low), len(high), 20_000)
    low      = low.shuffle(seed=42).select(range(bal_cap))
    high     = high.shuffle(seed=42).select(range(bal_cap))
    combined = concatenate_datasets([low, high]).shuffle(seed=42)
    print(f"  → After balance: {len(combined)} examples")
    return combined


if __name__ == "__main__":
    config = MetricConfig(
        metric_name    = "inclusivity",
        task_type      = "regression",
        dataset_loader = load_inclusivity_data,
        output_dir     = "saved_models",
        epochs         = 5,
        batch_size     = 16,
        lr             = 1e-5,
        fp16           = False,
        bf16           = False,
        max_length     = 256,
        model_name     = "microsoft/deberta-base",
    )
    run_training(config)
    

# Politeness

import io
import math
import zipfile
import csv
import numpy as np
import requests
from datasets import load_dataset, Dataset, concatenate_datasets
from base_trainer import MetricConfig, run_training

STANFORD_URL = "https://cs.cornell.edu/~cristian/Politeness_files/Stanford_politeness_corpus.zip"


def safe_label(value: float):
    if not math.isfinite(value):
        return None
    return max(0.0, min(1.0, value))


def noisy(base: float, std: float = 0.06) -> float:
    return float(np.clip(base + np.random.normal(0, std), 0.0, 1.0))


def load_frfede() -> Dataset:
    raw = load_dataset("frfede/politeness-corpus", split="train")
    LABEL_MAP = {0: 0.0, 1: 0.5, 2: 1.0}
    rows = []
    for ex in raw:
        text    = str(ex.get("text", "")).strip()
        lbl_int = ex.get("label", None)
        if not text or lbl_int is None:
            continue
        base = LABEL_MAP.get(int(lbl_int), None)
        if base is None:
            continue
        lbl = safe_label(noisy(base))
        if lbl is None:
            continue
        rows.append({"text": text[:512], "label": lbl})
    return Dataset.from_list(rows).shuffle(seed=42)


def load_stanford(cap: int) -> Dataset:
    try:
        r = requests.get(STANFORD_URL, timeout=60)
        r.raise_for_status()
    except Exception as e:
        print(f"  → Stanford download failed ({e}), skipping.")
        return Dataset.from_list([])

    rows = []
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        for name in z.namelist():
            if not name.endswith(".csv"):
                continue
            with z.open(name) as f:
                reader = csv.DictReader(io.TextIOWrapper(f, encoding="utf-8"))
                for record in reader:
                    text      = str(record.get("Request", record.get("Text", ""))).strip()
                    score_str = str(record.get("Score", record.get("Normalized Score", ""))).strip()
                    if not text or not score_str:
                        continue
                    try:
                        score = float(score_str)
                    except ValueError:
                        continue
                    lbl = safe_label((score + 3.0) / 6.0)
                    if lbl is None:
                        continue
                    rows.append({"text": text[:512], "label": lbl})

    ds = Dataset.from_list(rows).shuffle(seed=42)
    return ds.select(range(min(cap, len(ds))))


def load_davidson_politeness(cap: int) -> Dataset:
    raw  = load_dataset("tdavidson/hate_speech_offensive", split="train")
    rows = []
    for ex in raw:
        text = str(ex.get("tweet", "")).strip()
        if not text:
            continue
        hate_c    = int(ex.get("hate_speech_count",        0) or 0)
        offensive = int(ex.get("offensive_language_count", 0) or 0)
        neither   = int(ex.get("neither_count",            0) or 0)
        total     = hate_c + offensive + neither
        if total == 0:
            continue
        offensiveness = (hate_c * 1.0 + offensive * 0.5) / total
        lbl = safe_label(noisy(1.0 - offensiveness, std=0.05))
        if lbl is None:
            continue
        rows.append({"text": text[:512], "label": lbl})
    ds = Dataset.from_list(rows).shuffle(seed=42)
    return ds.select(range(min(cap, len(ds))))


def load_politeness_data() -> Dataset:
    frfede   = load_frfede()
    cap      = len(frfede)
    stanford = load_stanford(cap)
    davidson = load_davidson_politeness(cap)
    sources  = [s for s in [frfede, stanford, davidson] if len(s) > 0]
    combined = concatenate_datasets(sources).shuffle(seed=42)
    return combined.filter(lambda x: x["label"] is not None and math.isfinite(x["label"]))


if __name__ == "__main__":
    config = MetricConfig(
        metric_name    = "politeness",
        task_type      = "regression",
        dataset_loader = load_politeness_data,
        output_dir     = "saved_models",
        epochs         = 4,
        batch_size     = 16,
        lr             = 2e-5,
        bf16           = False,
        fp16           = False,
        max_length     = 128,
        model_name     = "microsoft/deberta-base",
    )
    run_training(config)
    
# Safety

from datasets import load_dataset, Dataset, concatenate_datasets
from base_trainer import MetricConfig, run_training


def load_safety_data():
    ds  = load_dataset("google/civil_comments")
    raw = ds["train"].shuffle(seed=42).select(range(200_000))

    def preprocess(example):
        text     = example.get("text", "")
        toxicity = float(example.get("toxicity", 0.0) or 0.0)
        return {"text": str(text), "label": 1.0 - min(toxicity, 1.0)}

    processed = raw.map(preprocess, remove_columns=raw.column_names)

    safe_ex  = processed.filter(lambda x: x["label"] > 0.5)
    toxic_ex = processed.filter(lambda x: x["label"] <= 0.5)
    print(f"  Safe: {len(safe_ex)}, Toxic: {len(toxic_ex)}")

    bal_cap  = min(len(safe_ex), len(toxic_ex), 15_000)
    safe_ex  = safe_ex.shuffle(seed=42).select(range(bal_cap))
    toxic_ex = toxic_ex.shuffle(seed=42).select(range(bal_cap))
    processed = concatenate_datasets([safe_ex, toxic_ex]).shuffle(seed=42)
    processed = processed.map(lambda x: {"label": 0.05 if x["label"] < 0.5 else 0.95})
    print(f"  → Final dataset: {len(processed)} examples (balanced)")
    return processed


if __name__ == "__main__":
    config = MetricConfig(
        metric_name    = "safety",
        dataset_loader = load_safety_data,
        output_dir     = "saved_models",
        epochs         = 4,
        batch_size     = 16,
        lr             = 1e-5,
        bf16           = False,
        fp16           = False,
        model_name     = "microsoft/deberta-base",
    )
    run_training(config)