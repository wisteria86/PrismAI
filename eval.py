import torch
import evaluate
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

print("Loading Tokenizer and Model...")
REPO_ID = "Wisteria86/coedit" 
VERSION_TAG = "v2.0.1"

print(f"Loading {REPO_ID} (version: {VERSION_TAG})...")

tokenizer = AutoTokenizer.from_pretrained(
    REPO_ID, 
    revision=VERSION_TAG
)

model = AutoModelForSeq2SeqLM.from_pretrained(
    REPO_ID, 
    revision=VERSION_TAG
).to("cuda")


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = model.to(DEVICE)
model.eval()

print("Model ready for inference!")

rouge = evaluate.load('rouge')
bleu = evaluate.load('bleu')

print("Model ready for inference!")

def generate_text(prompt):
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            temperature=0.7,
            do_sample=True,
            repetition_penalty=1.1
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

eval_data = [
    {
        "input": "Make this text easier to understand: The utilization of complex methodological frameworks is often obfuscatory.",
        "reference": "Using complex methods is often confusing."
    },
    {
        "input": "Make this Complex: Yo, send me the files ASAP so I can look 'em over.",
        "reference": "Please send me the files as soon as possible so I may review them."
    },
    {
        "input": "Make this polite: You are completely wrong and your idea sucks.",
        "reference": "I disagree with your approach and think we should explore other options."
    },
    {
        "input": "Make it Polite: Shut up and just do your stupid job.",
        "reference": "Please focus on completing your assigned tasks."
    },
    {
        "input": "Make this Formal: Every businessman must bring his laptop to the conference.",
        "reference": "Every business professional must bring their laptop to the conference."
    },
    {
        "input": "Make this Polite: The absolutely terrible CEO ruined the company with his idiotic decisions.",
        "reference": "The CEO's decisions negatively impacted the company's performance."
    },
    {
        "input": "Make This Polite: I am absolutely furious and heartbroken about this disastrous failure!",
        "reference": "I am highly disappointed about this negative outcome."
    },
    {
        "input": "Simplify This: The nocturnal avian predator descends swiftly upon its unsuspecting prey.",
        "reference": "The owl flies down quickly to catch its food."
    },
    {
        "input": "Fix grammatical errors in this sentence: I am goodest at the programming.",
        "reference": "I am good at programming."
    }
]

print("\nRunning Evaluation...")
predictions = []
references = []

for item in eval_data:
    pred = generate_text(item["input"])
    predictions.append(pred)
    references.append([item["reference"]])
    
    print(f"[Input]: {item['input']}")
    print(f"[Target]: {item['reference']}")
    print(f"[Pred]:   {pred}\n")

print("Calculating Metrics...")
rouge_results = rouge.compute(predictions=predictions, references=references)
bleu_results = bleu.compute(predictions=predictions, references=references)

print("="*40)
print(f"ROUGE-1: {rouge_results['rouge1']:.4f}")
print(f"ROUGE-L: {rouge_results['rougeL']:.4f}")
print(f"BLEU:    {bleu_results['bleu']:.4f}")
print("="*40)