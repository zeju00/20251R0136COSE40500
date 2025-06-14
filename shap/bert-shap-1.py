import shap
import torch
import numpy as np
import json
from transformers import BertTokenizer, BertForSequenceClassification
from datasets import load_dataset
from tqdm import tqdm
import torch.nn.functional as F


# 1. Load tokenizer and model
model_path = "../models/ft_BERT_agnews_full_dataset"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)
model.eval().cuda()

# 2. Load and sample AG News
dataset = load_dataset("ag_news", split="train")  # 전체 로드, 순서 유지
samples_per_class = 2500
class_counts = {0: 0, 1: 0, 2: 0, 3: 0}
selected_samples = []

for ex in dataset:
    label = ex["label"]
    if class_counts[label] < samples_per_class:
        selected_samples.append(ex)
        class_counts[label] += 1
    if all(c == samples_per_class for c in class_counts.values()):
        break

texts = [ex["text"] for ex in selected_samples]
true_labels = [ex["label"] for ex in selected_samples]
# 4. SHAP용 forward 함수 정의

def forward_func(texts):
    # SHAP이 넘기는 pre-tokenized 단어 리스트들 (List[List[str]])
    texts = list(texts)
    # tokenizer는 batch 처리 가능하므로 그대로 넘김
    inputs = tokenizer(
        texts,
        # is_split_into_words=True,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128
    )
    inputs = {k: v.cuda() for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    #print(f"model output shape: {outputs.logits.shape}")
    #exit(0)
    return outputs.logits  # ✅ [batch_size, num_labels] shape 유지!

# 5. SHAP Explainer 생성 및 해석

explainer = shap.Explainer(forward_func, masker=tokenizer, algorithm="partition")
shap_values = explainer(texts)

results = []
for i, text in enumerate(texts):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    inputs = {k: v.cuda() for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits[0]
        probs = F.softmax(logits, dim=0)
        pred_label = torch.argmax(probs).item()
    
    inputs_full = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=128)
    input_ids = inputs_full["input_ids"][0]
    tokens_all = tokenizer.convert_ids_to_tokens(input_ids)

    # get shap values
    token_shap_values_full = shap_values.values[0]

    # restore tokens and shap values
    token_shap = {
        tok: list(val) for tok, val in zip(tokens_all, token_shap_values_full)
    }
    
    # Create result dictionary
    result = {
    "text": text,
    "true_label": int(true_label),
    "predicted_label": int(pred_label),
    "token_shap": token_shap,
    "probabilities": probs.tolist()
    }
    results.append(result)

# Save results to JSON file
with open("shap_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"Results saved to shap_results.json")