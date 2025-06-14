from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset, DatasetDict
import evaluate
import torch
import torch.nn.functional as F
from sklearn.datasets import fetch_20newsgroups

# 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

# TinyBERT 모델
model_name = "huawei-noah/TinyBERT_General_4L_312D"

# 모델 & 토크나이저 로딩
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=20)  # ✅ 20 classes

# ✅ 20 Newsgroups 데이터셋 로딩
# 1. 데이터 로드
train_data = fetch_20newsgroups(subset="train", remove=("headers", "footers", "quotes"))
test_data = fetch_20newsgroups(subset="test", remove=("headers", "footers", "quotes"))

# 2. Hugging Face Dataset 형태로 변환
train_dataset = Dataset.from_dict({"text": train_data.data, "label": train_data.target})
test_dataset = Dataset.from_dict({"text": test_data.data, "label": test_data.target})
dataset = DatasetDict({"train": train_dataset, "test": test_dataset})

label_names = train_data.target_names
print(label_names)

# 텍스트 전처리
def preprocess(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=128)

encoded = dataset.map(preprocess, batched=True)

# 학습/평가용 샘플 축소 (전체 쓰고 싶으면 select 제거)
#train_dataset = encoded["train"].select(range(4000))
#eval_dataset = encoded["test"].select(range(1000))
train_dataset = encoded["train"]
eval_dataset = encoded["test"]
# 정확도 측정
accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = torch.argmax(torch.tensor(logits), dim=1)
    return accuracy.compute(predictions=preds, references=labels)

# 학습 설정
training_args = TrainingArguments(
    output_dir="./tinybert-newsgroups",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=32,
    num_train_epochs=3,
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=100,
    report_to='none',
    no_cuda=True,
)

# Trainer 설정
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()

# 예측 테스트
text = "The government plans to increase taxes on imported cars."

inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
model.to(device)
inputs = {k: v.to(device) for k, v in inputs.items()}

model.eval()
with torch.no_grad():
    outputs = model(**inputs)
    probs = F.softmax(outputs.logits, dim=1).cpu().numpy()
    pred = torch.argmax(outputs.logits, dim=1).item()

# 라벨 정보 추출 (데이터셋에서 제공)
label_names = train_data.target_names

print("\n🧠 예측 클래스:", label_names[pred])
print("\n📊 Soft labels (확률):")
for i, p in enumerate(probs[0]):
    print(f"{label_names[i]}: {p:.4f}")
