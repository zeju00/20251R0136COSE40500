from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import load_dataset
import evaluate
import torch
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# TinyBERT 모델 (사전학습)
model_name = "huawei-noah/TinyBERT_General_4L_312D"

# 토크나이저 & 모델 불러오기
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=4)

# AG News 데이터셋 불러오기
dataset = load_dataset("ag_news")

def preprocess(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=128)

encoded = dataset.map(preprocess, batched=True)

# 빠른 실행을 위해 일부 샘플만 사용 (full 학습 원하면 이 부분 생략)
#train_dataset = encoded["train"].select(range(5000))
train_dataset = encoded["train"]

eval_dataset = encoded["test"].select(range(1000))

# 정확도 측정 함수
accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = torch.argmax(torch.tensor(logits), dim=1)
    return accuracy.compute(predictions=preds, references=labels)

# 학습 설정
training_args = TrainingArguments(
    output_dir="./tinybert-agnews",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=32,
    num_train_epochs=2,
    # evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=100,
    # load_best_model_at_end=True,
    report_to='none',
    no_cuda=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()

# 테스트용 뉴스 기사 문장
text = "NASA launched a new satellite to study the sun's activity."

inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)

model.to(device)

inputs = {k: v.to(device) for k, v in inputs.items()}  # 입력도 GPU로 이동

model.eval()
with torch.no_grad():
    outputs = model(**inputs)
    probs = F.softmax(outputs.logits, dim=1).cpu().numpy()
    pred = torch.argmax(outputs.logits, dim=1).item()

labels = ["World", "Sports", "Business", "Sci/Tech"]
# 출력
print("예측 클래스:", labels[pred])
print("\nSoft labels (확률):")
for i, p in enumerate(probs[0]):
    print(f"{labels[i]}: {p.item():.4f}")
