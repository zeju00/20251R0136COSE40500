from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import load_dataset
import evaluate
import torch
import torch.nn.functional as F

# ✅ 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

# ✅ TinyBERT 사전학습 모델
model_name = "huawei-noah/TinyBERT_General_4L_312D"

# ✅ 토크나이저 & 분류용 모델 (14개 클래스)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=14)

# ✅ DBPedia 데이터셋 로딩
dataset = load_dataset("dbpedia_14")

# ✅ 전처리 함수
def preprocess(example):
    return tokenizer(example["content"], truncation=True, padding="max_length", max_length=128)

encoded = dataset.map(preprocess, batched=True)

# ✅ 학습/검증 세트 샘플링 (전체 학습하고 싶으면 select 제거)
#train_dataset = encoded["train"].select(range(300000))
#eval_dataset = encoded["test"].select(range(30000))
train_dataset = encoded["train"]
eval_dataset = encoded["test"]
# ✅ 평가 메트릭 로딩
accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = torch.argmax(torch.tensor(logits), dim=1)
    return accuracy.compute(predictions=preds, references=labels)

# ✅ 학습 파라미터
training_args = TrainingArguments(
    output_dir="./tinybert-dbpedia",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=32,
    num_train_epochs=2,
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=100,
    report_to='none',
    no_cuda=True,  # GPU 사용 시 False로 변경
)

# ✅ 트레이너 정의
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
)

# ✅ 학습 시작
trainer.train()
