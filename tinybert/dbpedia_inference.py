import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# ✅ 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("✅ device:", device)

# ✅ 학습된 TinyBERT DBPedia 모델 경로
checkpoint_path = "./tinybert-dbpedia/checkpoint-70000"  # 저장된 체크포인트로 수정

# ✅ 토크나이저 & 모델 로드
tokenizer = AutoTokenizer.from_pretrained("huawei-noah/TinyBERT_General_4L_312D")
model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path)
model.to(device)
model.eval()

# ✅ DBPedia test 데이터셋 로딩
dataset = load_dataset("dbpedia_14")
test_texts = dataset["test"]["content"]
test_labels = dataset["test"]["label"]
label_names = dataset["test"].features["label"].names

# ✅ 배치 단위 추론
preds = []
wrong_samples = []  # 오답 저장용
batch_size = 32

for i in tqdm(range(0, len(test_texts), batch_size)):
    batch_texts = test_texts[i:i+batch_size]
    batch_labels = test_labels[i:i+batch_size]

    encodings = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
    encodings = {k: v.to(device) for k, v in encodings.items()}

    with torch.no_grad():
        logits = model(**encodings).logits
        batch_preds = torch.argmax(logits, dim=1).cpu().tolist()
        preds.extend(batch_preds)

        # 오답 확인
        for j, (pred, label) in enumerate(zip(batch_preds, batch_labels)):
            if pred != label:
                wrong_samples.append({
                    "text": batch_texts[j],
                    "true_label": label_names[label],
                    "pred_label": label_names[pred]
                })

# ✅ 정확도 계산
acc = accuracy_score(test_labels, preds)
print(f"\n✅ TinyBERT DBPedia 평균 정확도: {acc:.4f}")
print(f"❌ 오답 수: {len(wrong_samples)}")

# ✅ 오답 몇 개 출력 예시
print("\n📌 예측에 실패한 샘플 예시:")
for i, sample in enumerate(wrong_samples[:5]):
    print(f"\n[{i+1}]")
    print(f"문장: {sample['text']}")
    print(f"  ▶ 정답: {sample['true_label']}")
    print(f"  ▶ 예측: {sample['pred_label']}")
