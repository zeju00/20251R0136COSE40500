# tinybert_shap_example.py

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import shap
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

# 1. 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. 모델과 토크나이저 불러오기
model_path = "./tinybert-agnews/checkpoint-1250"  # 원하는 checkpoint 디렉토리
base_tokenizer_path = "huawei-noah/TinyBERT_General_4L_312D"

tokenizer = AutoTokenizer.from_pretrained(base_tokenizer_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.to(device)
model.eval()

# 3. forward 함수 정의
def forward_func(texts):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.logits

# 4. 분석할 문장
texts = [
    "NASA successfully launched a new satellite.",
    "The stock market crashed after inflation data was released."
]

# 5. SHAP 분석
explainer = shap.Explainer(forward_func, tokenizer)
shap_values = explainer(texts)

# 6. 시각화
for i, text in enumerate(texts):
    print(f"\n📝 문장 {i+1}: {text}")
    shap.plots.text(shap_values[i], display=True)

# 7. (선택) 개별 클래스별 확률 출력
print("\n🔢 Softmax 확률:")
for i, text in enumerate(texts):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits[0]
        probs = F.softmax(logits, dim=0).cpu().numpy()
        print(f"문장 {i+1}: {probs.round(4)}")

