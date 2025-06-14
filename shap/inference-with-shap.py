from datasets import load_dataset
from transformers import BertTokenizer
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.stats import spearmanr
import json

class AgNewsTestDataset(Dataset):
    def __init__(self, bert_shap_path, split="test", max_len=128):
        dataset = load_dataset("ag_news", split=split)
        self.texts = dataset["text"]
        self.labels = dataset["label"]
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.max_len = max_len
        with open(bert_shap_path, "r") as f:
            self.bert_shaps = json.load(f)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        enc = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
        token_shap_dict = self.bert_shaps[idx].get("token_shap", {})
        bert_shap_vec = [token_shap_dict.get(tok, [0.0]*4) for tok in tokens]  # [L, C]
        bert_shap_tensor = torch.tensor(bert_shap_vec, dtype=torch.float)     # [L, C]
        return input_ids, attention_mask, label, bert_shap_tensor

# ---------- Model ----------
class DualPathMLP(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # Classification path
        self.class_fc1 = nn.Linear(embed_dim, hidden_dim)
        self.class_fc2 = nn.Linear(hidden_dim, output_dim)

        # XAI path
        self.xai_fc1 = nn.Linear(embed_dim, hidden_dim)
        self.xai_fc2 = nn.Linear(hidden_dim, output_dim)

        self.dropout = nn.Dropout(0.1)

    def embed_only(self, input_ids):
        return self.embedding(input_ids)

    def forward_class(self, embedded, attention_mask):
        mask = attention_mask.unsqueeze(-1)
        pooled = (embedded * mask).sum(1) / mask.sum(1).clamp(min=1)
        x = self.dropout(F.relu(self.class_fc1(pooled)))
        return self.class_fc2(x)

    def forward_xai(self, embedded):
        B, L, D = embedded.shape
        x = F.relu(self.xai_fc1(embedded))
        return self.xai_fc2(x)  # [B, L, C]

def evaluate(model, dataloader, device, n_classes):
    model.eval()
    correct = 0
    total = 0
    token_cosine_scores = []
    spearman_scores = []

    with torch.no_grad():
        for input_ids, attention_mask, labels, bert_shap in tqdm(dataloader, desc="Evaluating"):
            input_ids = input_ids.to(device)               # [B, L]
            attention_mask = attention_mask.to(device)     # [B, L]
            labels = labels.to(device)                     # [B]
            bert_shap = bert_shap.to(device)               # [B, L, C]

            embedded = model.embed_only(input_ids)
            logits = model.forward_class(embedded, attention_mask)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            # MLP SHAP-like: use XAI path
            mlp_shap = model.forward_xai(embedded).detach() # [B, L, C]
            #print('mlp_shap.shape: ', mlp_shap.shape)
            #print('mlp_shap: ', mlp_shap)
            #exit(0)

            for b in range(embedded.size(0)):
                mlp_tok = mlp_shap[b]                      # [L, C]
                bert_tok = bert_shap[b]                    # [L, C]
                #print('mlp_tok: ', mlp_tok)
                #print('bert_tok: ', bert_tok)
                #exit(0)
                mask = attention_mask[b]     # [L]

                valid_indices = mask.nonzero(as_tuple=True)[0]  # padding 제외한 인덱스
                mlp_tok = mlp_tok[valid_indices]    # [valid_L, C]
                bert_tok = bert_tok[valid_indices]  # [valid_L, C]
                #print('mlp_tok: ', mlp_tok)
                #print('bert_tok: ', bert_tok)
                #exit(0)

                # Token-level cosine similarity per token
                for t in range(min(mlp_tok.size(0), bert_tok.size(0))):
                    mlp_token_vec = mlp_shap[b, t]
                    bert_token_vec = bert_tok[t]                               # [C]
                    #print('mlp_token_vec: ', mlp_token_vec.unsqueeze(0))
                    #print('bert_token_vec: ', bert_token_vec.unsqueeze(0))
                    cos_per_dim = F.cosine_similarity(
                        mlp_token_vec.unsqueeze(0),  # [C, 1]
                        bert_token_vec.unsqueeze(0)  # [C, 1]
                    )  # shape: [C]
                    #print('cos_per_dim: ', cos_per_dim)
                    #exit(0)
                    token_cosine_scores.append(cos_per_dim.cpu().tolist())

                # Sentence-level Spearman correlation
                mlp_flat = []
                for t in range(mlp_tok.size(0)):
                    vec = mlp_shap[b, t].cpu().numpy()
                    mlp_flat.extend(vec.tolist())
                bert_flat = bert_tok.view(-1).cpu().numpy()

                rho, _ = spearmanr(mlp_flat, bert_flat)
                spearman_scores.append(rho if not np.isnan(rho) else 0.0)
    #print('token_cosine_scores: ', token_cosine_scores[:10])
    #print('spearman_scores: ', spearman_scores[:10])
    
    acc = correct / total
    cos_avg = np.median(token_cosine_scores)
    spr_avg = np.median(spearman_scores)

    return acc, cos_avg, spr_avg

def run_test_evaluation():
    BATCH_SIZE = 64
    MAX_LEN = 128
    N_CLASSES = 4
    EMBED_DIM = 128
    HIDDEN_DIM = 256

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = DualPathMLP(tokenizer.vocab_size, EMBED_DIM, HIDDEN_DIM, N_CLASSES)
    model.load_state_dict(torch.load("dualpath_model_7epochs.pt"))
    model.to(device)

    test_dataset = AgNewsTestDataset(bert_shap_path="result_test.json", max_len=MAX_LEN)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    acc, cos_sim, spr_corr = evaluate(model, test_loader, device, n_classes=N_CLASSES)
    print(f"✅ Test Accuracy: {acc * 100:.2f}%")
    print(f"🔍 Avg Cosine Similarity: {cos_sim:.4f}")
    print(f"🔍 Avg Spearman Correlation: {spr_corr:.4f}")

if __name__ == "__main__":
    run_test_evaluation()