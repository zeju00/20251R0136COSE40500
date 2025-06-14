import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from scipy.stats import spearmanr
import shap
from tqdm import tqdm

# =============================
# 1. Dataset: load result.json
# =============================
class XaiTokenizedDataset(Dataset):
    def __init__(self, result_json_path: str, tokenizer: BertTokenizer, max_len: int = 128):
        with open(result_json_path, 'r') as f:
            examples = json.load(f)
        self.texts      = [ex['text']       for ex in examples]
        self.labels     = [ex['true_label'] for ex in examples]
        self.bert_shaps = [ex['token_shap'] for ex in examples]
        self.tokenizer  = tokenizer
        self.max_len    = max_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors='pt'
        )
        input_ids      = enc['input_ids'].squeeze(0)        # [L]
        attention_mask = enc['attention_mask'].squeeze(0)   # [L]
        label          = torch.tensor(self.labels[idx], dtype=torch.long)

        # align BERT SHAP vectors to tokens (pad missing tokens with zeros)
        tokens   = self.tokenizer.convert_ids_to_tokens(input_ids)
        shap_vecs = [
            self.bert_shaps[idx].get(tok, [0.0]*4)
            for tok in tokens
        ]
        bert_shap = torch.tensor(shap_vecs, dtype=torch.float)  # [L, 4]

        return input_ids, attention_mask, label, bert_shap

# ===================================
# 2. MLP classifier with learnable Embedding
# ===================================
class MLPWithEmbedding(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.fc1       = nn.Linear(embed_dim, hidden_dim)
        self.fc2       = nn.Linear(hidden_dim, output_dim)
        self.dropout   = nn.Dropout(0.1)

    def forward(self, input_ids: torch.LongTensor, attention_mask: torch.LongTensor):
        # [B, L] → [B, L, E]
        emb = self.embedding(input_ids)
        # masked mean pooling → [B, E]
        mask   = attention_mask.unsqueeze(-1)                   # [B, L, 1]
        summed = torch.sum(emb * mask, dim=1)                   # [B, E]
        counts = mask.sum(dim=1).clamp(min=1)                   # [B, 1]
        pooled = summed / counts                                # [B, E]

        x = self.dropout(F.relu(self.fc1(pooled)))              # [B, H]
        logits = self.fc2(x)                                    # [B, C]
        return logits

# ===================================
# 3. XAI loss: Cosine + Spearman
# ===================================
def xai_loss(mlp_shap: torch.Tensor, bert_shap: torch.Tensor) -> torch.Tensor:
    # mlp_shap, bert_shap: [B, L, C]
    B, L, C = mlp_shap.shape
    mlp_vec  = mlp_shap.view(B, -1)    # [B, L*C]
    bert_vec = bert_shap.view(B, -1)   # [B, L*C]

    cos_sim  = F.cosine_similarity(mlp_vec, bert_vec, dim=1)  # [B]
    loss_cos = 1.0 - cos_sim

    rho_list = []
    for i in range(B):
        r, _ = spearmanr(
            mlp_vec[i].cpu().numpy(),
            bert_vec[i].cpu().numpy()
        )
        rho_list.append(r if not torch.isnan(torch.tensor(r)) else 0.0)
    rho      = torch.tensor(rho_list, device=mlp_shap.device)
    loss_spr = 1.0 - rho

    return (loss_cos + loss_spr).mean()

# ===================================
# 4. Main training routine
# ===================================
def main():
    # Hyperparameters
    RESULT_JSON = 'result.json'
    BATCH_SIZE  = 32
    MAX_LEN     = 128
    EMBED_DIM   = 128
    HIDDEN_DIM  = 256
    N_CLASSES   = 4
    LR          = 1e-3
    EPOCHS      = 5
    ALPHA       = 0.7  # weight for L_ce vs L_xai

    # 4.1 Tokenizer & Dataset
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    dataset   = XaiTokenizedDataset(RESULT_JSON, tokenizer, max_len=MAX_LEN)
    loader    = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    # 4.2 Model, optimizer, loss
    model     = MLPWithEmbedding(tokenizer.vocab_size, EMBED_DIM, HIDDEN_DIM, N_CLASSES).cuda()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    ce_loss   = nn.CrossEntropyLoss()

    # 4.3 SHAP explainer (MLP only)
    explainer = shap.Explainer(model, masker=tokenizer, algorithm='gradient')

    # 4.4 Training loop
    for epoch in range(1, EPOCHS+1):
        model.train()
        running_loss = 0.0

        for input_ids, attention_mask, labels, bert_shap in tqdm(loader, desc=f"Epoch {epoch}/{EPOCHS}"):
            input_ids, attention_mask = input_ids.cuda(), attention_mask.cuda()
            labels, bert_shap          = labels.cuda(), bert_shap.cuda()

            optimizer.zero_grad()
            # (a) Cross-entropy
            logits = model(input_ids, attention_mask)
            L_ce   = ce_loss(logits, labels)

            # (b) SHAP values for MLP
            shap_expl = explainer(list(input_ids))  # list of shap.Explanation
            mlp_shap  = torch.stack([torch.tensor(x.values) for x in shap_expl], dim=0).cuda()

            # (c) XAI loss
            L_xai = xai_loss(mlp_shap, bert_shap)

            # (d) Combined loss
            loss = ALPHA * L_ce + (1 - ALPHA) * L_xai
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(loader)
        print(f"[Epoch {epoch}] Avg Loss: {avg_loss:.4f}")

    # Optionally: save model
    torch.save(model.state_dict(), 'mlp_xai_model.pt')

if __name__ == '__main__':
    main()