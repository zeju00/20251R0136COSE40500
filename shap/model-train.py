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
import math
import numpy as np

# ---------- Dataset & Collate ----------
class XaiTokenizedDataset(Dataset):
    def __init__(self, result_json_path, tokenizer, max_len=128):
        with open(result_json_path, 'r') as f:
            raw_examples = json.load(f)
        examples = [ex for ex in raw_examples if isinstance(ex.get("text"), str)]
        self.texts = [ex['text'] for ex in examples]
        self.labels = [ex['true_label'] for ex in examples]
        self.bert_shaps = [ex['token_shap'] for ex in examples]
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        enc = self.tokenizer(self.texts[idx], padding='max_length', truncation=True, max_length=self.max_len, return_tensors='pt')
        input_ids = enc['input_ids'].squeeze(0)
        attention_mask = enc['attention_mask'].squeeze(0)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        shap_dict = self.bert_shaps[idx]
        tokens = self.tokenizer.convert_ids_to_tokens(enc['input_ids'].squeeze(0))
        shap_vecs = [shap_dict.get(tok, [0.0] * 4) for tok in tokens]
        bert_shap = torch.tensor(shap_vecs, dtype=torch.float)
        return input_ids, attention_mask, label, bert_shap, self.texts[idx]

def custom_collate(batch):
    input_ids = torch.stack([item[0] for item in batch])
    attention_mask = torch.stack([item[1] for item in batch])
    labels = torch.stack([item[2] for item in batch])
    bert_shap = torch.stack([item[3] for item in batch])
    texts = [item[4] for item in batch]
    return input_ids, attention_mask, labels, bert_shap, texts

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

# ---------- Wrapper ----------
class ClassifierWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, x):
        attention_mask = torch.ones_like(x[:, :, 0])  # 임시 attention mask
        return self.model.forward_class(x, attention_mask)

# ---------- Soft Spearman Functions ----------
def soft_rank(x, regularization_strength=1.0):
    x = x.unsqueeze(-1)
    pairwise_diff = x - x.transpose(-2, -1)
    soft_sign = torch.sigmoid(-regularization_strength * pairwise_diff)
    return soft_sign.sum(dim=-1)

def soft_spearmanr(x, y, eps=1e-8):
    sr_x = soft_rank(x)
    sr_y = soft_rank(y)
    sr_x = (sr_x - sr_x.mean(dim=-1, keepdim=True)) / (sr_x.std(dim=-1, keepdim=True) + eps)
    sr_y = (sr_y - sr_y.mean(dim=-1, keepdim=True)) / (sr_y.std(dim=-1, keepdim=True) + eps)
    return (sr_x * sr_y).mean(dim=-1)

# ---------- XAI Loss ----------
def xai_loss(mlp_shap, bert_shap, attention_mask):
    B, L, C = mlp_shap.shape
    #print(B, L, C)
    #print(mlp_shap.shape)
    #print(bert_shap.shape)
    valid_indices = attention_mask.nonzero(as_tuple=True)[0]
    mlp_shap = mlp_shap[:, valid_indices]
    bert_shap = bert_shap[:, valid_indices]
    
    cos_sim = []
    for t in range(min(mlp_shap.size(1), bert_shap.size(1))):
        mlp_tok = mlp_shap[:, t]
        bert_tok = bert_shap[:, t]
        cos = F.cosine_similarity(mlp_tok, bert_tok) # 계속하기
        cos_sim.append(cos.cpu().tolist())


    #mlp_vec = mlp_shap.sum(dim=1)
    #bert_vec = bert_shap.sum(dim=1)
    #print(np.mean(cos_sim))
    #exit(0)
    loss_cos = 1.0 - np.mean(cos_sim)
    #mlp_shap_ = mlp_shap.permute(2, 0, 1)
    #print(bert_shap.shape)
    #print(mlp_shap.shape)
    '''
    rho_list = []
    mlp_flat = []
    for t in range(mlp_tok.size(0)):
        vec = mlp_tok[t].detach().cpu().numpy()
        mlp_flat.extend(vec.tolist())
    
    bert_flat = bert_tok.view(-1).cpu().numpy()  # [L * C]
    
    rho, _ = spearmanr(mlp_flat, bert_flat)
    rho_list.append(rho if not np.isnan(rho) else 0.0)
    
    rho_list = []
    for i in range(B):
        #print(i)
        sample_rhos = []
        for c in range(C):
            #print(i, c)
            x = mlp_shap_[i, :, c].detach().cpu().numpy()
            y = bert_shap[i, :, c].detach().cpu().numpy()

            if len(x) != len(y) or len(x) < 2:
                continue  # skip invalid comparisons
            r, _ = spearmanr(x, y)
            if math.isnan(r):
                r = 0.0
            sample_rhos.append(r)
        rho_list.append(sum(sample_rhos) / L)
    '''
    flat_mlp = mlp_shap.view(mlp_shap.size(0), -1)  # [B, L*C]
    flat_bert = bert_shap.view(bert_shap.size(0), -1)  # [B, L*C]
    rho_tensor = soft_spearmanr(flat_mlp, flat_bert)
    loss_spr = 1.0 - rho_tensor.mean()
    #loss_spr = 1.0 - np.mean(rho_list)
    return loss_cos, loss_spr

# ---------- Training Loop ----------
def main():
    RESULT_JSON = 'result.json'
    BATCH_SIZE = 1
    MAX_LEN = 128
    EMBED_DIM = 128
    HIDDEN_DIM = 256
    N_CLASSES = 4
    LR = 1e-3
    EPOCHS = 7
    ALPHA = 0.5

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    dataset = XaiTokenizedDataset(RESULT_JSON, tokenizer, max_len=MAX_LEN)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate)

    device = torch.device('cuda:2')
    print(device)
    model = DualPathMLP(tokenizer.vocab_size, EMBED_DIM, HIDDEN_DIM, N_CLASSES).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    ce_loss = nn.CrossEntropyLoss()

    background_input_ids, _, _, _, _ = next(iter(loader))
    background_embeds = model.embed_only(background_input_ids[:MAX_LEN].to(device))
    explainer = shap.DeepExplainer(ClassifierWrapper(model), background_embeds)
    
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_ce = total_cos = total_spr = 0.0

        for input_ids, attention_mask, labels, bert_shap, _ in tqdm(loader, desc=f"Epoch {epoch}"):
            input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
            labels, bert_shap = labels.to(device), bert_shap.to(device)
            #print(bert_shap.shape)
            #exit(0)

            optimizer.zero_grad()

            embedded = model.embed_only(input_ids)

            # Classification path
            logits = model.forward_class(embedded, attention_mask)
            loss_cls = ce_loss(logits, labels)

            # XAI path
            shap_values = explainer.shap_values(embedded, check_additivity=False)
            shap_tensor = torch.stack([torch.tensor(sv, dtype=torch.float32) for sv in shap_values], dim=-1)
            shap_tensor = shap_tensor.sum(2).to(device)  # [B, L, C]
            xai_output = model.forward_xai(embedded)
            #print(xai_output.shape)
            #exit(0)
            loss_cos, loss_spr = xai_loss(xai_output, bert_shap, attention_mask)

            # Total loss
            loss = ALPHA * loss_cls + (1 - ALPHA) * (loss_cos + loss_spr)
            loss.backward()
            optimizer.step()

            total_ce += loss_cls.item()
            total_cos += loss_cos.item()
            total_spr += loss_spr.item()

        print(f"[Epoch {epoch}] CE: {ALPHA * total_ce/len(loader):.4f}, COS: {(1 - ALPHA) * total_cos/len(loader):.4f}, SPR: {(1 - ALPHA) * total_spr/len(loader):.4f}")

    torch.save(model.state_dict(), f"dualpath_model_{epoch}epochs.pt")

if __name__ == "__main__":
    main()
