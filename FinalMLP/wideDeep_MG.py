import pandas as pd
import numpy as np
import os, random
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

CFG = {
    'BATCH_SIZE': 4096,
    'EPOCHS': 5,
    'LEARNING_RATE': 1e-3,
    'SEED': 42
}

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

os.environ["CUDA_VISIBLE_DEVICES"] = "7" 

seed_everything(CFG['SEED'])
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("VISIBLE_DEVICES =", os.environ["CUDA_VISIBLE_DEVICES"])
print("Current device index:", torch.cuda.current_device())
print("Physical device count:", torch.cuda.device_count())
if device.type == "cuda":
    print("CUDA device name:", torch.cuda.get_device_name(0))

print("데이터 로드 시작")
train = pd.read_parquet("../data/train.parquet", engine="pyarrow")
test = pd.read_parquet("../data/test.parquet", engine="pyarrow")
print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")
print("데이터 로드 완료")

target_col = "clicked"
seq_col = "seq"
FEATURE_EXCLUDE = {target_col, seq_col, "ID"}
feature_cols = [c for c in train.columns if c not in FEATURE_EXCLUDE]

cat_cols = ["gender", "age_group", "inventory_id", "l_feat_14"]
num_cols = [c for c in feature_cols if c not in cat_cols]
print(f"Num features: {len(num_cols)} | Cat features: {len(cat_cols)}")

def encode_categoricals(train_df, test_df, cat_cols):
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        all_values = pd.concat([train_df[col], test_df[col]], axis=0).astype(str).fillna("UNK")
        le.fit(all_values)
        train_df[col] = le.transform(train_df[col].astype(str).fillna("UNK"))
        test_df[col]  = le.transform(test_df[col].astype(str).fillna("UNK"))
        encoders[col] = le
        print(f"{col} unique categories: {len(le.classes_)}")
    return train_df, test_df, encoders

train, test, cat_encoders = encode_categoricals(train, test, cat_cols)

class ClickDataset(Dataset):
    def __init__(self, df, num_cols, cat_cols, seq_col, target_col=None, has_target=True):
        self.df = df.reset_index(drop=True)
        self.num_cols = num_cols
        self.cat_cols = cat_cols
        self.seq_col = seq_col
        self.target_col = target_col
        self.has_target = has_target
        self.num_X = self.df[self.num_cols].astype(float).fillna(0).values
        self.cat_X = self.df[self.cat_cols].astype(int).values
        self.seq_strings = self.df[self.seq_col].astype(str).values
        if self.has_target:
            self.y = self.df[self.target_col].astype(np.float32).values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        num_x = torch.tensor(self.num_X[idx], dtype=torch.float)
        cat_x = torch.tensor(self.cat_X[idx], dtype=torch.long)
        s = self.seq_strings[idx]
        if s:
            arr = np.fromstring(s, sep=",", dtype=np.float32)
        else:
            arr = np.array([0.0], dtype=np.float32)
        seq = torch.from_numpy(arr)
        if self.has_target:
            y = torch.tensor(self.y[idx], dtype=torch.float)
            return num_x, cat_x, seq, y
        else:
            return num_x, cat_x, seq

def collate_fn_train(batch):
    num_x, cat_x, seqs, ys = zip(*batch)
    num_x = torch.stack(num_x)
    cat_x = torch.stack(cat_x)
    ys = torch.stack(ys)
    seqs_padded = nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=0.0)
    seq_lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long)
    seq_lengths = torch.clamp(seq_lengths, min=1)
    return num_x, cat_x, seqs_padded, seq_lengths, ys

def collate_fn_infer(batch):
    num_x, cat_x, seqs = zip(*batch)
    num_x = torch.stack(num_x)
    cat_x = torch.stack(cat_x)
    seqs_padded = nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=0.0)
    seq_lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long)
    seq_lengths = torch.clamp(seq_lengths, min=1)
    return num_x, cat_x, seqs_padded, seq_lengths

### Mirror Gradient ###

def _grad_unit_norm(model, eps=1e-12):
    num = 0.0
    for p in model.parameters():
        if p.grad is None:
            continue
        num += (p.grad.detach()**2).sum().item()
    denom = (num ** 0.5) + eps
    return denom

def mg_perturb_(model, alpha, grad_norm):
    """파라미터를 grad 방향으로 alpha만큼 이동 (정규화 포함)"""
    for p in model.parameters():
        if p.grad is None:
            continue
        p.data.add_(p.grad.detach(), alpha=alpha / grad_norm)

def mg_unperturb_(model, alpha, grad_norm):
    """이동 복구 (원점으로 되돌림)"""
    for p in model.parameters():
        if p.grad is None:
            continue
        p.data.add_(p.grad.detach(), alpha=-(alpha / grad_norm))

MG_ENABLE = True          # 끄려면 False
MG_ALPHA = 0.3            # 거울 이동 크기 (0.1~0.5 권장)
MG_BETA = 1.0             # 최종 grad = g + beta * g_m
GRAD_CLIP_NORM = 5.0      # 선택: 터지는 걸 방지

######################

class CrossNetwork(nn.Module):
    def __init__(self, input_dim, num_layers=2):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(input_dim, 1, bias=True) for _ in range(num_layers)
        ])

    def forward(self, x0):
        x = x0
        for w in self.layers:
            x = x0 * w(x) + x
        return x

class WideDeepCTR(nn.Module):
    def __init__(self, num_features, cat_cardinalities, emb_dim=16, lstm_hidden=64,
                 hidden_units=[512,256,128], dropout=[0.1,0.2,0.3]):
        super().__init__()
        self.emb_layers = nn.ModuleList([
            nn.Embedding(cardinality, emb_dim) for cardinality in cat_cardinalities
        ])
        cat_input_dim = emb_dim * len(cat_cardinalities)
        self.bn_num = nn.BatchNorm1d(num_features)
        self.lstm = nn.LSTM(input_size=1, hidden_size=lstm_hidden,
                            num_layers=2, batch_first=True, bidirectional=True)
        seq_out_dim = lstm_hidden * 2
        self.cross = CrossNetwork(num_features + cat_input_dim + seq_out_dim, num_layers=2)
        input_dim = num_features + cat_input_dim + seq_out_dim
        layers = []
        for i, h in enumerate(hidden_units):
            layers += [nn.Linear(input_dim, h), nn.ReLU(), nn.Dropout(dropout[i % len(dropout)])]
            input_dim = h
        layers += [nn.Linear(input_dim, 1)]
        self.mlp = nn.Sequential(*layers)

    def forward(self, num_x, cat_x, seqs, seq_lengths):
        num_x = self.bn_num(num_x)
        cat_embs = [emb(cat_x[:, i]) for i, emb in enumerate(self.emb_layers)]
        cat_feat = torch.cat(cat_embs, dim=1)
        seqs = seqs.unsqueeze(-1)
        packed = nn.utils.rnn.pack_padded_sequence(seqs, seq_lengths.cpu(),
                                                   batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed)
        h = torch.cat([h_n[-2], h_n[-1]], dim=1)
        z = torch.cat([num_x, cat_feat, h], dim=1)
        z_cross = self.cross(z)
        out = self.mlp(z_cross)
        return out.squeeze(1)

def train_model(train_df, num_cols, cat_cols, seq_col, target_col, batch_size, epochs, lr, device):
    train_dataset = ClickDataset(train_df, num_cols, cat_cols, seq_col, target_col, True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_fn_train, pin_memory=True)
    cat_cardinalities = [len(cat_encoders[c].classes_) for c in cat_cols]

    model = WideDeepCTR(
        num_features=len(num_cols),
        cat_cardinalities=cat_cardinalities,
        emb_dim=16,
        lstm_hidden=64,
        hidden_units=[512,256,128],
        dropout=[0.1,0.2,0.3]
    ).to(device)

    # 클래스 불균형 보정
    pos = float(train_df[target_col].sum())
    neg = float(len(train_df) - pos)
    pos_weight_value = neg / max(pos, 1.0)
    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=2, T_mult=2)

    print("학습 시작 (Mirror Gradient = {})".format(MG_ENABLE))
    for epoch in range(1, epochs+1):
        model.train()
        total_loss = 0.0

        for num_x, cat_x, seqs, lens, ys in tqdm(train_loader, desc=f"[Train Epoch {epoch}]"):
            num_x, cat_x, seqs, lens, ys = num_x.to(device), cat_x.to(device), seqs.to(device), lens.to(device), ys.to(device)

            # ---------- 1) Normal step ----------
            optimizer.zero_grad(set_to_none=True)
            logits = model(num_x, cat_x, seqs, lens)
            loss = criterion(logits, ys)
            loss.backward()

            # (선택) grad clip
            if GRAD_CLIP_NORM is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)

            if MG_ENABLE:
                # g를 복사 저장 (파라미터 순서 보장)
                g_main = []
                for p in model.parameters():
                    if p.grad is None:
                        g_main.append(None)
                    else:
                        g_main.append(p.grad.detach().clone())

                # 현재 grad의 norm (g 기준)
                g_norm = _grad_unit_norm(model)

                # ---------- 2) Mirror step (no optimizer step) ----------
                mg_perturb_(model, alpha=MG_ALPHA, grad_norm=g_norm)

                optimizer.zero_grad(set_to_none=True)
                logits_m = model(num_x, cat_x, seqs, lens)
                loss_m = criterion(logits_m, ys)
                loss_m.backward()

                if GRAD_CLIP_NORM is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)

                # 이동 복구
                mg_unperturb_(model, alpha=MG_ALPHA, grad_norm=g_norm)

                # ---------- 3) 최종 grad = g + beta * g_m ----------
                with torch.no_grad():
                    for p, g in zip(model.parameters(), g_main):
                        if p.grad is None:
                            continue
                        # 현재 p.grad는 g_m
                        if g is not None:
                            p.grad.mul_(MG_BETA).add_(g, alpha=1.0)   # g + beta * g_m
                        else:
                            p.grad.mul_(MG_BETA)  # g가 없으면 g_m만

                optimizer.step()
            else:
                # 일반 학습
                optimizer.step()

            scheduler.step()
            total_loss += loss.item() * ys.size(0)

        total_loss /= len(train_dataset)
        print(f"[Epoch {epoch}] Train Loss: {total_loss:.6f}")
        if torch.cuda.is_available():
            print(f"[DEBUG] GPU Allocated: {torch.cuda.memory_allocated()/1024**2:.2f} MB")

    print("학습 완료")
    return model

print("모델 학습 실행")
model = train_model(
    train_df=train,
    num_cols=num_cols,
    cat_cols=cat_cols,
    seq_col=seq_col,
    target_col=target_col,
    batch_size=CFG['BATCH_SIZE'],
    epochs=CFG['EPOCHS'],
    lr=CFG['LEARNING_RATE'],
    device=device
)

print("추론 시작")
test_dataset = ClickDataset(test, num_cols, cat_cols, seq_col, has_target=False)
test_loader = DataLoader(test_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False,
                         collate_fn=collate_fn_infer, pin_memory=True)
model.eval()
outs = []
with torch.no_grad():
    for num_x, cat_x, seqs, lens in tqdm(test_loader, desc="[Inference]"):
        num_x, cat_x, seqs, lens = num_x.to(device), cat_x.to(device), seqs.to(device), lens.to(device)
        outs.append(torch.sigmoid(model(num_x, cat_x, seqs, lens)).cpu())
test_preds = torch.cat(outs).numpy()
print("추론 완료")

submit = pd.read_csv('../data/sample_submission.csv')
submit['clicked'] = test_preds
submit.to_csv('./wideDeep_MG_submission.csv', index=False)
print("제출 파일 저장 완료")