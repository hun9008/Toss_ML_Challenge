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

os.environ["CUDA_VISIBLE_DEVICES"] = "5"

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

# FinalMLP_PT가 이름을 써서 context를 찾으므로 리스트는 '이름' 유지
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

# ==========================
# DEBUG MODE: 작은 데이터만 사용
# ==========================
DEBUG_MODE = True            # 빠른 연기 테스트 시 True
DEBUG_POS = 2000             # 사용할 양성 샘플 수
DEBUG_NEG_PER_POS = 2        # 음성:양성 비율
DEBUG_TEST_ROWS = 10000      # 테스트도 일부만

if DEBUG_MODE:
    pos_df = train[train[target_col] == 1]
    neg_df = train[train[target_col] == 0]
    use_pos = min(DEBUG_POS, len(pos_df))
    use_neg = min(use_pos * DEBUG_NEG_PER_POS, len(neg_df))
    train_small = pd.concat([
        pos_df.sample(n=use_pos, random_state=42),
        neg_df.sample(n=use_neg, random_state=42)
    ], axis=0).sample(frac=1.0, random_state=42).reset_index(drop=True)
    test_small = test.sample(n=min(DEBUG_TEST_ROWS, len(test)), random_state=42).reset_index(drop=True)
    print(f"[DEBUG] Using tiny dataset: train={len(train_small)} (pos={use_pos}, neg={use_neg}), test={len(test_small)}")
    train = train_small
    test = test_small

# -----------------------------
# Dataset / Collate (그대로)
# -----------------------------
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

# -----------------------------
# FinalMLP (PyTorch native)
# -----------------------------
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_units, dropout=0.0, activation=nn.ReLU, use_bn=False):
        super().__init__()
        layers = []
        last = input_dim
        for h in hidden_units:
            layers.append(nn.Linear(last, h))
            if use_bn:
                layers.append(nn.BatchNorm1d(h))
            layers.append(activation())
            if dropout and dropout > 0:
                layers.append(nn.Dropout(dropout))
            last = h
        self.net = nn.Sequential(*layers)
        self.out_dim = last
    def forward(self, x):
        return self.net(x)

class InteractionAggregation(nn.Module):
    def __init__(self, x_dim, y_dim, output_dim=1, num_heads=1):
        super().__init__()
        assert x_dim % num_heads == 0 and y_dim % num_heads == 0, "Input dim must be divisible by num_heads!"
        self.num_heads = num_heads
        self.output_dim = output_dim
        self.head_x_dim = x_dim // num_heads
        self.head_y_dim = y_dim // num_heads
        self.w_x = nn.Linear(x_dim, output_dim)
        self.w_y = nn.Linear(y_dim, output_dim)
        self.w_xy = nn.Parameter(torch.Tensor(num_heads * self.head_x_dim * self.head_y_dim, output_dim))
        nn.init.xavier_normal_(self.w_xy)
    def forward(self, x, y):
        out = self.w_x(x) + self.w_y(y)
        head_x = x.view(-1, self.num_heads, self.head_x_dim)
        head_y = y.view(-1, self.num_heads, self.head_y_dim)
        w = self.w_xy.view(self.num_heads, self.head_x_dim, -1)
        tmp = torch.matmul(head_x.unsqueeze(2), w).view(-1, self.num_heads, self.output_dim, self.head_y_dim)
        xy = torch.matmul(tmp, head_y.unsqueeze(-1)).squeeze(-1)
        out = out + xy.sum(dim=1)
        return out

class FeatureSelection(nn.Module):
    def __init__(self, emb_dim, field_names, num_cols, cat_cols,
                 fs1_context, fs2_context, fs_hidden_units=[128]):
        super().__init__()
        self.emb_dim = emb_dim
        self.field_names = field_names
        self.fs1_context = fs1_context or []
        self.fs2_context = fs2_context or []
        feature_dim = emb_dim * len(field_names)
        self.fs1_bias = nn.Parameter(torch.zeros(1, emb_dim))
        self.fs2_bias = nn.Parameter(torch.zeros(1, emb_dim))
        in1 = emb_dim * max(1, len(self.fs1_context))
        in2 = emb_dim * max(1, len(self.fs2_context))
        self.fs1_gate = MLP(in1, fs_hidden_units + [feature_dim], dropout=0.0, activation=nn.ReLU, use_bn=False)
        self.fs2_gate = MLP(in2, fs_hidden_units + [feature_dim], dropout=0.0, activation=nn.ReLU, use_bn=False)
        self.sigmoid = nn.Sigmoid()
        self.name2idx = {name: i for i, name in enumerate(self.field_names)}
    def _gather_context_embs(self, field_embs, ctx_names):
        if len(ctx_names) == 0:
            return self.fs1_bias
        idxs = [self.name2idx[n] for n in ctx_names if n in self.name2idx]
        if len(idxs) == 0:
            return self.fs1_bias
        ctx = torch.cat([field_embs[i] for i in idxs], dim=1)
        return ctx
    def forward(self, field_embs):
        B = field_embs[0].size(0)
        flat_emb = torch.cat(field_embs, dim=1)
        if len(self.fs1_context) == 0:
            fs1_in = self.fs1_bias.repeat(B, 1)
        else:
            fs1_in = self._gather_context_embs(field_embs, self.fs1_context)
        gt1 = self.sigmoid(self.fs1_gate(fs1_in)) * 2.0
        feat1 = flat_emb * gt1
        if len(self.fs2_context) == 0:
            fs2_in = self.fs2_bias.repeat(B, 1)
        else:
            fs2_in = self._gather_context_embs(field_embs, self.fs2_context)
        gt2 = self.sigmoid(self.fs2_gate(fs2_in)) * 2.0
        feat2 = flat_emb * gt2
        return feat1, feat2

class FinalMLP_PT(nn.Module):
    def __init__(self,
                 num_cols, cat_cardinalities,
                 feature_names,
                 emb_dim=16,
                 lstm_hidden=64,
                 mlp1_units=[512, 256],
                 mlp2_units=[512, 256],
                 fs_hidden_units=[128],
                 fs1_context=['hour', 'gender', 'age_group', 'day_of_week'],
                 fs2_context=['inventory_id', 'l_feat_14'],
                 num_heads=2,
                 dropout=0.0):
        super().__init__()
        self.num_cols = num_cols
        self.cat_cardinalities = cat_cardinalities
        self.emb_dim = emb_dim
        self.cat_embs = nn.ModuleList([nn.Embedding(c, emb_dim) for c in cat_cardinalities])
        self.num_proj = nn.ModuleList([nn.Linear(1, emb_dim) for _ in range(len(num_cols))])
        self.lstm = nn.LSTM(input_size=1, hidden_size=lstm_hidden,
                            num_layers=2, batch_first=True, bidirectional=True)
        seq_out_dim = lstm_hidden * 2
        self.seq_proj = nn.Linear(seq_out_dim, emb_dim)
        self.field_names = list(cat_cols) + list(num_cols) + ['__seq__']
        self.fs = FeatureSelection(emb_dim=emb_dim,
                                   field_names=self.field_names,
                                   num_cols=num_cols,
                                   cat_cols=cat_cols,
                                   fs1_context=fs1_context,
                                   fs2_context=fs2_context,
                                   fs_hidden_units=fs_hidden_units)
        feature_dim = emb_dim * len(self.field_names)
        self.mlp1 = MLP(feature_dim, mlp1_units, dropout=dropout, activation=nn.ReLU, use_bn=False)
        self.mlp2 = MLP(feature_dim, mlp2_units, dropout=dropout, activation=nn.ReLU, use_bn=False)
        self.agg = InteractionAggregation(self.mlp1.out_dim, self.mlp2.out_dim, output_dim=1, num_heads=num_heads)
    def forward(self, num_x, cat_x, seqs, seq_lengths):
        B = num_x.size(0)
        cat_fields = [emb(cat_x[:, i]) for i, emb in enumerate(self.cat_embs)]
        num_fields = []
        for i, lin in enumerate(self.num_proj):
            xi = num_x[:, i:i+1]
            num_fields.append(lin(xi))
        seqs = seqs.unsqueeze(-1)
        packed = nn.utils.rnn.pack_padded_sequence(seqs, seq_lengths.cpu(),
                                                   batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed)
        h = torch.cat([h_n[-2], h_n[-1]], dim=1)
        seq_field = self.seq_proj(h)
        field_embs = cat_fields + num_fields + [seq_field]
        feat1, feat2 = self.fs(field_embs)
        x = self.mlp1(feat1)
        y = self.mlp2(feat2)
        logit = self.agg(x, y).squeeze(1)
        return logit

# -----------------------------
# Train / Infer
# -----------------------------
def train_model(train_df, num_cols, cat_cols, seq_col, target_col, batch_size, epochs, lr, device):
    train_dataset = ClickDataset(train_df, num_cols, cat_cols, seq_col, target_col, True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_fn_train, pin_memory=True)
    cat_cardinalities = [len(cat_encoders[c].classes_) for c in cat_cols]
    model = FinalMLP_PT(
        num_cols=num_cols,
        cat_cardinalities=cat_cardinalities,
        feature_names=cat_cols + num_cols + ['__seq__'],
        emb_dim=16,
        lstm_hidden=64,
        mlp1_units=[512, 256],
        mlp2_units=[512, 256],
        fs_hidden_units=[256],
        fs1_context=['hour', 'gender', 'age_group', 'day_of_week'],
        fs2_context=['inventory_id', 'l_feat_14'],
        num_heads=2,
        dropout=0.0
    ).to(device)
    pos = float(train_df[target_col].sum())
    neg = float(len(train_df) - pos)
    pos_weight_value = neg / max(pos, 1.0)
    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=2, T_mult=2)
    print("학습 시작")
    for epoch in range(1, epochs+1):
        model.train()
        total_loss = 0.0
        for num_x, cat_x, seqs, lens, ys in tqdm(train_loader, desc=f"[Train Epoch {epoch}]"):
            num_x, cat_x, seqs, lens, ys = num_x.to(device), cat_x.to(device), seqs.to(device), lens.to(device), ys.to(device)
            optimizer.zero_grad()
            logits = model(num_x, cat_x, seqs, lens)
            loss = criterion(logits, ys)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item() * ys.size(0)
        total_loss /= len(train_dataset)
        print(f"[Epoch {epoch}] Train Loss: {total_loss:.4f}")
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
submit.to_csv('./FinalMLP_submission.csv', index=False)
print("제출 파일 저장 완료")