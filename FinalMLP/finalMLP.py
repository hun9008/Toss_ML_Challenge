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

os.environ["CUDA_VISIBLE_DEVICES"] = "6" 

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

# 네가 명시한 카테고리/수치 분리 (FinalMLP_PT가 이름을 써서 context를 찾으므로 리스트는 '이름'을 유지해야 함)
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
    """FuxiCTR의 MLP_Block과 유사: 마지막 은닉 표현을 반환 (output layer 없음)."""
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
    """FuxiCTR FinalMLP의 Aggregation과 동일 동작."""
    def __init__(self, x_dim, y_dim, output_dim=1, num_heads=1):
        super().__init__()
        assert x_dim % num_heads == 0 and y_dim % num_heads == 0, \
            "Input dim must be divisible by num_heads!"
        self.num_heads = num_heads
        self.output_dim = output_dim
        self.head_x_dim = x_dim // num_heads
        self.head_y_dim = y_dim // num_heads
        self.w_x = nn.Linear(x_dim, output_dim)
        self.w_y = nn.Linear(y_dim, output_dim)
        self.w_xy = nn.Parameter(torch.Tensor(num_heads * self.head_x_dim * self.head_y_dim, output_dim))
        nn.init.xavier_normal_(self.w_xy)

    def forward(self, x, y):
        # (B, out) from linear terms
        out = self.w_x(x) + self.w_y(y)  # (B, output_dim)
        # head interaction
        head_x = x.view(-1, self.num_heads, self.head_x_dim)          # (B, H, Dx)
        head_y = y.view(-1, self.num_heads, self.head_y_dim)          # (B, H, Dy)
        # reshape w_xy to (H, Dx, out)
        w = self.w_xy.view(self.num_heads, self.head_x_dim, -1)       # (H, Dx, out)
        # (B,H,1,Dx) @ (H,Dx,out) -> (B,H,out)
        tmp = torch.matmul(head_x.unsqueeze(2), w).view(-1, self.num_heads, self.output_dim, self.head_y_dim)
        xy = torch.matmul(tmp, head_y.unsqueeze(-1)).squeeze(-1)      # (B,H,out)
        out = out + xy.sum(dim=1)                                     # sum over heads
        return out

class FeatureSelection(nn.Module):
    """FuxiCTR의 FS와 유사: context 임베딩으로 gate 생성 → flat_emb에 곱."""
    def __init__(self, emb_dim, field_names, num_cols, cat_cols,
                 fs1_context, fs2_context, fs_hidden_units=[128]):
        super().__init__()
        self.emb_dim = emb_dim
        self.field_names = field_names  # 전체 필드 이름 순서(= cat + num + ['__seq__'])
        self.fs1_context = fs1_context or []
        self.fs2_context = fs2_context or []

        feature_dim = emb_dim * len(field_names)
        # bias(빈 컨텍스트) 대비
        self.fs1_bias = nn.Parameter(torch.zeros(1, emb_dim))
        self.fs2_bias = nn.Parameter(torch.zeros(1, emb_dim))

        # 게이트 MLP
        in1 = emb_dim * max(1, len(self.fs1_context))
        in2 = emb_dim * max(1, len(self.fs2_context))
        self.fs1_gate = MLP(in1, fs_hidden_units + [feature_dim], dropout=0.0, activation=nn.ReLU, use_bn=False)
        self.fs2_gate = MLP(in2, fs_hidden_units + [feature_dim], dropout=0.0, activation=nn.ReLU, use_bn=False)
        self.sigmoid = nn.Sigmoid()

        # context index 빠르게 참조하기 위한 매핑
        self.name2idx = {name: i for i, name in enumerate(self.field_names)}

    def _gather_context_embs(self, field_embs, ctx_names):
        if len(ctx_names) == 0:
            # (B, emb_dim)
            return self.fs1_bias  # placeholder, 호출부에서 repeat
        idxs = [self.name2idx[n] for n in ctx_names if n in self.name2idx]
        if len(idxs) == 0:
            # 안전장치: 아무것도 못 찾으면 bias 사용
            return self.fs1_bias
        # field_embs: List[(B, emb_dim)] → (B, len(idxs)*emb_dim)
        ctx = torch.cat([field_embs[i] for i in idxs], dim=1)
        return ctx

    def forward(self, field_embs):
        # field_embs: list of (B, emb_dim)
        B = field_embs[0].size(0)
        flat_emb = torch.cat(field_embs, dim=1)          # (B, F*emb_dim)

        # fs1
        if len(self.fs1_context) == 0:
            fs1_in = self.fs1_bias.repeat(B, 1)          # (B, emb_dim)
        else:
            fs1_in = self._gather_context_embs(field_embs, self.fs1_context)
        gt1 = self.sigmoid(self.fs1_gate(fs1_in)) * 1.0  # (B, F*emb_dim)
        feat1 = flat_emb * gt1

        # fs2
        if len(self.fs2_context) == 0:
            fs2_in = self.fs2_bias.repeat(B, 1)
        else:
            fs2_in = self._gather_context_embs(field_embs, self.fs2_context)
        gt2 = self.sigmoid(self.fs2_gate(fs2_in)) * 1.0
        feat2 = flat_emb * gt2

        return feat1, feat2  # 둘 다 (B, F*emb_dim)

class FinalMLP_PT(nn.Module):
    """
    - FuxiCTR FinalMLP를 파이토치 순정으로 재현
    - 각 수치필드를 emb_dim으로 투영, 카테고리는 embedding, seq는 BiLSTM→Linear로 emb_dim으로 투영
    - FS 게이트 후 Dual-MLP → InteractionAggregation
    """
    def __init__(self,
                 num_cols, cat_cardinalities,  # 구조 정의
                 feature_names,                # num_cols + cat_cols + ['__seq__'] 순서의 전체 이름
                 emb_dim=16,
                 lstm_hidden=64,
                 mlp1_units=[512, 256],
                 mlp2_units=[512, 256],
                 fs_hidden_units=[128],
                 fs1_context=['hour', 'gender', 'age_group', 'day_of_week'],
                 fs2_context=['inventory_id', 'l_feat_14'],
                 num_heads=2,
                 dropout=0.1):
        super().__init__()
        self.num_cols = num_cols
        self.cat_cardinalities = cat_cardinalities
        self.emb_dim = emb_dim

        self.num_bn = nn.BatchNorm1d(len(num_cols))

        # 임베딩/프로젝션
        self.cat_embs = nn.ModuleList([nn.Embedding(c, emb_dim) for c in cat_cardinalities])
        self.num_proj = nn.ModuleList([nn.Linear(1, emb_dim) for _ in range(len(num_cols))])

        self.lstm = nn.LSTM(input_size=1, hidden_size=lstm_hidden,
                            num_layers=2, batch_first=True, bidirectional=True)
        self.seq_proj = nn.Linear(lstm_hidden * 2, emb_dim)

        self.field_names = list(cat_cols) + list(num_cols) + ['__seq__']

        self.fs = FeatureSelection(
            emb_dim=emb_dim,
            field_names=self.field_names,
            num_cols=num_cols,
            cat_cols=cat_cols,
            fs1_context=fs1_context,
            fs2_context=fs2_context,
            fs_hidden_units=fs_hidden_units
        )

        feature_dim = emb_dim * len(self.field_names)

        # --- 드롭아웃 켠 MLP ---
        self.mlp1 = MLP(feature_dim, mlp1_units, dropout=dropout, activation=nn.ReLU, use_bn=False)
        self.mlp2 = MLP(feature_dim, mlp2_units, dropout=dropout, activation=nn.ReLU, use_bn=False)

        self.agg = InteractionAggregation(self.mlp1.out_dim, self.mlp2.out_dim,
                                          output_dim=1, num_heads=num_heads)

    def forward(self, num_x, cat_x, seqs, seq_lengths):
        B = num_x.size(0)

        # --- 새로 추가: 수치 입력 BN 적용 ---
        if num_x.ndim == 2 and num_x.size(1) == len(self.num_cols):
            num_x = self.num_bn(num_x)

        cat_fields = [emb(cat_x[:, i]) for i, emb in enumerate(self.cat_embs)]

        num_fields = []
        for i, lin in enumerate(self.num_proj):
            xi = num_x[:, i:i+1]            # (B,1)
            num_fields.append(lin(xi))      # (B,emb_dim)

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
# Train / Infer (그대로, 모델만 교체)
# -----------------------------
def train_model(train_df, num_cols, cat_cols, seq_col, target_col, batch_size, epochs, lr, device):
    train_dataset = ClickDataset(train_df, num_cols, cat_cols, seq_col, target_col, True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_fn_train, pin_memory=True)

    cat_cardinalities = [len(cat_encoders[c].classes_) for c in cat_cols]
    # FinalMLP_PT로 교체
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
        dropout=0.1
    ).to(device)

    # class imbalance 고려 동일
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

            # 배치마다 scheduler.step() ← 삭제!
            total_loss += loss.item() * ys.size(0)

        # ← 에폭이 끝난 뒤에 한 번만
        scheduler.step()

        total_loss /= len(train_dataset)
        print(f"[Epoch {epoch}] Train Loss: {total_loss:.4f}")
        if torch.cuda.is_available():
            print(f"[DEBUG] GPU Allocated: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
    print("학습 완료")

    os.makedirs("./checkpoints", exist_ok=True)
    torch.save({"state_dict": model.state_dict()}, "./checkpoints/finalmlp.pt")
    print("모델 가중치 저장 완료: ./checkpoints/finalmlp.pt")
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
submit.to_csv('./FinalMLP_BN_submission.csv', index=False)
print("제출 파일 저장 완료")