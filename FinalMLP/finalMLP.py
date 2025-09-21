# finalmlp_ctr.py
import os, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, log_loss

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# =========================
# 사용자 설정
# =========================
DATA_DIR = "./data"
TRAIN_PQ = f"{DATA_DIR}/train.parquet"
TEST_PQ  = f"{DATA_DIR}/test.parquet"
SAMPLE_CSV = f"{DATA_DIR}/sample_submission.csv"
TARGET = "clicked"
ID_COL = "ID"
SEQ_COL = "seq"          # FinalMLP에서는 사용하지 않음
SUBMIT_PATH = "./finalmlp_submit.csv"

SEED = 42
TEST_SIZE = 0.1
BATCH_SIZE = 8192
EPOCHS = 8
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 오버샘플링 목표 비율 (학습셋에서 pos:neg ≈ 1:R)
OS_R = 10

def set_seed(seed=SEED):
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed(seed)

# =========================
# 데이터 & 전처리
# =========================
def read_data():
    train = pd.read_parquet(TRAIN_PQ, engine="pyarrow")
    test  = pd.read_parquet(TEST_PQ,  engine="pyarrow")
    # 미사용 열 제거
    for df in (train, test):
        if SEQ_COL in df.columns:
            df.drop(columns=[SEQ_COL], inplace=True)
    return train, test

def split_features(df: pd.DataFrame):
    exclude = {TARGET, ID_COL, SEQ_COL}
    feature_cols = [c for c in df.columns if c not in exclude]
    num_cols = [c for c in feature_cols if np.issubdtype(df[c].dtype, np.number)]
    cat_cols = [c for c in feature_cols if not np.issubdtype(df[c].dtype, np.number)]
    return feature_cols, cat_cols, num_cols

def label_encode_fit_transform(train, test, cat_cols):
    encoders = {}
    for c in cat_cols:
        le = LabelEncoder()
        all_vals = pd.concat([train[c].astype(str), test[c].astype(str)], axis=0)
        le.fit(all_vals.fillna(""))
        train[c] = le.transform(train[c].astype(str).fillna(""))
        test[c]  = le.transform(test[c].astype(str).fillna(""))
        encoders[c] = le
    return train, test, encoders

# =========================
# PyTorch Dataset
# =========================
class CTRDataset(Dataset):
    def __init__(self, df, feature_cols, cat_cols, num_cols, target=None):
        self.df = df.reset_index(drop=True)
        self.feature_cols = feature_cols
        self.cat_cols = cat_cols
        self.num_cols = num_cols
        self.has_target = target is not None
        self.target_col = target

        # 텐서 준비
        self.X_cat = self.df[self.cat_cols].fillna(0).astype(np.int64).values if self.cat_cols else None
        self.X_num = self.df[self.num_cols].fillna(0).astype(np.float32).values if self.num_cols else None
        if self.has_target:
            self.y = self.df[self.target_col].astype(np.float32).values

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        x_cat = torch.from_numpy(self.X_cat[idx]) if self.X_cat is not None else torch.empty(0, dtype=torch.long)
        x_num = torch.from_numpy(self.X_num[idx]) if self.X_num is not None else torch.empty(0, dtype=torch.float32)
        if self.has_target:
            y = torch.tensor(self.y[idx], dtype=torch.float32)
            return x_cat, x_num, y
        else:
            return x_cat, x_num

# =========================
# FinalMLP 구현 (간결화 버전)
# - 공용 임베딩(범주형) + 연속형 concat → 두 개의 MLP 스트림
# - 스트림별 FeatureGate로 입력 차원별 가중(learned gate)
# - 스트림 출력 결합 시, 선형 결합 + 쌍대 상호작용(outer-like)을 압축한 bilinear-style term
# 참고: AAAI'23 / arXiv(2304.00902)
# =========================
class FeatureGate(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(in_dim, in_dim, bias=True),
            nn.Sigmoid()
        )
    def forward(self, x):
        g = self.gate(x)
        return x * g

class MLP(nn.Module):
    def __init__(self, in_dim, hidden=(512,256,128), drop=0.1):
        super().__init__()
        layers = []
        d = in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(drop)]
            d = h
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

class FinalMLP(nn.Module):
    def __init__(self, cat_cardinalities, num_dim, emb_dim=16,
                 hidden=(512,256,128), drop=0.1):
        super().__init__()
        self.has_cat = len(cat_cardinalities) > 0
        self.has_num = num_dim > 0

        if self.has_cat:
            self.emb_layers = nn.ModuleList([nn.Embedding(v, emb_dim) for v in cat_cardinalities])
            emb_out_dim = emb_dim * len(cat_cardinalities)
        else:
            emb_out_dim = 0

        self.in_dim = emb_out_dim + (num_dim if self.has_num else 0)

        # 두 스트림의 게이트
        self.gate1 = FeatureGate(self.in_dim)
        self.gate2 = FeatureGate(self.in_dim)

        # 두 스트림의 MLP
        self.mlp1 = MLP(self.in_dim, hidden=hidden, drop=drop)
        self.mlp2 = MLP(self.in_dim, hidden=hidden, drop=drop)

        # 스트림 상호작용 집계: 선형 결합 + bilinear 유사 항
        out_dim = hidden[-1]
        self.combine_linear = nn.Linear(out_dim * 2, out_dim)
        self.bilinear = nn.Bilinear(out_dim, out_dim, out_dim)

        # 최종 로짓
        self.logit = nn.Linear(out_dim, 1)

        # 연속형 정규화(옵션): 배치정규화
        if self.has_num:
            self.bn_num = nn.BatchNorm1d(num_dim)

    def embed_cat(self, x_cat):
        if not self.has_cat:
            return None
        embs = [emb(x_cat[:, i]) for i, emb in enumerate(self.emb_layers)]
        return torch.cat(embs, dim=1)  # (B, emb_dim * n_cat)

    def forward(self, x_cat, x_num):
        zs = []
        if self.has_cat:
            z_cat = self.embed_cat(x_cat)
            zs.append(z_cat)
        if self.has_num:
            # (B, num_dim) → BN
            z_num = self.bn_num(x_num)
            zs.append(z_num)
        z = torch.cat(zs, dim=1) if len(zs) > 1 else zs[0]

        # 두 스트림: 게이트 → MLP
        z1 = self.gate1(z)
        z2 = self.gate2(z)
        h1 = self.mlp1(z1)
        h2 = self.mlp2(z2)

        # 상호작용 집계
        h = self.combine_linear(torch.cat([h1, h2], dim=1)) + self.bilinear(h1, h2)
        h = torch.relu(h)

        logit = self.logit(h).squeeze(1)
        return logit

# =========================
# 학습/평가 루프
# =========================
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total = 0; loss_sum = 0
    for x_cat, x_num, y in loader:
        x_cat = x_cat.to(DEVICE)
        x_num = x_num.to(DEVICE)
        y = y.to(DEVICE)

        optimizer.zero_grad()
        logit = model(x_cat, x_num)
        loss = criterion(logit, y)
        loss.backward()
        optimizer.step()

        bs = y.size(0)
        total += bs
        loss_sum += loss.item() * bs
    return loss_sum / total

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    ys = []; ps = []
    for batch in loader:
        if len(batch) == 3:
            x_cat, x_num, y = batch
        else:
            x_cat, x_num = batch; y = None
        x_cat = x_cat.to(DEVICE)
        x_num = x_num.to(DEVICE)
        logit = model(x_cat, x_num)
        prob = torch.sigmoid(logit).detach().cpu().numpy()
        ps.append(prob)
        if y is not None:
            ys.append(y.numpy())
    ps = np.concatenate(ps).reshape(-1)
    if ys:
        ys = np.concatenate(ys).reshape(-1)
        return ys, ps
    else:
        return None, ps

def main():
    set_seed()

    # 1) 데이터 로드
    train, test = read_data()
    assert TARGET in train.columns

    # 라벨 0/1 보장
    train[TARGET] = (pd.to_numeric(train[TARGET], errors="coerce").fillna(0) > 0).astype(np.float32)

    # 2) 결측 간단 처리
    for df in (train, test):
        for c in df.columns:
            if np.issubdtype(df[c].dtype, np.number):
                df[c] = df[c].fillna(0)
            else:
                df[c] = df[c].fillna("")

    # 3) 피처 분리
    feature_cols, cat_cols, num_cols = split_features(train)

    # 4) 범주형 인코딩 (train+test 합쳐 fit)
    train, test, _ = label_encode_fit_transform(train, test, cat_cols)

    # 5) Train/Valid split (검증은 원분포 유지)
    X = train[feature_cols].copy()
    y = train[TARGET].values.astype(np.float32)

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y
    )

    # 6) 오버샘플링(양성만 복제; 언더샘플링 없음)
    tr = X_train.copy(); tr[TARGET] = y_train
    pos_df = tr[tr[TARGET] == 1]; neg_df = tr[tr[TARGET] == 0]
    pos, neg = len(pos_df), len(neg_df)
    needed_pos = int(np.ceil(neg / OS_R))
    mult = max(1, needed_pos // max(pos,1))
    rem  = max(0, needed_pos - mult * pos)
    pos_rep = pd.concat([pos_df]*mult, ignore_index=True)
    if rem > 0: pos_rep = pd.concat([pos_rep, pos_df.sample(n=rem, random_state=SEED)], ignore_index=True)
    tr_os = pd.concat([neg_df, pos_rep], ignore_index=True).sample(frac=1, random_state=SEED).reset_index(drop=True)

    y_train_os = tr_os[TARGET].values.astype(np.float32)
    X_train_os = tr_os.drop(columns=[TARGET])

    # 7) 파이토치 데이터로더
    def build_loaders(df_trainX, y_trainX, df_validX):
        # cardinalities (각 범주형 컬럼별 max+1)
        cat_cards = [int(max(train[c].max(), test[c].max())) + 1 for c in cat_cols] if cat_cols else []
        num_dim = len(num_cols)

        train_ds = CTRDataset(
            pd.concat([df_trainX.reset_index(drop=True), pd.Series(y_trainX, name=TARGET)], axis=1),
            feature_cols, cat_cols, num_cols, target=TARGET
        )
        valid_df = pd.concat([X_valid.reset_index(drop=True), pd.Series(y_valid, name=TARGET)], axis=1)
        valid_ds = CTRDataset(valid_df, feature_cols, cat_cols, num_cols, target=TARGET)
        test_ds  = CTRDataset(test[feature_cols], feature_cols, cat_cols, num_cols, target=None)

        train_ld = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
        valid_ld = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
        test_ld  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
        return cat_cards, num_dim, train_ld, valid_ld, test_ld

    cat_cards, num_dim, train_ld, valid_ld, test_ld = build_loaders(X_train_os, y_train_os, X_valid)

    # 8) 모델/옵티마/로스
    model = FinalMLP(cat_cards, num_dim, emb_dim=16, hidden=(512,256,128), drop=0.1).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # 9) 학습 루프 + EarlyStop(patience=2)
    best_auc = -1; patience = 2; bad = 0
    for epoch in range(1, EPOCHS+1):
        tr_loss = train_one_epoch(model, train_ld, criterion, optimizer)
        yv, pv = evaluate(model, valid_ld)
        v_loss = log_loss(yv, pv, eps=1e-7); v_auc = roc_auc_score(yv, pv)
        print(f"[Epoch {epoch}] train_loss={tr_loss:.5f}  val_logloss={v_loss:.5f}  val_auc={v_auc:.5f}")

        if v_auc > best_auc:
            best_auc = v_auc; bad = 0
            torch.save(model.state_dict(), "finalmlp_best.pt")
        else:
            bad += 1
            if bad >= patience:
                print("Early stopping.")
                break

    # 10) 로드 & 테스트 예측
    model.load_state_dict(torch.load("finalmlp_best.pt", map_location=DEVICE))
    _, test_pred = evaluate(model, test_ld)  # shape (N,)

    # 11) ID 정렬 + (선택) 사전확률 보정
    submit = pd.read_csv(SAMPLE_CSV)
    assert ID_COL in submit.columns and ID_COL in test.columns
    pred_df = pd.DataFrame({ID_COL: test[ID_COL].values, "clicked": test_pred.astype(float)})
    submit[ID_COL] = submit[ID_COL].astype(pred_df[ID_COL].dtype)
    pred_df = pred_df.set_index(ID_COL).reindex(submit[ID_COL]).reset_index()
    if pred_df["clicked"].isna().any():
        pred_df["clicked"] = pred_df["clicked"].fillna(0.0)

    # (옵션) 베이스레이트 보정 (LogLoss 안정화에 유리)
    def prior_correction(p_hat, pi_true, pi_train):
        p_hat = np.clip(p_hat, 1e-7, 1-1e-7)
        num = (pi_true / pi_train) * p_hat
        den = num + ((1 - pi_true) / (1 - pi_train)) * (1 - p_hat)
        return np.clip(num / den, 1e-7, 1-1e-7)

    pi_true  = (train[TARGET].astype(float) > 0).mean()
    pi_train = (y_train_os == 1).mean()
    print("pi_true/pi_train:", pi_true, pi_train)
    # 필요시 아래 줄 주석 해제
    # pred_df["clicked"] = prior_correction(pred_df["clicked"].values, pi_true, pi_train)

    out = submit[[ID_COL]].copy()
    out["clicked"] = pred_df["clicked"].astype(float)
    assert (out[ID_COL].values == submit[ID_COL].values).all()
    out.to_csv(SUBMIT_PATH, index=False)
    print("Saved:", SUBMIT_PATH)

if __name__ == "__main__":
    main()