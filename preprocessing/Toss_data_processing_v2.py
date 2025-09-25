# ======================
# 0) Colab / Imports
# ======================
from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
import json
import numpy as np

# ======================
# 1) Paths & Load
# ======================
data_dir = Path('/content/drive/MyDrive/toss/data')
dataset_id = 'toss_ctr_v1'
out_dir = data_dir / dataset_id
out_dir.mkdir(parents=True, exist_ok=True)

all_train = pd.read_parquet(data_dir / 'train.parquet', engine='pyarrow')
test_df  = pd.read_parquet(data_dir / 'test.parquet', engine='pyarrow').drop(columns=['ID'], errors='ignore')

print("Train shape:", all_train.shape)
print("Test shape:",  test_df.shape)

# ======================
# 2) Train balance & split
# ======================
label_col = 'clicked'
pos = all_train[all_train[label_col] == 1]
neg = all_train[all_train[label_col] == 0].sample(n=len(pos)*2, random_state=42)
train_bal = pd.concat([pos, neg], axis=0).sample(frac=1, random_state=42).reset_index(drop=True)
print("Balanced train shape:", train_bal.shape)

train_df, valid_df = train_test_split(
    train_bal,
    test_size=0.1,
    random_state=42,
    stratify=train_bal[label_col]
)
print("Train split shape:", train_df.shape)
print("Valid split shape:", valid_df.shape)

# ======================
# 3) Drop columns not used
# ======================
exclude_cols = {'ID', 'seq'}
for df in (train_df, valid_df, test_df):
    drop_exist = [c for c in exclude_cols if c in df.columns]
    if drop_exist:
        df.drop(columns=drop_exist, inplace=True, errors='ignore')

# ======================
# 4) Feature columns (train/valid ONLY)
# ======================
def current_feature_cols(df_list, label):
    cols = set()
    for d in df_list:
        cols |= set(d.columns)
    cols = [c for c in sorted(cols) if c != label]
    return cols

feature_cols = current_feature_cols([train_df, valid_df], label_col)

# ======================
# 5) DType inference (with forced categoricals)
# ======================
def is_numeric(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series)

# ★ 범주형 강제: 정수라도 카테고리로 써야 하는 컬럼들
CATEGORICAL_FORCE = {'gender', 'age_group', 'inventory_id', 'day_of_week', 'hour'}

def infer_dtype(col: str) -> str:
    if col in CATEGORICAL_FORCE:
        return 'categorical'
    for df in (train_df, valid_df):
        if col in df.columns:
            return 'numeric' if is_numeric(df[col]) else 'categorical'
    return 'categorical'

feat_types = {c: infer_dtype(c) for c in feature_cols}

# ======================
# 6) (중요) Numeric robust scaling (train 통계로만)
# ======================
num_cols = [c for c, t in feat_types.items() if t == 'numeric']

# train 기준 통계
robust_stats = {}
for c in num_cols:
    if c in train_df.columns:
        s = pd.to_numeric(train_df[c], errors='coerce')
        med = np.nanmedian(s)
        mad = np.nanmedian(np.abs(s - med)) + 1e-6
        robust_stats[c] = (float(med), float(mad))

def scale_numeric_inplace(df: pd.DataFrame):
    for c in num_cols:
        if c in df.columns:
            s = pd.to_numeric(df[c], errors='coerce').astype('float32').values
            med, mad = robust_stats.get(c, (0.0, 1.0))
            z = (s - med) / mad
            z = np.clip(z, -10, 10)  # 포화 방지용
            df[c] = z.astype('float32')

# encode 전에 원본에 스케일 적용
scale_numeric_inplace(train_df)
scale_numeric_inplace(valid_df)
scale_numeric_inplace(test_df)

# ======================
# 7) Categorical vocab (train+valid ONLY)
# ======================
def build_vocab(col: str):
    uniques = set()
    for df in (train_df, valid_df):
        if col in df.columns:
            vals = df[col].astype('string').fillna('<NA>').unique().tolist()
            uniques.update(vals)
    vocab = sorted(list(uniques))
    return {v: i+1 for i, v in enumerate(vocab)}  # 0 = padding/UNK

cat_mappings = {c: build_vocab(c) for c, t in feat_types.items() if t == 'categorical'}

# ======================
# 8) Encode function
# ======================
def encode_inplace(df: pd.DataFrame, has_label: bool):
    if has_label and label_col in df.columns:
        df[label_col] = pd.to_numeric(df[label_col], errors='coerce').fillna(0).astype('float32')
    for c in feature_cols:
        if feat_types[c] == 'categorical':
            if c in df.columns:
                df[c] = df[c].astype('string').fillna('<NA>').map(cat_mappings[c]).fillna(0).astype('int32')
            else:
                df[c] = np.int32(0)
        else:
            if c in df.columns:
                # 스케일은 이미 적용되어 있지만 안전하게 한 번 더 보정
                df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0).astype('float32')
            else:
                df[c] = np.float32(0.0)
    cols_order = [label_col] + feature_cols if has_label else feature_cols
    return df[cols_order]

# ======================
# 9) Encode & Save
# ======================
train_enc = encode_inplace(train_df.copy(), has_label=True)
valid_enc = encode_inplace(valid_df.copy(), has_label=True)
test_enc  = encode_inplace(test_df.copy(),  has_label=False)

# 저장
train_enc.to_parquet(out_dir / 'train.parquet', index=False)
valid_enc.to_parquet(out_dir / 'valid.parquet', index=False)

test_enc_with_label = test_enc.copy()
test_enc_with_label.insert(0, label_col, np.zeros(len(test_enc), dtype=np.float32))
test_enc_with_label.to_parquet(out_dir / 'test.parquet', index=False)

print("✔ 최종 저장 완료:", out_dir)

# ======================
# 10) feature_map.json 생성 (total_features는 카테고리 vocab 합만)
# ======================
features_list = []
for c in feature_cols:
    if feat_types[c] == 'categorical':
        vocab_size = len(cat_mappings[c]) + 1  # padding 포함(=0)
        features_list.append({
            c: {
                "source": "",
                "type": "categorical",
                "padding_idx": 0,
                "vocab_size": int(vocab_size)
            }
        })
    else:
        features_list.append({
            c: {
                "source": "",
                "type": "numeric"
            }
        })

# total_features = 카테고리 vocab_size 합
total_features = int(sum(
    int(list(f.values())[0].get("vocab_size", 0))
    for f in features_list
))

feature_map = {
    "dataset_id": dataset_id,
    "num_fields": len(feature_cols),
    "total_features": total_features,
    "input_length": len(feature_cols),
    "labels": [label_col],
    "features": features_list
}

with open(out_dir / 'feature_map.json', 'w', encoding='utf-8') as f:
    json.dump(feature_map, f, ensure_ascii=False, indent=4)

print("✔ feature_map.json 저장 완료:", out_dir / 'feature_map.json')
print("   num_fields =", feature_map['num_fields'],
      " total_features =", feature_map['total_features'])

# ======================
# 11) --- 무결성/위생 체크 블록 ---
# ======================

print("\n[CHECK] 라벨 분포")
print("train mean:", float(train_enc[label_col].mean()),
      " valid mean:", float(valid_enc[label_col].mean()))
print("train value_counts:\n", train_enc[label_col].value_counts())
print("valid value_counts:\n", valid_enc[label_col].value_counts())

print("\n[CHECK] 카테고리 non-zero 비율(훈련)")
cat_cols = [c for c, t in feat_types.items() if t == 'categorical']
for c in cat_cols:
    if c in train_enc.columns:
        nz = (train_enc[c] != 0).mean()
        print(f"{c:20s} nonzero_rate(train) = {nz:.3f}")

print("\n[CHECK] numeric zero-variance (훈련)")
num_cols = [c for c, t in feat_types.items() if t == 'numeric']
zero_vars = []
for c in num_cols:
    if c in train_enc.columns:
        if np.nanstd(train_enc[c].values) == 0:
            zero_vars.append(c)
print("zero-var numeric (train):", zero_vars[:20])

print("\n[CHECK] feature_map total_features vs sum(vocabs)")
sum_vocabs = sum(int(list(f.values())[0].get("vocab_size", 0)) for f in feature_map['features'])
print("feature_map['total_features'] =", feature_map['total_features'], " ; sum_vocabs =", sum_vocabs)

print("\n[CHECK] Encoded dtypes(head)")
print(train_enc.head(1).dtypes.to_string())