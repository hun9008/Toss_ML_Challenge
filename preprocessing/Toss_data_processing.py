from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
import json
import numpy as np

data_dir = Path('/content/drive/MyDrive/toss/data')
dataset_id = 'toss_ctr_v1'
out_dir = data_dir / dataset_id
out_dir.mkdir(parents=True, exist_ok=True)

all_train = pd.read_parquet(data_dir / 'train.parquet', engine='pyarrow')
test_df = pd.read_parquet(data_dir / 'test.parquet', engine='pyarrow').drop(columns=['ID'], errors='ignore')

print("Train shape:", all_train.shape)
print("Test shape:", test_df.shape)

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

exclude_cols = {'ID', 'seq'}

for df in (train_df, valid_df, test_df):
    drop_exist = [c for c in exclude_cols if c in df.columns]
    if drop_exist:
        df.drop(columns=drop_exist, inplace=True, errors='ignore')

def current_feature_cols(df_list, label):
    cols = set()
    for d in df_list:
        cols |= set(d.columns)
    cols = [c for c in sorted(cols) if c != label]
    return cols

feature_cols = current_feature_cols([train_df, valid_df, test_df], label_col)

def is_numeric(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series)

def infer_dtype(col: str) -> str:
    for df in (train_df, valid_df, test_df):
        if col in df.columns:
            return 'numeric' if is_numeric(df[col]) else 'categorical'
    return 'categorical'

feat_types = {c: infer_dtype(c) for c in feature_cols}

def build_vocab(col: str):
    uniques = set()
    for df in (train_df, valid_df, test_df):
        if col in df.columns:
            vals = df[col].astype('string').fillna('<NA>').unique().tolist()
            uniques.update(vals)
    vocab = sorted(list(uniques))
    return {v: i+1 for i, v in enumerate(vocab)}  # 0은 padding

cat_mappings = {c: build_vocab(c) for c, t in feat_types.items() if t == 'categorical'}

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
                df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0).astype('float32')
            else:
                df[c] = np.float32(0.0)
    # 열 순서 고정
    cols_order = [label_col] + feature_cols if has_label else feature_cols
    return df[cols_order]

train_enc = encode_inplace(train_df.copy(), has_label=True)
valid_enc = encode_inplace(valid_df.copy(), has_label=True)
test_enc  = encode_inplace(test_df.copy(),  has_label=False)

# 1) 인코딩된 DF만 저장 (이것만 남겨두기)
train_enc.to_parquet(out_dir / 'train.parquet', index=False)
valid_enc.to_parquet(out_dir / 'valid.parquet', index=False)

# 2) test에도 라벨 컬럼 추가(전부 0.0으로 두면 됨. 점수 산출용이므로 OK)
test_enc_with_label = test_enc.copy()
test_enc_with_label.insert(0, label_col, np.zeros(len(test_enc), dtype=np.float32))
test_enc_with_label.to_parquet(out_dir / 'test.parquet', index=False)

print("✔ 최종 저장 완료:", out_dir)

features_list = []
total_features = 0
for c in feature_cols:
    if feat_types[c] == 'categorical':
        vocab_size = len(cat_mappings[c]) + 1  # padding 포함
        features_list.append({
            c: {
                "source": "",
                "type": "categorical",
                "padding_idx": 0,
                "vocab_size": int(vocab_size)
            }
        })
        total_features += vocab_size
    else:
        features_list.append({
            c: {
                "source": "",
                "type": "numeric"
            }
        })

feature_map = {
    "dataset_id": dataset_id,
    "num_fields": len(feature_cols),
    "total_features": int(total_features),
    "input_length": len(feature_cols),
    "labels": [label_col],
    "features": features_list
}

with open(out_dir / 'feature_map.json', 'w', encoding='utf-8') as f:
    json.dump(feature_map, f, ensure_ascii=False, indent=4)

print("✔ feature_map.json 저장 완료:", out_dir / 'feature_map.json')
print("   num_fields =", feature_map['num_fields'],
      " total_features =", feature_map['total_features'])
