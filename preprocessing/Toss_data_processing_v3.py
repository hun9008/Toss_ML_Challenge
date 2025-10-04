import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
import json
import numpy as np
from sklearn.preprocessing import LabelEncoder

# ======================
# 1) Paths & Load
# ======================
data_dir = Path('/home/hun/CTR_Prediction/Toss_ML_Challenge/data')
FuxiCTR_data_dir = Path("/home/hun/CTR_Prediction/Toss_ML_Challenge/FuxiCTR/data")
dataset_id = 'toss_ctr_v3'
out_dir = FuxiCTR_data_dir / dataset_id
out_dir.mkdir(parents=True, exist_ok=True)

all_train = pd.read_parquet(data_dir / 'train.parquet', engine='pyarrow')
test_df  = pd.read_parquet(data_dir / 'test.parquet',  engine='pyarrow')  # ID/seq 유지
print("Train shape:", all_train.shape)
print("Test shape:",  test_df.shape)

# ======================
# 2) Split (전체 데이터 사용, 다운샘플링 없음)
# ======================
label_col = 'clicked'
assert label_col in all_train.columns, "clicked 라벨이 필요합니다."

train_df, valid_df = train_test_split(
    all_train,
    test_size=0.1,
    random_state=42,
    stratify=all_train[label_col]
)
print("Train split shape:", train_df.shape)
print("Valid split shape:", valid_df.shape)

# ======================
# 3) Feature sets (PyTorch 코드와 동일 정책)
# ======================
seq_col = "seq"
FEATURE_EXCLUDE = {label_col, seq_col, "ID"}

# PyTorch 코드에서 명시한 카테고리 컬럼만 사용
cat_cols = ["gender", "age_group", "inventory_id", "l_feat_14"]

# feature_cols = 라벨/seq/ID 제외 전 컬럼
all_cols = sorted(set(train_df.columns) | set(valid_df.columns))
feature_cols = [c for c in all_cols if c not in FEATURE_EXCLUDE]

# num_cols = feature_cols - cat_cols
num_cols = [c for c in feature_cols if c not in cat_cols]

# ======================
# 4) Categorical encoding (train+test 결합으로 fit)
# ======================
def encode_categoricals(train_df, valid_df, test_df, cat_cols):
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        # NaN 처리 후 문자열화 (fillna를 먼저!)
        all_vals = pd.concat(
            [train_df.get(col, pd.Series(dtype=object)),
             valid_df.get(col, pd.Series(dtype=object)),
             test_df.get(col,  pd.Series(dtype=object))],
            axis=0
        ).fillna("UNK").astype(str)
        le.fit(all_vals)

        if col in train_df.columns:
            train_df[col] = le.transform(train_df[col].fillna("UNK").astype(str))
        else:
            train_df[col] = 0

        if col in valid_df.columns:
            valid_df[col] = le.transform(valid_df[col].fillna("UNK").astype(str))
        else:
            valid_df[col] = 0

        if col in test_df.columns:
            test_df[col]  = le.transform(test_df[col].fillna("UNK").astype(str))
        else:
            test_df[col] = 0

        encoders[col] = le
        print(f"[Enc] {col} classes = {len(le.classes_)}")
    return train_df, valid_df, test_df, encoders

train_df, valid_df, test_df, cat_encoders = encode_categoricals(train_df.copy(), valid_df.copy(), test_df.copy(), cat_cols)

# ======================
# 5) Numeric casting (간단: float32)
# ======================
def cast_numeric_inplace(df, num_cols):
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0).astype('float32')
        else:
            df[c] = np.float32(0.0)

cast_numeric_inplace(train_df, num_cols)
cast_numeric_inplace(valid_df, num_cols)
cast_numeric_inplace(test_df,  num_cols)

# ======================
# 6) 카테고리/라벨 캐스팅
# ======================
for c in cat_cols:
    train_df[c] = train_df[c].astype("int32")
    valid_df[c] = valid_df[c].astype("int32")
    test_df[c]  = test_df[c].astype("int32")

train_df[label_col] = pd.to_numeric(train_df[label_col], errors='coerce').fillna(0).astype('float32')
valid_df[label_col] = pd.to_numeric(valid_df[label_col], errors='coerce').fillna(0).astype('float32')

# ======================
# 7) 최종 컬럼 정렬 + seq 보존
#     - feature_cols 는 라벨/seq/ID 제외 (모델 입력 피처)
#     - 저장 파일에는 seq 컬럼을 **그대로 추가**하여 PyTorch에서 사용 가능하게 함
# ======================
def arrange_and_keep_seq(df, has_label=True):
    cols_order = [label_col] + feature_cols if has_label else feature_cols
    df_out = df[cols_order].copy()
    # seq를 파일에 보존 (feature_map에는 포함하지 않음)
    if seq_col in df.columns:
        # 문자열로 통일 저장
        df_out[seq_col] = df[seq_col].astype("string")
    return df_out

train_enc = arrange_and_keep_seq(train_df, has_label=True)
valid_enc = arrange_and_keep_seq(valid_df, has_label=True)
test_enc  = arrange_and_keep_seq(test_df,  has_label=False)

# ======================
# 8) Save
# ======================
train_enc.to_parquet(out_dir / 'train.parquet', index=False)
valid_enc.to_parquet(out_dir / 'valid.parquet', index=False)

# 테스트에는 라벨 0을 맨 앞에 추가(스키마 정렬 목적). seq는 그대로 유지.
test_enc_with_label = test_enc.copy()
test_enc_with_label.insert(0, label_col, np.zeros(len(test_enc), dtype=np.float32))
test_enc_with_label.to_parquet(out_dir / 'test.parquet', index=False)

print("✔ 최종 저장 완료:", out_dir)

# ======================
# 9) feature_map.json (LabelEncoder 방식에 맞춰 vocab_size=클래스 개수, padding_idx 생략)
#     - feature_map에는 seq를 포함하지 않음
# ======================
features_list = []

# 카테고리 피처 메타
for c in cat_cols:
    vocab_size = int(len(cat_encoders[c].classes_))
    features_list.append({
        c: {
            "source": "",
            "type": "categorical",
            "vocab_size": vocab_size
        }
    })

# 수치 피처 메타
for c in num_cols:
    features_list.append({
        c: {
            "source": "",
            "type": "numeric"
        }
    })

feature_map = {
    "dataset_id": dataset_id,
    "num_fields": len(feature_cols),     # 모델 입력 피처 수 (seq 제외)
    "input_length": len(feature_cols),
    "labels": [label_col],
    "features": features_list
    # total_features(=임베딩 총크기)는 LabelEncoder 방식에선 꼭 필요하진 않아서 생략
}

with open(out_dir / 'feature_map.json', 'w', encoding='utf-8') as f:
    json.dump(feature_map, f, ensure_ascii=False, indent=4)

print("✔ feature_map.json 저장 완료:", out_dir / 'feature_map.json')
print("   num_fields =", feature_map['num_fields'])

# ======================
# 10) Checks
# ======================
print("\n[CHECK] 라벨 분포")
print("train mean:", float(train_enc[label_col].mean()),
      " valid mean:", float(valid_enc[label_col].mean()))
print("train value_counts:\n", train_enc[label_col].value_counts())
print("valid value_counts:\n", valid_enc[label_col].value_counts())

print("\n[CHECK] Encoded dtypes(head)")
print(train_enc.head(1).dtypes.to_string())

print("\n[CHECK] Head with seq preserved")
print(train_enc[[label_col] + cat_cols + num_cols + [seq_col]].head(2))