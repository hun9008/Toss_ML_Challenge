import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

# -----------------------
# 0) 경로/설정
# -----------------------
in_dir  = Path("./raw")     # 원본 train/test 위치
out_dir = Path("./")        # 최종 .parquet 저장 위치
out_dir.mkdir(parents=True, exist_ok=True)

train_path = in_dir / "raw_train.parquet"
test_path  = in_dir / "raw_test.parquet"

target_col = "clicked"
seq_col    = "seq"
FEATURE_EXCLUDE = {target_col, seq_col, "ID"}

# 이 모델이 기대하는 범주형 고정 셋
cat_cols_fixed = ["gender","age_group","inventory_id","l_feat_14"]

# -----------------------
# 1) 로드
# -----------------------
train = pd.read_parquet(train_path, engine="pyarrow")
test  = pd.read_parquet(test_path,  engine="pyarrow")

# -----------------------
# 2) 기본 클린업
# -----------------------
# ID 제거(샘플 식별자라면)
for df in (train, test):
    if "ID" in df.columns:
        df.drop(columns=["ID"], inplace=True, errors="ignore")

# seq 보정: 존재/문자열화/NaN -> ""
def ensure_seq_string(df: pd.DataFrame, seq_col: str):
    if seq_col not in df.columns:
        # 없으면 빈 문자열로 채운 새 컬럼 생성
        df[seq_col] = ""
    else:
        df[seq_col] = df[seq_col].astype(str).fillna("")
    return df

train = ensure_seq_string(train, seq_col)
test  = ensure_seq_string(test,  seq_col)

# -----------------------
# 3) 피처 열 결정
# -----------------------
all_cols = [c for c in train.columns if c not in FEATURE_EXCLUDE]  # train 기준
# cat은 고정 셋, num은 나머지
cat_cols = [c for c in cat_cols_fixed if c in all_cols]
num_cols = [c for c in all_cols if c not in cat_cols]

# -----------------------
# 4) 카테고리 라벨인코딩 (train+test 합쳐서)
# -----------------------
encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    # train+test 합쳐서 fit, NaN -> "UNK"
    all_vals = pd.concat([train[col], test[col]], axis=0)
    all_vals = all_vals.astype(str).fillna("UNK")
    le.fit(all_vals)
    # transform
    train[col] = le.transform(train[col].astype(str).fillna("UNK")).astype(np.int32)
    test[col]  = le.transform(test[col].astype(str).fillna("UNK")).astype(np.int32)
    encoders[col] = le

# -----------------------
# 5) 수치형 처리 (no scaling)
# -----------------------
for col in num_cols:
    # 결측 0, float32
    train[col] = pd.to_numeric(train[col], errors="coerce").fillna(0).astype(np.float32)
    test[col]  = pd.to_numeric(test[col],  errors="coerce").fillna(0).astype(np.float32)

# -----------------------
# 6) 라벨/타입 정리
# -----------------------
assert target_col in train.columns, f"{target_col} not in train!"
train[target_col] = pd.to_numeric(train[target_col], errors="coerce").fillna(0).astype(np.float32)

# 컬럼 순서: train -> [clicked] + [num_cols + cat_cols + seq]
train_cols_order = [target_col] + num_cols + cat_cols + [seq_col]
test_cols_order  = num_cols + cat_cols + [seq_col]

# 누락 열(드물게 있을 수 있음) 0/"" 채움
def ensure_columns(df, cols, num_cols, cat_cols, seq_col, has_target):
    for c in cols:
        if c not in df.columns:
            if c == seq_col:
                df[c] = ""
            elif c in num_cols:
                df[c] = np.float32(0.0)
            elif c in cat_cols:
                df[c] = np.int32(0)
            elif has_target and c == target_col:
                df[c] = np.float32(0.0)
    return df[cols]

train_out = ensure_columns(train, train_cols_order, num_cols, cat_cols, seq_col, has_target=True)
test_out  = ensure_columns(test,  test_cols_order,  num_cols, cat_cols, seq_col, has_target=False)

# -----------------------
# 7) 저장
# -----------------------
train_out.to_parquet(out_dir / "train.parquet", index=False)
test_out.to_parquet(out_dir / "test.parquet",  index=False)

print("✔ Saved:", out_dir / "train.parquet")
print("✔ Saved:", out_dir / "test.parquet")

# -----------------------
# 8) 메타 저장(재현용, 선택)
# -----------------------
meta = {
    "cat_cols": cat_cols,
    "num_cols": num_cols,
    "seq_col": seq_col,
    "target_col": target_col,
    "encoders": {
        c: encoders[c].classes_.tolist() for c in encoders
    },
    "dtypes": {
        "train": {c: str(train_out[c].dtype) for c in train_out.columns},
        "test":  {c: str(test_out[c].dtype)  for c in test_out.columns},
    }
}
import json
with open(out_dir / "ctr_preproc_meta.json", "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)
print("✔ Saved meta:", out_dir / "ctr_preproc_meta.json")

# -----------------------
# 9) feature_map.json 생성
# -----------------------
feature_cols = num_cols + cat_cols  # label/seq 제외
features_list = []

# categorical: vocab_size = LabelEncoder classes_ 개수
for c in feature_cols:
    if c in cat_cols:
        vocab_size = len(encoders[c].classes_)
        features_list.append({
            c: {
                "source": "",
                "type": "categorical",
                "padding_idx": 0,           # 0을 패딩/UNK로 쓰려면 인코딩 단계에서 0 예약이 필요함(현재는 0~(K-1) 사용)
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

# total_features = 모든 범주형 vocab_size의 합
total_features = int(sum(
    int(list(f.values())[0].get("vocab_size", 0)) for f in features_list
))

feature_map = {
    "dataset_id": "toss_ctr_v1",             # 필요시 dataset_id 변수로 교체
    "num_fields": len(feature_cols),
    "total_features": total_features,
    "input_length": len(feature_cols),
    "labels": [target_col],
    "features": features_list
}

with open(out_dir / "feature_map.json", "w", encoding="utf-8") as f:
    json.dump(feature_map, f, ensure_ascii=False, indent=4)

print("✔ feature_map.json 저장 완료:", out_dir / "feature_map.json")
print("   num_fields =", feature_map["num_fields"],
      " total_features =", feature_map["total_features"])