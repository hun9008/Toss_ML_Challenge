# =========================================================================
# Copyright (C) 2024. The FuxiCTR Library. All rights reserved.
# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================


from sklearn.metrics import roc_auc_score, log_loss, accuracy_score, average_precision_score
import numpy as np
import pandas as pd
import multiprocessing as mp
from collections import OrderedDict

def weighted_logloss(y_true, y_pred):
    """WLL: pos/neg 기여를 50:50로 맞춘 가중 LogLoss"""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n_pos = max((y_true == 1).sum(), 1)
    n_neg = max((y_true == 0).sum(), 1)
    w_pos = 0.5 / n_pos
    w_neg = 0.5 / n_neg
    sw = np.where(y_true == 1, w_pos, w_neg)
    # sklearn log_loss는 sample_weight 지원
    return log_loss(y_true, y_pred, sample_weight=sw, eps=1e-7)

# def evaluate_metrics(y_true, y_pred, metrics, group_id=None):
#     return_dict = OrderedDict()
#     group_metrics = []
#     for metric in metrics:
#         if metric in ['logloss', 'binary_crossentropy']:
#             return_dict[metric] = log_loss(y_true, y_pred, eps=1e-7)
#         elif metric == 'AUC':
#             return_dict[metric] = roc_auc_score(y_true, y_pred)
#         elif metric in ["gAUC", "avgAUC", "MRR"] or metric.startswith("NDCG"):
#             return_dict[metric] = 0
#             group_metrics.append(metric)
#         else:
#             raise ValueError("metric={} not supported.".format(metric))
#     if len(group_metrics) > 0:
#         assert group_id is not None, "group_index is required."
#         metric_funcs = []
#         for metric in group_metrics:
#             try:
#                 metric_funcs.append(eval(metric))
#             except:
#                 raise NotImplementedError('metrics={} not implemented.'.format(metric))
#         score_df = pd.DataFrame({"group_index": group_id,
#                                  "y_true": y_true,
#                                  "y_pred": y_pred})
#         results = []
#         pool = mp.Pool(processes=mp.cpu_count() // 2)
#         for idx, df in score_df.groupby("group_index"):
#             results.append(pool.apply_async(evaluate_block, args=(df, metric_funcs)))
#         pool.close()
#         pool.join()
#         results = [res.get() for res in results]
#         sum_results = np.array(results).sum(0)
#         average_result = list(sum_results[:, 0] / sum_results[:, 1])
#         return_dict.update(dict(zip(group_metrics, average_result)))
#     return return_dict
def evaluate_metrics(y_true, y_pred, metrics, group_id=None):
    # 안전장치: 타입/범위 정리
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(float)
    # logloss/확률 기반 지표의 수치 안정성 확보
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)

    return_dict = OrderedDict()
    group_metrics = []

    for metric in metrics:
        # 표기 변형 허용 (대소문자/별칭)
        m = str(metric).strip()
        m_lower = m.lower()

        if m_lower in ['logloss', 'binary_crossentropy']:
            return_dict['logloss'] = log_loss(y_true, y_pred, eps=1e-7)

        elif m_lower == 'wll' or m_lower == 'weighted_logloss':
            # 50:50 클래스 기여의 가중 로그로스
            return_dict['WLL'] = weighted_logloss(y_true, y_pred)

        elif m_lower in ['ap', 'average_precision']:
            # 평균정밀도: 단일 클래스면 0.0으로 처리
            try:
                return_dict['AP'] = average_precision_score(y_true, y_pred)
            except ValueError:
                return_dict['AP'] = 0.0

        elif m == 'AUC':
            # 단일 클래스면 예외가 나므로 방어
            try:
                return_dict['AUC'] = roc_auc_score(y_true, y_pred)
            except ValueError:
                # 클래스가 하나뿐이면 AUC 정의 불가 → 0.5로 관례적 처리
                return_dict['AUC'] = 0.5

        elif m in ["gAUC", "avgAUC", "MRR"] or m.startswith("NDCG"):
            # 그룹 지표는 아래 블록에서 한 번에 계산
            return_dict[m] = 0
            group_metrics.append(m)

        else:
            raise ValueError(f"metric={metric} not supported.")

    # ----- 그룹 지표 처리(원본 로직 유지) -----
    if len(group_metrics) > 0:
        assert group_id is not None, "group_index is required."
        metric_funcs = []
        for metric in group_metrics:
            try:
                metric_funcs.append(eval(metric))
            except:
                raise NotImplementedError(f'metrics={metric} not implemented.')
        score_df = pd.DataFrame({"group_index": group_id,
                                 "y_true": y_true,
                                 "y_pred": y_pred})
        results = []
        pool = mp.Pool(processes=max(mp.cpu_count() // 2, 1))
        for _, df in score_df.groupby("group_index"):
            results.append(pool.apply_async(evaluate_block, args=(df, metric_funcs)))
        pool.close()
        pool.join()
        results = [res.get() for res in results]
        sum_results = np.array(results).sum(0)
        average_result = list(sum_results[:, 0] / (sum_results[:, 1] + 1e-12))
        return_dict.update(dict(zip(group_metrics, average_result)))

    return return_dict

def evaluate_block(df, metric_funcs):
    res_list = []
    for fn in metric_funcs:
        v = fn(df.y_true.values, df.y_pred.values)
        if type(v) == tuple:
            res_list.append(v)
        else: # add group weight
            res_list.append((v, 1))
    return res_list

def avgAUC(y_true, y_pred):
    """ avgAUC used in MIND news recommendation """
    if np.sum(y_true) > 0 and np.sum(y_true) < len(y_true):
        auc = roc_auc_score(y_true, y_pred)
        return (auc, 1)
    else: # in case all negatives or all positives for a group
        return (0, 0)

def gAUC(y_true, y_pred):
    """ gAUC defined in DIN paper """
    if np.sum(y_true) > 0 and np.sum(y_true) < len(y_true):
        auc = roc_auc_score(y_true, y_pred)
        n_samples = len(y_true)
        return (auc * n_samples, n_samples)
    else: # in case all negatives or all positives for a group
        return (0, 0)

def MRR(y_true, y_pred):
    order = np.argsort(y_pred)[::-1]
    y_true = np.take(y_true, order)
    rr_score = y_true / (np.arange(len(y_true)) + 1)
    mrr = np.sum(rr_score) / (np.sum(y_true) + 1e-12)
    return mrr


class NDCG(object):
    """Normalized discounted cumulative gain metric."""
    def __init__(self, k=1):
        self.topk = k

    def dcg_score(self, y_true, y_pred):
        order = np.argsort(y_pred)[::-1]
        y_true = np.take(y_true, order[:self.topk])
        gains = 2 ** y_true - 1
        discounts = np.log2(np.arange(len(y_true)) + 2)
        return np.sum(gains / discounts)

    def __call__(self, y_true, y_pred):
        idcg = self.dcg_score(y_true, y_true)
        dcg = self.dcg_score(y_true, y_pred)
        return dcg / (idcg + 1e-12)


