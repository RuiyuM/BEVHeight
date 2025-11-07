# -*- coding: utf-8 -*-
# active_learning/methods/binEnt30_permutations.py
# 六种顺序的一体化实现（ABC/ACB/BCA/BAC/CAB/CBA）
# 更新：Stage-C 取消 frac_C1_global 配比，改为
# “全局NLL从小到大 + 等配额下 per-class 最不相似(最大 NLL) 贪心”

from __future__ import annotations
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass
import numpy as np
import torch

# ===== 类别映射（按你的数据集调整） =====
CAR_IDS = {0, 3}        # car + bus -> Car
PED_IDS = {8}           # pedestrian
CYC_IDS = {6, 7}        # bicycle + motorcycle -> Cyclist
CLS_ORDER = ("car", "ped", "cyc")

try:
    from active_learning.base import BaseActiveSelector
except Exception:
    class BaseActiveSelector(object):
        def __init__(self, **kwargs): pass

# -------------------- 基础工具 --------------------
def _move_to_device(sweep_imgs, mats, device):
    if torch.is_tensor(sweep_imgs):
        sweep_imgs = sweep_imgs.to(device, non_blocking=True)
    if isinstance(mats, dict):
        for k, v in mats.items():
            if torch.is_tensor(v):
                mats[k] = v.to(device, non_blocking=True)
    return sweep_imgs, mats

def _to_numpy(a):
    if a is None: return None
    if torch.is_tensor(a): return a.detach().cpu().numpy()
    try:
        return np.asarray(a)
    except Exception:
        return None

def _extract_H_img_and_mdepth(backbone_out, expect_profile: bool):
    """
    兼容返回格式：
    - (feat,)                                    -> 无 H_img / 无 m_depth
    - (feat, H_img)                              -> 仅 H_img
    - (feat, H_img, m_depth_img)                 -> H_img + 深度覆盖（[B,D]）
    """
    if not isinstance(backbone_out, (tuple, list)) or len(backbone_out) < 1:
        raise RuntimeError("Backbone must return tuple with feat; when return_bin_entropy=True return (..., H_img[, m_depth_img]).")
    if len(backbone_out) == 1:
        return None, None
    if len(backbone_out) == 2:
        H_img = backbone_out[-1]
        if not torch.is_tensor(H_img):
            raise RuntimeError("H_img must be a torch.Tensor.")
        return H_img.view(-1).detach().cpu().float(), None
    # len >= 3
    H_img = backbone_out[-2]
    m_depth = backbone_out[-1]
    if not torch.is_tensor(H_img):
        raise RuntimeError("H_img must be a torch.Tensor.")
    H_img = H_img.view(-1).detach().cpu().float()
    if expect_profile:
        if (m_depth is None) or (not torch.is_tensor(m_depth)):
            raise RuntimeError("Depth profile (m_depth_img) must be returned when return_depth_profile=True.")
        m_depth = m_depth.detach().cpu().float()  # [B, D]
    else:
        m_depth = None
    return H_img, m_depth

def _per_image_counts_from_labels(labels) -> Tuple[int,int,int]:
    c=p=y=0
    if labels is None: return c,p,y
    lab = labels.detach().cpu().numpy().astype(np.int64) if torch.is_tensor(labels) \
          else np.asarray(labels, np.int64)
    c = int(sum(np.sum(lab == i) for i in CAR_IDS))
    p = int(sum(np.sum(lab == i) for i in PED_IDS))
    y = int(sum(np.sum(lab == i) for i in CYC_IDS))
    return c,p,y

def _per_image_diversity(c:int,p:int,y:int, alpha:float=0.5) -> float:
    n = c+p+y
    if n <= 0: return 0.0
    f = np.array([c+alpha, p+alpha, y+alpha], np.float64)
    f = f / f.sum()
    H = -np.sum(f * np.log(np.clip(f, 1e-8, 1.0)))
    return float(H * np.log1p(n))

def _xyzh_from_bboxes_np(b: np.ndarray) -> np.ndarray:
    if b is None or b.ndim != 2 or b.shape[0] == 0:
        return np.empty((0,4), np.float32)
    x = b[:, 0:1]; y = b[:, 1:2]
    z = b[:, 2:3] if b.shape[1] >= 3 else np.zeros_like(x)
    h = b[:, 5:6] if b.shape[1] >= 6 else z.copy()
    return np.concatenate([x.astype(np.float32), y.astype(np.float32), z.astype(np.float32), h.astype(np.float32)], axis=1)

def _avg_nll_weighted(xyh: np.ndarray, mu: np.ndarray, cov_inv: np.ndarray, logdet: float,
                      empty_nll: float, w: Optional[np.ndarray] = None) -> float:
    if xyh.size == 0:
        return float(empty_nll)
    xyh3 = xyh[:, :3]
    diff = xyh3 - mu[None,:]
    m = np.einsum('ni,ij,nj->n', diff, cov_inv, diff)
    if w is not None and w.size == m.size and float(w.sum()) > 1e-6:
        m = float((m * w).sum() / w.sum())
    else:
        m = float(m.mean())
    D = 3
    return 0.5 * (m + logdet + D * np.log(2*np.pi))

def _normalize_scores(scores: Optional[np.ndarray], score_thr: float, as_weight: bool, norm_to01: bool) -> np.ndarray:
    if scores is None or scores.size == 0:
        return np.zeros((0,), np.float32) if as_weight else np.ones((0,), np.float32)
    s = scores.astype(np.float32, copy=False)
    if score_thr > 0:
        keep = s >= score_thr
        s = s[keep]
    if not as_weight:
        return np.ones_like(s, dtype=np.float32)
    if norm_to01:
        lo, hi = float(s.min(initial=0.0)), float(s.max(initial=1.0))
        if hi > lo:
            s = (s - lo) / (hi - lo)
        else:
            s = np.ones_like(s, dtype=np.float32)
    return s

# -------------------- 记录结构 --------------------
@dataclass
class Rec:
    pool_idx: int
    H_img: float
    c: int; p: int; y: int
    d_img: float
    q: float
    # Global NLL 用
    xyh_all: np.ndarray
    w_all: Optional[np.ndarray]
    # Class-wise 用
    xyh_car: np.ndarray
    w_car: Optional[np.ndarray]
    xyh_ped: np.ndarray
    w_ped: Optional[np.ndarray]
    xyh_cyc: np.ndarray
    w_cyc: Optional[np.ndarray]
    # Stage-A(1)：深度覆盖向量（D=90），已 L1 归一
    m_depth: Optional[np.ndarray] = None

# =============================================================
# 选择器（可变顺序）
# =============================================================
class _PermBaseSelector(BaseActiveSelector):
    """
    可配置顺序的 Learnability 选择器：
    - Stage A：低熵门控 + 深度覆盖均衡（Δ log-coverage 贪心）
    - Stage B：per-image diversity 门控 + 全局类均衡（log-ratio 贪心）
    - Stage C：按 Global NLL 从小到大遍历 + 等配额下 per-class 最不相似（最大 NLL）贪心
    """
    def __init__(
        self,
        # ------- Stage-A 参数 -------
        frac_A_keep_total: float = 0.30,   # A 目标规模（相对 N）
        frac_A0_expand: float = 1.5,       # 低熵候选放大系数
        a1_eps: float = 1e-6,              # A-1 Δlog-coverage 的 eps
        a1_tie_reliability_weight: float = 1e-6,

        # ------- Stage-B 参数 -------
        frac_B1_gate_of_input: float = 0.50,   # 相对“当前输入”的比例
        frac_B2_final_total: float = 0.10,     # B 目标规模（相对 N）
        dirichlet_alpha: float = 0.5,
        tie_quality_weight: float = 0.1,

        # ------- Stage-C 参数 -------
        min_objs_per_class: int = 1,

        # ------- 公共统计超参 -------
        empty_nll: float = 1e-6 + 1e6,
        gauss_eps: float = 1e-3,
        score_thr: float = 0.0,
        score_weight: bool = True,
        score_norm: bool = True,

        # ------- 顺序 -------
        stage_order: str = "ABC",          # 如 "ACB"/"BCA"/"BAC"/"CAB"/"CBA"

        # ------- 兼容（忽略但不报错） -------
        frac_C1_global: float = 0.80,      # 已弃用（ignored）
        **kwargs
    ):
        super().__init__(**kwargs)
        assert 0.0 < frac_A_keep_total < 0.9
        assert frac_A0_expand >= 1.0
        assert 0.0 < frac_B1_gate_of_input < 1.0
        assert 0.0 < frac_B2_final_total < 0.9
        assert min_objs_per_class >= 1
        assert set(stage_order) == {"A","B","C"} and len(stage_order) == 3

        # A
        self.frac_A_keep_total = float(frac_A_keep_total)
        self.frac_A0_expand = float(frac_A0_expand)
        self.a1_eps = float(a1_eps)
        self.a1_tie_rel = float(a1_tie_reliability_weight)

        # B
        self.frac_B1_gate_of_input = float(frac_B1_gate_of_input)
        self.frac_B2_final_total = float(frac_B2_final_total)
        self.alpha = float(dirichlet_alpha)
        self.w_q = float(tie_quality_weight)

        # C
        self.min_objs_per_class = int(min_objs_per_class)

        # Stats / weights
        self.empty_nll = float(empty_nll)
        self.gauss_eps = float(gauss_eps)
        self.score_thr = float(score_thr)
        self.score_weight = bool(score_weight)
        self.score_norm = bool(score_norm)

        self.stage_order = stage_order

    # --------- 前向：backbone -> head -> get_bboxes ---------
    @torch.no_grad()
    def _decode_batch(self, model, sweep_imgs, mats, img_metas, device,
                      need_entropy: bool, need_depth_profile: bool):
        sweep_imgs, mats = _move_to_device(sweep_imgs, mats, device)
        out = model.model.backbone(
            sweep_imgs, mats,
            is_return_height=False,
            return_bin_entropy=need_entropy,
            return_depth_profile=need_depth_profile,
            depth_profile_mode="hard",
            depth_profile_weight="none",
        )
        H_img, m_depth = _extract_H_img_and_mdepth(out, expect_profile=need_depth_profile)
        feat = out[0] if isinstance(out, (tuple, list)) else out
        preds = model.model.head(feat)
        results = model.model.get_bboxes(preds, img_metas)
        return H_img, m_depth, results

    # --------- 从 labeled GT 拟合：全局高斯 + 类条件高斯 ---------
    @torch.no_grad()
    def _fit_global_and_classwise_from_labeled(self, labeled_loader):
        X_all = []
        feats = { "car": [], "ped": [], "cyc": [] }

        if labeled_loader is not None:
            for batch in labeled_loader:
                if len(batch) < 6:  # (sweep_imgs, mats, _, img_metas, gt_boxes, gt_labels)
                    continue
                gt_boxes, gt_labels = batch[-2], batch[-1]
                if isinstance(gt_boxes, (list, tuple)) and isinstance(gt_labels, (list, tuple)):
                    it = zip(gt_boxes, gt_labels)
                else:
                    it = [(gt_boxes, gt_labels)]
                for gb, gl in it:
                    if gl is None: continue
                    labels = gl.detach().cpu().numpy().astype(np.int64) if torch.is_tensor(gl) \
                             else np.asarray(gl, np.int64)

                    arr = None
                    if gb is not None:
                        t = getattr(gb, "tensor", None)
                        if t is not None and torch.is_tensor(t):
                            arr = t.detach().cpu().numpy()
                        elif torch.is_tensor(gb):
                            arr = gb.detach().cpu().numpy()
                        elif isinstance(gb, np.ndarray):
                            arr = gb
                    if arr is None or arr.ndim != 2 or arr.shape[0] == 0:
                        continue
                    xyzh = _xyzh_from_bboxes_np(arr)
                    xyh  = xyzh[:, [0,1,3]]
                    if xyh.size == 0: continue
                    X_all.append(xyh)
                    m_car = np.isin(labels, list(CAR_IDS))
                    m_ped = np.isin(labels, list(PED_IDS))
                    m_cyc = np.isin(labels, list(CYC_IDS))
                    if m_car.any(): feats["car"].append(xyh[m_car])
                    if m_ped.any(): feats["ped"].append(xyh[m_ped])
                    if m_cyc.any(): feats["cyc"].append(xyh[m_cyc])

        def _fit(X_list):
            if not X_list:
                mu = np.zeros((3,), np.float32)
                cov = np.eye(3, dtype=np.float32)
            else:
                X = np.concatenate(X_list, axis=0).astype(np.float32, copy=False)
                if X.shape[0] < 2:
                    mu = X.mean(axis=0) if X.shape[0] > 0 else np.zeros((3,), np.float32)
                    cov = np.eye(3, dtype=np.float32)
                else:
                    mu = X.mean(axis=0).astype(np.float32)
                    cov = np.cov(X, rowvar=False).astype(np.float32)
                    if not np.isfinite(cov).all(): cov = np.eye(3, dtype=np.float32)
            cov = cov + self.gauss_eps * np.eye(3, dtype=np.float32)
            cov_inv = np.linalg.inv(cov)
            sign, logdet = np.linalg.slogdet(cov)
            logdet = float(logdet) if sign > 0 else 0.0
            return mu, cov_inv, logdet

        mu_g, covinv_g, logdet_g = _fit(X_all)
        stats_c = {}
        for name in CLS_ORDER:
            mu_c, covinv_c, logdet_c = _fit(feats[name])
            stats_c[name] = {"mu": mu_c, "cov_inv": covinv_c, "logdet": logdet_c}
        return (mu_g, covinv_g, logdet_g), stats_c

    # --------- 扫描未标注池 ---------
    @torch.no_grad()
    def _collect_pool_records(
        self, model, dataloader, device, unlabeled_indices: List[int],
    ) -> List[Rec]:
        model.eval(); model.model.to(device)
        recs: List[Rec] = []
        ptr = 0

        for batch in dataloader:
            if len(batch) < 4:
                raise RuntimeError("dataloader batch must provide (sweep_imgs, mats, _, img_metas, ...)")
            sweep_imgs, mats, _, img_metas = batch[:4]
            H_img, m_depth, results = self._decode_batch(
                model, sweep_imgs, mats, img_metas, device,
                need_entropy=True, need_depth_profile=True
            )
            B = int(H_img.numel())

            for j in range(B):
                if ptr + j >= len(unlabeled_indices): break
                out_j = results[j]
                if isinstance(out_j, (list, tuple)) and len(out_j) >= 3:
                    bboxes, scores, labels = out_j[0], out_j[1], out_j[2]
                else:
                    bboxes, scores, labels = out_j

                c,p,y = _per_image_counts_from_labels(labels)
                d_img = _per_image_diversity(c,p,y, self.alpha)

                b_np = _to_numpy(bboxes)
                s_np = _to_numpy(scores)
                l_np = _to_numpy(labels)

                w_all = _normalize_scores(s_np, self.score_thr, as_weight=self.score_weight, norm_to01=self.score_norm)

                if b_np is not None and b_np.ndim == 2 and b_np.shape[0] > 0:
                    xyzh = _xyzh_from_bboxes_np(b_np)
                    xyh_all = xyzh[:, [0,1,3]]
                else:
                    xyh_all = np.empty((0,3), np.float32)

                xyh_car = np.empty((0,3), np.float32); w_car=None
                xyh_ped = np.empty((0,3), np.float32); w_ped=None
                xyh_cyc = np.empty((0,3), np.float32); w_cyc=None
                if b_np is not None and b_np.ndim == 2 and b_np.shape[0] > 0 and l_np is not None:
                    lab = l_np.astype(np.int64)
                    m_car = np.isin(lab, list(CAR_IDS))
                    m_ped = np.isin(lab, list(PED_IDS))
                    m_cyc = np.isin(lab, list(CYC_IDS))
                    if m_car.any():
                        xyh_car = xyh_all[m_car]
                        w_car  = w_all[m_car] if w_all is not None and w_all.size == xyh_all.shape[0] else None
                    if m_ped.any():
                        xyh_ped = xyh_all[m_ped]
                        w_ped  = w_all[m_ped] if w_all is not None and w_all.size == xyh_all.shape[0] else None
                    if m_cyc.any():
                        xyh_cyc = xyh_all[m_cyc]
                        w_cyc  = w_all[m_cyc] if w_all is not None and w_all.size == xyh_all.shape[0] else None

                # m_depth: [B, D]
                m_depth_j = None
                if m_depth is not None:
                    v = m_depth[j].detach().cpu().numpy().astype(np.float32)
                    s = float(v.sum())
                    if s > 0:
                        v = (v / s).astype(np.float32)
                    m_depth_j = v

                recs.append(
                    Rec(pool_idx=ptr+j,
                        H_img=float(H_img[j].item()),
                        c=c, p=p, y=y, d_img=d_img, q=-float(H_img[j].item()),
                        xyh_all=xyh_all, w_all=w_all,
                        xyh_car=xyh_car, w_car=w_car,
                        xyh_ped=xyh_ped, w_ped=w_ped,
                        xyh_cyc=xyh_cyc, w_cyc=w_cyc,
                        m_depth=m_depth_j)
                )
            ptr += B

        if len(recs) != len(unlabeled_indices):
            raise RuntimeError(f"Collected {len(recs)} recs != pool {len(unlabeled_indices)}; check dataloader subset.")
        return recs

    # =================== Stage A（低熵门控 + 深度覆盖均衡） ===================
    def _stage_A(self, cand_in: List[Rec], N: int, t: int) -> List[Rec]:
        if not cand_in or t <= 0: return []
        A_keep = min(max(1, int(np.ceil(self.frac_A_keep_total * N))), len(cand_in))
        A_keep = min(A_keep, t)

        # A-0：低熵门控（放大候选）
        order_by_H = np.argsort([r.H_img for r in cand_in])  # 小在前
        A0_size = min(len(cand_in), max(A_keep, int(np.ceil(self.frac_A0_expand * A_keep))))
        idx_A0 = order_by_H[:A0_size]
        cand_A0 = [cand_in[i] for i in idx_A0]

        # A-1：Δ log-coverage 贪心（D=90）
        eps = self.a1_eps
        D = None
        for r in cand_A0:
            if r.m_depth is not None:
                D = int(len(r.m_depth)); break
        if D is None:
            return cand_A0[:A_keep]

        Z = np.zeros((D,), dtype=np.float64)  # 用 Z 计数，增益为 log(eps+Z+m) - log(eps+Z)
        avail = cand_A0[:]
        chosen: List[Rec] = []
        for _ in range(min(A_keep, len(avail))):
            best_i, best_gain = -1, -1e18
            base = eps + Z
            log_base = np.log(base)
            for i, r in enumerate(avail):
                if r is None or r.m_depth is None: continue
                m = r.m_depth.astype(np.float64, copy=False)
                gain = float(np.sum(np.log(base + m) - log_base)) - self.a1_tie_rel * float(r.H_img)
                if gain > best_gain:
                    best_gain, best_i = gain, i
            if best_i < 0: break
            rb = avail[best_i]
            chosen.append(rb)
            Z += rb.m_depth.astype(np.float64, copy=False)
            avail[best_i] = None

        if len(chosen) < A_keep:
            pool_pos = set(id(x) for x in chosen)
            for r in cand_A0:
                if len(chosen) >= A_keep: break
                if id(r) not in pool_pos:
                    chosen.append(r)
        return chosen[:A_keep]

    # =================== Stage B（多样门控 + 类均衡） ===================
    def _stage_B(self, cand_in: List[Rec], N: int, t: int, labeled_dataloader) -> List[Rec]:
        if not cand_in or t <= 0: return []
        B_keep_total = min(max(1, int(np.ceil(self.frac_B2_final_total * N))), len(cand_in))
        B_keep_total = min(B_keep_total, t)

        # B1：per-image diversity gate（相对输入的比例）
        B1_keep = max(1, int(np.ceil(self.frac_B1_gate_of_input * len(cand_in))))
        order_by_div = np.argsort([-r.d_img for r in cand_in])  # 大在前
        cand_B1 = [cand_in[i] for i in order_by_div[:B1_keep]]

        # B2：类混合 log-ratio 贪心
        seed_counts = np.zeros(3, np.float64)
        if labeled_dataloader is not None:
            for batch in labeled_dataloader:
                if len(batch) < 6: continue
                gt_labels = batch[-1]
                if isinstance(gt_labels, (list, tuple)):
                    for gl in gt_labels:
                        if gl is None: continue
                        c,p,y = _per_image_counts_from_labels(gl)
                        seed_counts += np.array([c,p,y], np.float64)
                else:
                    c,p,y = _per_image_counts_from_labels(gt_labels)
                    seed_counts += np.array([c,p,y], np.float64)

        def _log_ratio_vec(S_cnts: np.ndarray, tau: np.ndarray) -> np.ndarray:
            S_sum = S_cnts.sum()
            if S_sum <= 0:
                return -np.log(np.clip(tau, 1e-8, 1.0))
            P = S_cnts / S_sum
            return np.log(np.clip(P, 1e-8, 1.0)) - np.log(np.clip(tau, 1e-8, 1.0))

        tau_C = np.array([1/3,1/3,1/3], np.float64)

        avail = cand_B1[:]
        chosen: List[Rec] = []
        S_counts = seed_counts.astype(np.float64).copy()

        for _ in range(min(B_keep_total, len(avail))):
            best_i, best_score = -1, -1e18
            g = _log_ratio_vec(S_counts, tau_C)
            for i, r in enumerate(avail):
                if r is None: continue
                add = np.array([r.c, r.p, r.y], np.float64)
                gain = - float(g @ add) + self.w_q * r.q
                if gain > best_score:
                    best_score, best_i = gain, i
            if best_i < 0: break
            rb = avail[best_i]
            chosen.append(rb)
            S_counts += np.array([rb.c, rb.p, rb.y], np.float64)
            avail[best_i] = None

        if len(chosen) < min(B_keep_total, len(cand_B1)):
            pool_pos = set(id(x) for x in chosen)
            for r in cand_B1:
                if len(chosen) >= B_keep_total: break
                if id(r) not in pool_pos:
                    chosen.append(r)

        return chosen[:B_keep_total] if chosen else cand_B1[:min(B_keep_total, len(cand_B1))]

    # =================== Stage C（新版）===================
    # 统一：按 global NLL 从小到大遍历；在等配额约束下，贪心选“per-class 最不相似(最大 NLL)”
    def _stage_C(self, cand_in: List[Rec], t: int,
                 mu_g, covinv_g, logdet_g, stats_c) -> List[Rec]:
        if not cand_in or t <= 0: return []
        t = min(t, len(cand_in))
        M = len(cand_in)

        # 1) Global NLL 排序（小->大）：锚定到已学几何流形
        nll_global = np.asarray([
            _avg_nll_weighted(r.xyh_all, mu_g, covinv_g, logdet_g, self.empty_nll, r.w_all)
            for r in cand_in
        ], dtype=np.float64)
        order = np.argsort(nll_global)  # 0..M-1（小->大）

        # 2) 预计算 per-class NLL（最不相似 = NLL 大；无该类 -> -inf）
        S = np.full((3, M), -np.inf, dtype=np.float64)
        for j in range(M):
            r = cand_in[j]
            if r.xyh_car.shape[0] >= self.min_objs_per_class:
                mu = stats_c["car"]["mu"]; covinv = stats_c["car"]["cov_inv"]; logdet = stats_c["car"]["logdet"]
                S[0, j] = _avg_nll_weighted(r.xyh_car, mu, covinv, logdet, self.empty_nll, r.w_car)
            if r.xyh_ped.shape[0] >= self.min_objs_per_class:
                mu = stats_c["ped"]["mu"]; covinv = stats_c["ped"]["cov_inv"]; logdet = stats_c["ped"]["logdet"]
                S[1, j] = _avg_nll_weighted(r.xyh_ped, mu, covinv, logdet, self.empty_nll, r.w_ped)
            if r.xyh_cyc.shape[0] >= self.min_objs_per_class:
                mu = stats_c["cyc"]["mu"]; covinv = stats_c["cyc"]["cov_inv"]; logdet = stats_c["cyc"]["logdet"]
                S[2, j] = _avg_nll_weighted(r.xyh_cyc, mu, covinv, logdet, self.empty_nll, r.w_cyc)

        # 3) 等配额（每类相同 weight）
        quotas = np.array([t // 3, t // 3, t // 3], dtype=np.int64)
        for qfill in range(t % 3): quotas[qfill] += 1  # car, ped, cyc 顺序补余
        taken = np.zeros(3, dtype=np.int64)

        chosen_rel: List[int] = []
        used = np.zeros(M, dtype=bool)

        # —— 可选：确保第一个锚点来自全局最相似样本（若其对某类有贡献）
        seed_idx = int(order[0])
        if t > 0:
            best_c = -1; best_val = -np.inf
            for c in range(3):
                if quotas[c] <= 0: continue
                val = S[c, seed_idx]
                if np.isfinite(val) and val > best_val:
                    best_val = val; best_c = c
            if best_c >= 0:
                chosen_rel.append(seed_idx)
                used[seed_idx] = True
                taken[best_c] += 1

        # 4) 主循环：按 global NLL 升序扫描；若样本能为某个“尚有配额”的类提供最大增益，则选取
        for j in order:
            if len(chosen_rel) >= t: break
            if used[j]: continue
            best_c = -1; best_val = -np.inf
            for c in range(3):
                if taken[c] >= quotas[c]: continue
                val = S[c, j]
                if np.isfinite(val) and val > best_val:
                    best_val = val; best_c = c
            if best_c >= 0:
                chosen_rel.append(j)
                used[j] = True
                taken[best_c] += 1

        # 5) 若仍未满：用 union 分数（max_c S[c,*]）从剩余样本补齐（仍优先几何新颖）
        if len(chosen_rel) < t:
            union_scores = S.max(axis=0)
            union_scores[used] = -np.inf
            fill_order = np.argsort(-union_scores)
            for j in fill_order:
                if len(chosen_rel) >= t: break
                if not np.isfinite(union_scores[j]): continue
                chosen_rel.append(j); used[j] = True

        # 6) 兜底：用 global NLL 最小的剩余补足（保证锚定/稳定）
        if len(chosen_rel) < t:
            for j in order:
                if len(chosen_rel) >= t: break
                if not used[j]:
                    chosen_rel.append(j); used[j] = True

        return [cand_in[i] for i in chosen_rel[:t]]

    # =================== 目标规模计算（保证非增并以 k 收尾） ===================
    def _compute_stage_targets(self, N: int, k: int, order: str) -> Dict[str, int]:
        base = {
            "A": max(1, int(np.ceil(self.frac_A_keep_total * N))),
            "B": max(1, int(np.ceil(self.frac_B2_final_total * N))),
            "C": max(1, int(k)),
        }
        s1, s2, s3 = order[0], order[1], order[2]
        t1 = max(base[s1], base[s2], base[s3])
        t2 = max(base[s2], base[s3])
        t3 = base[s3]
        return {s1: t1, s2: t2, s3: t3}

    # =================== 主流程 ===================
    @torch.no_grad()
    def select(
        self,
        model: torch.nn.Module,
        dataloader,                    # unlabeled dataloader（需含 img_metas）
        device: torch.device,
        unlabeled_indices: List[int],
        k: int,
        labeled_dataloader=None,       # 已标注 dataloader（含 gt_boxes/gt_labels）
    ) -> List[int]:

        N = len(unlabeled_indices)
        if N == 0 or k <= 0:
            return []

        # 1) 全池统计
        recs = self._collect_pool_records(model, dataloader, device, unlabeled_indices)
        (mu_g, covinv_g, logdet_g), stats_c = self._fit_global_and_classwise_from_labeled(labeled_dataloader)

        # 2) 计算三个阶段的目标规模（随顺序自适应）
        targets = self._compute_stage_targets(N, k, self.stage_order)

        # 3) 依顺序串联
        stage_map = {
            "A": lambda cand, t: self._stage_A(cand, N=N, t=t),
            "B": lambda cand, t: self._stage_B(cand, N=N, t=t, labeled_dataloader=labeled_dataloader),
            "C": lambda cand, t: self._stage_C(cand, t=t, mu_g=mu_g, covinv_g=covinv_g, logdet_g=logdet_g, stats_c=stats_c),
        }

        cand: List[Rec] = recs
        for s in self.stage_order:
            t = targets[s]
            if not cand: break
            cand = stage_map[s](cand, t)

        if not cand:
            return []

        chosen_final_pool_idx = [r.pool_idx for r in cand[:min(k, len(cand))]]
        return [unlabeled_indices[i] for i in chosen_final_pool_idx]


# =============================================================
# 六个顺序的具体类（方便在配置里直接切换）
# =============================================================
class BinEnt30_Perm_ABC(_PermBaseSelector):
    def __init__(self, **kw):
        super().__init__(stage_order="ABC", **kw)

class BinEnt30_Perm_ACB(_PermBaseSelector):
    def __init__(self, **kw):
        super().__init__(stage_order="ACB", **kw)

class BinEnt30_Perm_BCA(_PermBaseSelector):
    def __init__(self, **kw):
        super().__init__(stage_order="BCA", **kw)

class BinEnt30_Perm_BAC(_PermBaseSelector):
    def __init__(self, **kw):
        super().__init__(stage_order="BAC", **kw)

class BinEnt30_Perm_CAB(_PermBaseSelector):
    def __init__(self, **kw):
        super().__init__(stage_order="CAB", **kw)

class BinEnt30_Perm_CBA(_PermBaseSelector):
    def __init__(self, **kw):
        super().__init__(stage_order="CBA", **kw)

# ===== 注册到 REGISTRY =====
try:
    from active_learning.methods import _METHOD_REGISTRY
    _METHOD_REGISTRY["binEnt30_perm_ABC"] = BinEnt30_Perm_ABC
    _METHOD_REGISTRY["binEnt30_perm_ACB"] = BinEnt30_Perm_ACB
    _METHOD_REGISTRY["binEnt30_perm_BCA"] = BinEnt30_Perm_BCA
    _METHOD_REGISTRY["binEnt30_perm_BAC"] = BinEnt30_Perm_BAC
    _METHOD_REGISTRY["binEnt30_perm_CAB"] = BinEnt30_Perm_CAB
    _METHOD_REGISTRY["binEnt30_perm_CBA"] = BinEnt30_Perm_CBA

    # 向后兼容：把原名映射到 ABC（经典顺序）
    _METHOD_REGISTRY["binEnt30_twoTier_global_then_classanti_adapt"] = BinEnt30_Perm_ABC
except Exception:
    pass
