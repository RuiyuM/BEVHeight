# active_learning/methods/binEnt30_twoTier_global_then_classanti_adapt_stage12.py
# Stage-12：仅保留 Stage-A + Stage-B（删除 Stage-C）
# A = 低熵门控（放大系数） + 深度覆盖均衡（log-ratio 贪心，D=90）
# B = 两层 diversity（B2=log-ratio 贪心，最终若干比例/至少k）
# 输出：直接返回 Stage-B 的结果（前 k 个）

from __future__ import annotations
from typing import List, Optional, Tuple
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
        def __init__(self, **kwargs): 
            pass

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
    if a is None:
        return None
    if torch.is_tensor(a):
        return a.detach().cpu().numpy()
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

def _per_image_counts_from_labels(labels) -> Tuple[int, int, int]:
    c = p = y = 0
    if labels is None:
        return c, p, y
    lab = labels.detach().cpu().numpy().astype(np.int64) if torch.is_tensor(labels) else np.asarray(labels, np.int64)
    c = int(sum(np.sum(lab == i) for i in CAR_IDS))
    p = int(sum(np.sum(lab == i) for i in PED_IDS))
    y = int(sum(np.sum(lab == i) for i in CYC_IDS))
    return c, p, y

def _per_image_diversity(c: int, p: int, y: int, alpha: float = 0.5) -> float:
    n = c + p + y
    if n <= 0:
        return 0.0
    f = np.array([c + alpha, p + alpha, y + alpha], np.float64)
    f = f / f.sum()
    H = -np.sum(f * np.log(np.clip(f, 1e-8, 1.0)))
    return float(H * np.log1p(n))

def _xyzh_from_bboxes_np(b: np.ndarray) -> np.ndarray:
    if b is None or b.ndim != 2 or b.shape[0] == 0:
        return np.empty((0, 4), np.float32)
    x = b[:, 0:1]; y = b[:, 1:2]
    z = b[:, 2:3] if b.shape[1] >= 3 else np.zeros_like(x)
    h = b[:, 5:6] if b.shape[1] >= 6 else z.copy()
    return np.concatenate([x.astype(np.float32), y.astype(np.float32), z.astype(np.float32), h.astype(np.float32)], axis=1)

def _avg_nll_weighted(xyh: np.ndarray, mu: np.ndarray, cov_inv: np.ndarray, logdet: float,
                      empty_nll: float, w: Optional[np.ndarray] = None) -> float:
    if xyh.size == 0:
        return float(empty_nll)
    xyh3 = xyh[:, :3]
    diff = xyh3 - mu[None, :]
    m = np.einsum('ni,ij,nj->n', diff, cov_inv, diff)
    if w is not None and w.size == m.size and float(w.sum()) > 1e-6:
        m = float((m * w).sum() / w.sum())
    else:
        m = float(m.mean())
    D = 3
    return 0.5 * (m + logdet + D * np.log(2 * np.pi))

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
        lo, hi = float(np.min(s, initial=0.0)), float(np.max(s, initial=1.0))
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
    # Global NLL 用（本变体不使用，但保留字段避免改动其他代码）
    xyh_all: np.ndarray
    w_all: Optional[np.ndarray]
    # Class-wise 用（本变体不使用）
    xyh_car: np.ndarray
    w_car: Optional[np.ndarray]
    xyh_ped: np.ndarray
    w_ped: Optional[np.ndarray]
    xyh_cyc: np.ndarray
    w_cyc: Optional[np.ndarray]
    # Stage-A(1) 用：深度覆盖向量（D=90，不做合并），已 L1 归一化
    m_depth: Optional[np.ndarray] = None

# =============================================================
# Stage-A + Stage-B（Stage-C 删除）
# A = 低熵门控（放大系数） + 深度覆盖均衡（log-ratio 贪心）
# B = 两层 diversity（B2=log-ratio 贪心）
# =============================================================
class BinEnt30_TwoTier_Stage12_GlobalThenClassAntiAdaptiveSelector(BaseActiveSelector):
    def __init__(
        self,
        # ------- Stage-A 参数（使用） -------
        frac_A_keep_total: float = 0.30,
        frac_A0_expand: float = 1.5,
        a1_eps: float = 1e-6,
        a1_tie_reliability_weight: float = 1e-6,

        # ------- Stage-B 参数（使用） -------
        frac_B1_gate_of_A: float = 0.50,
        frac_B2_final_total: float = 0.10,
        dirichlet_alpha: float = 0.5,
        tie_quality_weight: float = 0.1,

        # ------- Stage-C 参数（本变体不使用，保留兼容） -------
        frac_C1_global: float = 0.80,
        min_objs_per_class: int = 1,

        # ------- 公共统计超参 -------
        empty_nll: float = 1e-6 + 1e6,
        gauss_eps: float = 1e-3,
        score_thr: float = 0.0,
        score_weight: bool = True,
        score_norm: bool = True,

        # ------- 兼容旧参数（不再使用，仅为避免配置报错） -------
        epsilon_mode: str = "adaptive",
        epsilon_fixed: float = 0.10,
        eps_min: float = 0.05,
        eps_max: float = 0.25,
        eps_alpha_cls: float = 0.6,
        eps_ema: float = 0.7,
        geo_q_anchor: float = 0.75,
        geo_q_rest: float = 0.90,
        anti_agg: str = "max",
        class_weights: Tuple[float, float, float] = (1/3, 1/3, 1/3),
        anti_clip_percentile: float = 95.0,
        target_cls_dist: str = "uniform",
        **kwargs
    ):
        super().__init__(**kwargs)

        assert 0.0 < frac_A_keep_total < 0.9
        assert frac_A0_expand >= 1.0
        assert 0.0 < frac_B1_gate_of_A < 1.0
        assert 0.0 < frac_B2_final_total < 0.9
        # Stage-C 被删除，因此不对 C 的参数做约束

        # A
        self.frac_A_keep_total = float(frac_A_keep_total)
        self.frac_A0_expand = float(frac_A0_expand)
        self.a1_eps = float(a1_eps)
        self.a1_tie_rel = float(a1_tie_reliability_weight)

        # B
        self.frac_B1_gate_of_A = float(frac_B1_gate_of_A)
        self.frac_B2_final_total = float(frac_B2_final_total)
        self.alpha = float(dirichlet_alpha)
        self.w_q = float(tie_quality_weight)

        # C（不使用）
        self.frac_C1_global = float(frac_C1_global)
        self.min_objs_per_class = int(min_objs_per_class)

        # Stats / weights（recs 收集仍需要）
        self.empty_nll = float(empty_nll)
        self.gauss_eps = float(gauss_eps)
        self.score_thr = float(score_thr)
        self.score_weight = bool(score_weight)
        self.score_norm = bool(score_norm)

        # 兼容位（不使用）
        self._eps_prev: Optional[float] = None
        self._compat = dict(
            epsilon_mode=epsilon_mode, epsilon_fixed=epsilon_fixed, eps_min=eps_min, eps_max=eps_max,
            eps_alpha_cls=eps_alpha_cls, eps_ema=eps_ema, geo_q_anchor=geo_q_anchor, geo_q_rest=geo_q_rest,
            anti_agg=anti_agg, class_weights=class_weights, anti_clip_percentile=anti_clip_percentile,
            target_cls_dist=target_cls_dist
        )

    # --------- 前向：backbone -> head -> get_bboxes ---------
    @torch.no_grad()
    def _decode_batch(self, model, sweep_imgs, mats, img_metas, device,
                      need_entropy: bool, need_depth_profile: bool):
        sweep_imgs, mats = _move_to_device(sweep_imgs, mats, device)

        try:
            out = model.model.backbone(
                sweep_imgs, mats,
                is_return_height=False,
                return_bin_entropy=need_entropy,
                return_depth_profile=need_depth_profile,
                depth_profile_mode="hard",
                depth_profile_weight="none",
            )
        except TypeError as e:
            if need_depth_profile:
                raise RuntimeError(
                    "Backbone 不支持 return_depth_profile/depth_profile_* 参数，但本变体需要 depth profile（Stage-A）。"
                ) from e
            out = model.model.backbone(
                sweep_imgs, mats,
                is_return_height=False,
                return_bin_entropy=need_entropy,
            )

        H_img, m_depth = _extract_H_img_and_mdepth(out, expect_profile=need_depth_profile)
        feat = out[0] if isinstance(out, (tuple, list)) else out
        preds = model.model.head(feat)
        results = model.model.get_bboxes(preds, img_metas)
        return H_img, m_depth, results

    # --------- 扫描未标注池：收集 H_img + m_depth + 预测 ---------
    @torch.no_grad()
    def _collect_pool_records(
        self, model, dataloader, device, unlabeled_indices: List[int],
        need_depth_profile: bool = True,
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
                need_entropy=True,
                need_depth_profile=need_depth_profile,
            )
            B = int(H_img.numel())

            for j in range(B):
                if ptr + j >= len(unlabeled_indices):
                    break
                out_j = results[j]
                if isinstance(out_j, (list, tuple)) and len(out_j) >= 3:
                    bboxes, scores, labels = out_j[0], out_j[1], out_j[2]
                else:
                    bboxes, scores, labels = out_j

                c, p, y = _per_image_counts_from_labels(labels)
                d_img = _per_image_diversity(c, p, y, self.alpha)

                b_np = _to_numpy(bboxes)
                s_np = _to_numpy(scores)
                l_np = _to_numpy(labels)

                w_all = _normalize_scores(s_np, self.score_thr, as_weight=self.score_weight, norm_to01=self.score_norm)

                if b_np is not None and b_np.ndim == 2 and b_np.shape[0] > 0:
                    xyzh = _xyzh_from_bboxes_np(b_np)
                    xyh_all = xyzh[:, [0, 1, 3]]
                else:
                    xyh_all = np.empty((0, 3), np.float32)

                # class split（Stage-C 不用，但保留字段）
                xyh_car = np.empty((0, 3), np.float32); w_car = None
                xyh_ped = np.empty((0, 3), np.float32); w_ped = None
                xyh_cyc = np.empty((0, 3), np.float32); w_cyc = None
                if b_np is not None and b_np.ndim == 2 and b_np.shape[0] > 0 and l_np is not None:
                    lab = l_np.astype(np.int64)
                    m_car = np.isin(lab, list(CAR_IDS))
                    m_ped = np.isin(lab, list(PED_IDS))
                    m_cyc = np.isin(lab, list(CYC_IDS))
                    if m_car.any():
                        xyh_car = xyh_all[m_car]
                        w_car = w_all[m_car] if w_all is not None and w_all.size == xyh_all.shape[0] else None
                    if m_ped.any():
                        xyh_ped = xyh_all[m_ped]
                        w_ped = w_all[m_ped] if w_all is not None and w_all.size == xyh_all.shape[0] else None
                    if m_cyc.any():
                        xyh_cyc = xyh_all[m_cyc]
                        w_cyc = w_all[m_cyc] if w_all is not None and w_all.size == xyh_all.shape[0] else None

                # m_depth: [B, D]（来自 backbone），取第 j 个
                m_depth_j = None
                if m_depth is not None:
                    v = m_depth[j].detach().cpu().numpy().astype(np.float32)  # [D]
                    s = float(v.sum())
                    if s > 0:
                        v = (v / s).astype(np.float32)
                    m_depth_j = v

                recs.append(
                    Rec(
                        pool_idx=ptr + j,
                        H_img=float(H_img[j].item()),
                        c=c, p=p, y=y,
                        d_img=d_img,
                        q=-float(H_img[j].item()),
                        xyh_all=xyh_all, w_all=w_all,
                        xyh_car=xyh_car, w_car=w_car,
                        xyh_ped=xyh_ped, w_ped=w_ped,
                        xyh_cyc=xyh_cyc, w_cyc=w_cyc,
                        m_depth=m_depth_j,
                    )
                )
            ptr += B

        if len(recs) != len(unlabeled_indices):
            raise RuntimeError(f"Collected {len(recs)} recs != pool {len(unlabeled_indices)}; check dataloader subset.")
        return recs

    # =================== 主流程（Stage-12） ===================
    @torch.no_grad()
    def select(
        self,
        model: torch.nn.Module,
        dataloader,                    # unlabeled dataloader（需含 img_metas）
        device: torch.device,
        unlabeled_indices: List[int],
        k: int,
        labeled_dataloader=None,       # 已标注 dataloader（仅用于 B2 初始化计数；可为 None）
    ) -> List[int]:

        N = len(unlabeled_indices)
        if N == 0 or k <= 0:
            return []
        k = int(min(k, N))

        # ---- 扫描未标注池：H_img + m_depth + 预测 ----
        recs = self._collect_pool_records(
            model, dataloader, device, unlabeled_indices,
            need_depth_profile=True,
        )

        # ===================== Stage-A（与原版一致） =====================
        A_keep = min(N, max(k, int(np.ceil(self.frac_A_keep_total * N))))  # 至少保证 >=k
        A0_size = min(N, max(A_keep, int(np.ceil(self.frac_A0_expand * A_keep))))
        order_by_H = np.argsort([r.H_img for r in recs])  # 小在前
        idx_A0 = order_by_H[:A0_size]
        cand_A0 = [recs[i] for i in idx_A0]

        D = None
        for r in cand_A0:
            if r.m_depth is not None:
                D = int(len(r.m_depth))
                break
        if D is None:
            cand_A = cand_A0[:A_keep]
        else:
            T = np.full((D,), A_keep / float(D), dtype=np.float64)
            Z = np.full((D,), self.a1_eps, dtype=np.float64)
            avail = cand_A0[:]
            chosen_rel: List[int] = []

            for _ in range(min(A_keep, len(avail))):
                g = np.log(np.clip(Z, self.a1_eps, None)) - np.log(np.clip(T, self.a1_eps, None))
                best_i, best_val = -1, +1e18
                for i, r in enumerate(avail):
                    if r is None or r.m_depth is None:
                        continue
                    m = r.m_depth.astype(np.float64, copy=False)
                    val = float(g @ m) + self.a1_tie_rel * float(r.H_img)
                    if val < best_val:
                        best_val, best_i = val, i
                if best_i < 0:
                    break
                rb = avail[best_i]
                chosen_rel.append(rb.pool_idx)
                Z += rb.m_depth.astype(np.float64, copy=False)
                avail[best_i] = None

            pool_pos = set(chosen_rel)
            if len(chosen_rel) < A_keep:
                for r in cand_A0:
                    if len(chosen_rel) >= A_keep:
                        break
                    if r.pool_idx not in pool_pos:
                        chosen_rel.append(r.pool_idx)

            cand_A = [recs[i] for i in chosen_rel]

        if len(cand_A) == 0:
            return []

        # ===================== Stage-B（与原版一致） =====================
        B1_keep = min(len(cand_A), max(k, int(np.ceil(self.frac_B1_gate_of_A * len(cand_A)))))
        order_by_div = np.argsort([-r.d_img for r in cand_A])
        idx_B1 = order_by_div[:B1_keep]
        cand_B1 = [cand_A[i] for i in idx_B1]

        B2_keep_total = min(len(cand_B1), max(k, int(np.ceil(self.frac_B2_final_total * N))))

        seed_counts = np.zeros(3, np.float64)
        if labeled_dataloader is not None:
            for batch in labeled_dataloader:
                if len(batch) < 6:
                    continue
                gt_labels = batch[-1]
                if isinstance(gt_labels, (list, tuple)):
                    for gl in gt_labels:
                        if gl is None:
                            continue
                        c, p, y = _per_image_counts_from_labels(gl)
                        seed_counts += np.array([c, p, y], np.float64)
                else:
                    c, p, y = _per_image_counts_from_labels(gt_labels)
                    seed_counts += np.array([c, p, y], np.float64)

        S_counts = seed_counts.astype(np.float64).copy()

        def _log_ratio_vec(S_cnts: np.ndarray, tau: np.ndarray) -> np.ndarray:
            S_sum = S_cnts.sum()
            if S_sum <= 0:
                return -np.log(np.clip(tau, 1e-8, 1.0))
            P = S_cnts / S_sum
            return np.log(np.clip(P, 1e-8, 1.0)) - np.log(np.clip(tau, 1e-8, 1.0))

        tau_C = np.array([1/3, 1/3, 1/3], np.float64)

        avail = cand_B1[:]
        chosen_B_rel: List[int] = []
        for _ in range(min(B2_keep_total, len(avail))):
            best_i, best_score = -1, -1e18
            g = _log_ratio_vec(S_counts, tau_C)
            for i, r in enumerate(avail):
                if r is None:
                    continue
                add = np.array([r.c, r.p, r.y], np.float64)
                gain = - float(g @ add) + self.w_q * r.q
                if gain > best_score:
                    best_score, best_i = gain, i
            if best_i < 0:
                break
            rb = avail[best_i]
            chosen_B_rel.append(rb.pool_idx)
            S_counts += np.array([rb.c, rb.p, rb.y], np.float64)
            avail[best_i] = None

        pool_pos = set(chosen_B_rel)
        need = max(B2_keep_total, k)
        if len(chosen_B_rel) < min(need, len(cand_B1)):
            for r in cand_B1:
                if len(chosen_B_rel) >= min(need, len(cand_B1)):
                    break
                if r.pool_idx not in pool_pos:
                    chosen_B_rel.append(r.pool_idx)

        cand_B2 = [recs[i] for i in chosen_B_rel] if chosen_B_rel else cand_A[:]
        if len(cand_B2) == 0:
            return []

        # ===== Stage-C 删除：直接返回 Stage-B 的前 k 个 =====
        chosen_final_pool_idx = [r.pool_idx for r in cand_B2[:min(k, len(cand_B2))]]
        return [unlabeled_indices[i] for i in chosen_final_pool_idx]


# ===== 注册（新方法名：stage12） =====
try:
    from active_learning.methods import _METHOD_REGISTRY
    _METHOD_REGISTRY["binEnt30_twoTier_global_then_classanti_adapt_stage12"] = BinEnt30_TwoTier_Stage12_GlobalThenClassAntiAdaptiveSelector
except Exception:
    pass
