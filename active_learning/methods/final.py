# active_learning/methods/binEnt30_twoTier_then_gauss_sim.py
from __future__ import annotations
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass
import numpy as np
import torch

# ===== 类别映射（按你的数据集调整） =====
CAR_IDS = {0, 3}        # car + bus -> Car
PED_IDS = {8}           # pedestrian
CYC_IDS = {6, 7}        # bicycle + motorcycle -> Cyclist

try:
    from active_learning.base import BaseActiveSelector
except Exception:
    class BaseActiveSelector(object):
        def __init__(self, **kwargs): pass

# -------------------- 工具 --------------------
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

def _extract_H_img(backbone_out) -> torch.Tensor:
    """LSSFPN 返回 tuple，其中最后一项是 [B] 的图像级 bin-entropy。"""
    if not isinstance(backbone_out, (tuple, list)) or len(backbone_out) < 1:
        raise RuntimeError("Backbone must return (..., H_img) when return_bin_entropy=True.")
    H_img = backbone_out[-1]
    if not torch.is_tensor(H_img):
        raise RuntimeError("H_img must be a torch.Tensor.")
    return H_img.view(-1).detach().cpu().float()

def _per_image_counts_from_labels(labels) -> Tuple[int,int,int]:
    c=p=y=0
    if labels is None: return c,p,y
    lab = labels.detach().cpu().numpy().astype(np.int64) if torch.is_tensor(labels) else np.asarray(labels, np.int64)
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

def _set_entropy_from_counts(counts: np.ndarray, alpha: float=0.5) -> float:
    s = counts.astype(np.float64) + alpha
    s = s / (s.sum() if s.sum() > 0 else 1.0)
    return float(-np.sum(s * np.log(np.clip(s, 1e-8, 1.0))))

def _xyh_from_boxes_np(b: np.ndarray) -> np.ndarray:
    """
    从 **decode 后** 的 bboxes 数组提取 [x, y, h] 特征：
    - 常见布局: [x, y, z, dx, dy, dz, yaw, ...] -> h=dz 在索引 5；
    - 若维度不足，退化用 z 充当高度。
    """
    if b is None or b.ndim != 2 or b.shape[0] == 0:
        return np.empty((0,3), np.float32)
    x = b[:, 0:1]
    y = b[:, 1:2]
    if b.shape[1] >= 6:
        h = b[:, 5:6]  # dz
    elif b.shape[1] >= 3:
        h = b[:, 2:3]  # fallback: z
    else:
        return np.empty((0,3), np.float32)
    return np.concatenate([x.astype(np.float32), y.astype(np.float32), h.astype(np.float32)], axis=1)

def _avg_nll_weighted(xyh: np.ndarray, mu: np.ndarray, cov_inv: np.ndarray, logdet: float,
                      empty_nll: float, w: Optional[np.ndarray] = None) -> float:
    if xyh.size == 0:
        return float(empty_nll)
    diff = xyh - mu[None,:]
    m = np.einsum('ni,ij,nj->n', diff, cov_inv, diff)  # Mahalanobis^2 / object
    if w is not None and w.size == m.size and float(w.sum()) > 1e-6:
        m = float((m * w).sum() / w.sum())
    else:
        m = float(m.mean())
    D = 3
    return 0.5 * (m + logdet + D * np.log(2*np.pi))

# -------------------- 记录结构 --------------------
@dataclass
class Rec:
    pool_idx: int
    H_img: float
    c: int; p: int; y: int
    d_img: float
    q: float
    xyh: np.ndarray                  # (Ni,3) decode 后
    w: Optional[np.ndarray] = None   # (Ni,) 由 scores 归一化得到的权重

# =============================================================
# A=30% bin-entropy -> B=两层 diversity（最终10%）-> C=GaussSim 选 k
# 关键：C 步的目标高斯 (μ,Σ) **来自已标注集合的 GT**；候选的打分用“预测框”。
# =============================================================
class BinEnt30_TwoTier_ThenGaussSimSelector(BaseActiveSelector):
    def __init__(
        self,
        # A/B 占比（相对全池）
        frac_A_keep_total: float = 0.30,
        frac_B1_gate_of_A: float = 0.50,
        frac_B2_final_total: float = 0.10,
        dirichlet_alpha: float = 0.5,
        tie_quality_weight: float = 0.1,     # B2/C 的轻度质量 tie-breaker（-H_img）
        # C：高斯相似度
        gauss_mode: str = "sim",             # "sim"（低 NLL 优先）| "antisim"（高 NLL 优先）
        gauss_eps: float = 1e-3,             # 协方差稳定项
        empty_nll: float = 1e6,
        # seed 来源：counts 与 (μ,Σ) 的拟合来源（"gt" | "pred"），默认 "gt"
        seed_source: str = "gt",
        # 预测框的分数处理
        score_thr: float = 0.0,              # 过滤低分框（get_bboxes 后）
        score_weight: bool = True,           # 使用 scores 作为 NLL 权重
        score_norm: bool = True,             # 将 scores 线性归一化到 [0,1]
        **kwargs
    ):
        super().__init__(**kwargs)
        assert gauss_mode in ("sim","antisim")
        assert seed_source in ("gt","pred")
        self.frac_A_keep_total = float(frac_A_keep_total)
        self.frac_B1_gate_of_A = float(frac_B1_gate_of_A)
        self.frac_B2_final_total = float(frac_B2_final_total)
        self.alpha = float(dirichlet_alpha)
        self.w_q = float(tie_quality_weight)
        self.gauss_mode = gauss_mode
        self.gauss_eps = float(gauss_eps)
        self.empty_nll = float(empty_nll)
        self.seed_source = seed_source
        self.score_thr = float(score_thr)
        self.score_weight = bool(score_weight)
        self.score_norm = bool(score_norm)

    # --------- 统一解码：head -> get_bboxes（标准路径） ---------
    @torch.no_grad()
    def _decode_batch(self, model, sweep_imgs, mats, img_metas, device, need_entropy: bool):
        sweep_imgs, mats = _move_to_device(sweep_imgs, mats, device)
        out = model.model.backbone(
            sweep_imgs, mats, is_return_height=False, return_bin_entropy=need_entropy
        )
        feat = out[0] if isinstance(out, (tuple,list)) else out
        H_img = _extract_H_img(out) if need_entropy else None
        preds = model.model.head(feat)
        results = model.model.get_bboxes(preds, img_metas)  # list of (bboxes, scores, labels)
        return H_img, results

    @torch.no_grad()
    def _normalize_scores(self, s: np.ndarray) -> np.ndarray:
        if s is None or s.size == 0:
            return np.zeros((0,), np.float32)
        s = s.astype(np.float32, copy=False)
        if self.score_thr > 0:
            keep = s >= self.score_thr
            s = s[keep]
        if not self.score_weight:
            return np.ones_like(s, dtype=np.float32)
        if self.score_norm:
            lo, hi = float(s.min(initial=0.0)), float(s.max(initial=1.0))
            if hi > lo:
                s = (s - lo) / (hi - lo)
            else:
                s = np.ones_like(s, dtype=np.float32)
        return s

    # --------- 从已标注 dataloader 构建 seed（counts 与高斯目标） ---------
    @torch.no_grad()
    def _seed_from_labeled(self, model, labeled_loader, device) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        seed_counts = np.zeros(3, np.float64)
        feats_xyh: List[np.ndarray] = []

        if labeled_loader is None:
            return seed_counts, None, None

        model.eval(); model.model.to(device)

        for batch in labeled_loader:
            if len(batch) < 4:
                continue
            # 训练 batch 常见布局: (sweep_imgs, mats, _, img_metas, gt_boxes, gt_labels)
            sweep_imgs, mats, _, img_metas = batch[:4]

            if self.seed_source == "gt":
                # 直接从 GT 统计（无需前向）
                gt_boxes = batch[-2] if len(batch) >= 2 else None
                gt_labels= batch[-1] if len(batch) >= 1 else None
                # 统一为迭代器
                if isinstance(gt_labels, (list, tuple)) and isinstance(gt_boxes, (list, tuple)):
                    it = zip(gt_boxes, gt_labels)
                else:
                    it = [(gt_boxes, gt_labels)]
                for gb, gl in it:
                    if gl is None: continue
                    # counts
                    c,p,y = _per_image_counts_from_labels(gl)
                    seed_counts += np.array([c,p,y], np.float64)
                    # xyh from GT boxes
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
                    xyh = _xyh_from_boxes_np(arr)
                    if xyh.size > 0:
                        feats_xyh.append(xyh)
            else:
                # 用预测作为 seed（可做对照）
                try:
                    _, results = self._decode_batch(model, sweep_imgs, mats, img_metas, device, need_entropy=False)
                except Exception:
                    continue
                for (bboxes, scores, labels) in results:
                    c,p,y = _per_image_counts_from_labels(labels)
                    seed_counts += np.array([c,p,y], np.float64)
                    bboxes = _to_numpy(bboxes); scores = _to_numpy(scores)
                    xyh = _xyh_from_boxes_np(bboxes)
                    if xyh.size == 0: continue
                    _ = self._normalize_scores(scores)  # 这里不做权重拟合，简单用均值/协方差
                    feats_xyh.append(xyh)

        # 拟合高斯（μ, Σ）——若是 GT，通常无需权重
        if feats_xyh:
            X = np.concatenate(feats_xyh, axis=0).astype(np.float32, copy=False)
            mu = X.mean(axis=0).astype(np.float32)
            cov = np.cov(X, rowvar=False).astype(np.float32)
            if not np.isfinite(cov).all(): cov = np.eye(3, dtype=np.float32)
            cov = cov + self.gauss_eps * np.eye(3, dtype=np.float32)
            return seed_counts, mu, cov
        else:
            return seed_counts, None, None

    # --------- 扫描未标注池（单次前向：H_img + 解码预测）---------
    @torch.no_grad()
    def _collect_pool_records(self, model, dataloader, device, unlabeled_indices: List[int]) -> List[Rec]:
        model.eval(); model.model.to(device)
        recs: List[Rec] = []
        ptr = 0

        for batch in dataloader:
            if len(batch) < 4:
                raise RuntimeError("dataloader batch must provide (sweep_imgs, mats, _, img_metas, ...)")
            sweep_imgs, mats, _, img_metas = batch[:4]
            H_img, results = self._decode_batch(model, sweep_imgs, mats, img_metas, device, need_entropy=True)
            B = int(H_img.numel())

            for j in range(B):
                if ptr + j >= len(unlabeled_indices): break
                bboxes, scores, labels = results[j]
                c,p,y = _per_image_counts_from_labels(labels)
                d_img = _per_image_diversity(c,p,y, self.alpha)
                # xyh / 权重
                bboxes = _to_numpy(bboxes); scores = _to_numpy(scores)
                xyh = _xyh_from_boxes_np(bboxes)
                w = self._normalize_scores(scores) if scores is not None else None
                recs.append(
                    Rec(pool_idx=ptr+j,
                        H_img=float(H_img[j].item()),
                        c=c, p=p, y=y,
                        d_img=d_img,
                        q=-float(H_img[j].item()),
                        xyh=xyh, w=w)
                )
            ptr += B

        if len(recs) != len(unlabeled_indices):
            raise RuntimeError(f"Collected {len(recs)} recs != pool {len(unlabeled_indices)}; check dataloader subset.")
        return recs

    # =================== 主流程 ===================
    @torch.no_grad()
    def select(
        self,
        model: torch.nn.Module,
        dataloader,                    # unlabeled dataloader（需含 img_metas）
        device: torch.device,
        unlabeled_indices: List[int],
        k: int,
        labeled_dataloader=None,       # 已标注 dataloader（含 gt_boxes/gt_labels & img_metas）
    ) -> List[int]:

        N = len(unlabeled_indices)

        # 1) 用“已标注集”的 **GT**（默认）构建 seed：counts 与高斯目标
        seed_counts, mu, cov = self._seed_from_labeled(model, labeled_dataloader, device)
        if mu is None or cov is None:
            # 兜底
            mu = np.zeros((3,), np.float32)
            cov = np.eye(3, dtype=np.float32) * (1.0 + self.gauss_eps)
        cov = cov + self.gauss_eps * np.eye(3, dtype=np.float32)
        cov_inv = np.linalg.inv(cov)
        sign, logdet = np.linalg.slogdet(cov)
        logdet = float(logdet) if sign > 0 else 0.0

        # 2) 扫描未标注池：H_img + 预测（解码）
        recs = self._collect_pool_records(model, dataloader, device, unlabeled_indices)

        # ---------- A) bin-entropy：保留全池 30% ----------
        A_keep = max(1, int(np.ceil(self.frac_A_keep_total * N)))
        order_by_H = np.argsort([r.H_img for r in recs])  # 小在前
        idx_A = order_by_H[:A_keep]
        cand_A = [recs[i] for i in idx_A]

        # ---------- B1) per-image diversity gating（相对 A 的比例） ----------
        B1_keep = max(1, int(np.ceil(self.frac_B1_gate_of_A * len(cand_A))))
        order_by_div = np.argsort([-r.d_img for r in cand_A])  # 大在前
        idx_B1 = order_by_div[:B1_keep]
        cand_B1 = [cand_A[i] for i in idx_B1]

        # ---------- B2) 集合级类别熵贪心：保留全池 10% ----------
        B2_keep_total = max(1, int(np.ceil(self.frac_B2_final_total * N)))
        S_counts = seed_counts.astype(np.float64).copy()
        H_S = _set_entropy_from_counts(S_counts, self.alpha)
        avail = cand_B1[:]
        chosen_B: List[int] = []
        for _ in range(min(B2_keep_total, len(avail))):
            best_i, best_gain = -1, -1e18
            for i, r in enumerate(avail):
                if r is None: continue
                add = np.array([r.c, r.p, r.y], np.float64)
                H_new = _set_entropy_from_counts(S_counts + add, self.alpha)
                dH = H_new - H_S
                gain = dH + self.w_q * r.q
                if gain > best_gain:
                    best_gain, best_i = gain, i
            if best_i < 0: break
            rb = avail[best_i]
            chosen_B.append(rb.pool_idx)
            S_counts += np.array([rb.c, rb.p, rb.y], np.float64)
            H_S = _set_entropy_from_counts(S_counts, self.alpha)
            avail[best_i] = None

        # 候选不足时回填（按 per-image diversity）
        pool_pos = set(chosen_B)
        if len(chosen_B) < min(k, len(cand_B1)):
            for r in cand_B1:
                if len(chosen_B) >= max(B2_keep_total, k): break
                if r.pool_idx not in pool_pos:
                    chosen_B.append(r.pool_idx)

        cand_B2 = [recs[i] for i in chosen_B] if chosen_B else cand_A[:]  # 兜底

        # ---------- C) GaussSim（μ,Σ 来自已标注 GT）：按 NLL 排序选出最终 k ----------
        nlls = []
        for r in cand_B2:
            w = r.w if self.score_weight else None
            nlls.append(_avg_nll_weighted(r.xyh, mu, cov_inv, logdet, self.empty_nll, w))
        order = np.argsort(nlls)  # 低 NLL 更相似
        if self.gauss_mode == "antisim":
            order = order[::-1]   # 高 NLL 更不相似（对照组）
        top_rel = order[:min(k, len(order))].tolist()

        chosen_final = [cand_B2[i].pool_idx for i in top_rel]
        return [unlabeled_indices[i] for i in chosen_final]


# ===== 注册（兼容你之前的命名） =====
try:
    from active_learning.methods import _METHOD_REGISTRY
    _METHOD_REGISTRY["binEnt30_twoTier_then_gauss_sim"]  = BinEnt30_TwoTier_ThenGaussSimSelector
    _METHOD_REGISTRY["binEnt30_twoTier_then_gauss_pred"] = BinEnt30_TwoTier_ThenGaussSimSelector  # 向后兼容
except Exception:
    pass
