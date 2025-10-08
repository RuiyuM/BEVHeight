from __future__ import annotations
from typing import List
import numpy as np
import torch
from active_learning.base import BaseActiveSelector

class _LabelDiversityBase(BaseActiveSelector):
    """Compute per-image Shannon entropy of *ground-truth* class distribution.

    Entropy is normalized to [0,1] by dividing by ln(C), where C is total #classes.
    Images with no GT objects get entropy 0.0.
    """

    def __init__(self, maximize: bool, **kwargs):
        super().__init__(**kwargs)
        self.maximize = bool(maximize)

    def _num_classes(self, model) -> int:
        # Prefer explicit class_names length
        if hasattr(model, 'class_names') and model.class_names:
            return int(len(model.class_names))
        # Fallback: sum task class counts
        try:
            tasks = model.head_conf.get('tasks', [])
            return int(sum(t.get('num_class', 0) for t in tasks)) or 1
        except Exception:
            return 1

    def _gt_labels_for_index(self, model, idx: int):
        ds = model._train_dataset if model._train_dataset is not None else model._build_train_dataset()
        sample = ds[idx]
        # Expected tuple: (sweep_imgs, mats, _, _, gt_boxes, gt_labels)
        if isinstance(sample, (list, tuple)) and len(sample) >= 6:
            gt_labels = sample[5]
        else:
            gt_labels = None
        if isinstance(gt_labels, torch.Tensor):
            return gt_labels.detach().cpu().numpy().astype(np.int64)
        if hasattr(gt_labels, '__iter__'):
            try:
                return np.asarray(list(gt_labels), dtype=np.int64)
            except Exception:
                return np.empty((0,), dtype=np.int64)
        return np.empty((0,), dtype=np.int64)

    @torch.no_grad()
    def select(self, model, dataloader, device, unlabeled_indices, k):
        import numpy as np, math, torch
        C = max(1, self._num_classes(model))
        logC = math.log(C) if C > 1 else 1.0
        scores = []

        for batch in dataloader:
            # 与 training_step 一致的解包：(..., gt_boxes, gt_labels)
            _, _, _, _, _gt_boxes, _gt_labels = batch
            for gl in _gt_labels:
                if isinstance(gl, torch.Tensor):
                    gl = gl.detach().cpu().numpy()
                gl = gl[(gl >= 0) & (gl < C)]
                if gl.size == 0:
                    scores.append(0.0)
                    continue
                cnts = np.bincount(gl, minlength=C).astype(np.float64)
                p = cnts / cnts.sum()
                ent = -np.sum(np.where(p > 0, p * np.log(p), 0.0))
                scores.append(float(ent / logC))

        assert len(scores) == len(unlabeled_indices), f"mismatch: {len(scores)} vs {len(unlabeled_indices)}"
        order = np.argsort(scores)  # 小→大
        if self.maximize: order = order[::-1]  # 需要大→小时反转
        topk_rel = order[: min(k, len(order))].tolist()
        return [unlabeled_indices[i] for i in topk_rel]


class LabelDiversityHighSelector(_LabelDiversityBase):
    def __init__(self, **kwargs):
        super().__init__(maximize=True, **kwargs)

class LabelDiversityLowSelector(_LabelDiversityBase):
    def __init__(self, **kwargs):
        super().__init__(maximize=False, **kwargs)
