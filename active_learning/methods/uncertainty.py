from __future__ import annotations
from typing import List, Tuple
import torch
from active_learning.base import BaseActiveSelector

class UncertaintySelector(BaseActiveSelector):
    """Simple, fast uncertainty selector for BEVHeight detection.

    For each image (batch element), we compute two proxies:
    - Classification uncertainty: mean(1 - score_i) over detected boxes.
    - Regression uncertainty: mean(distance(center)/max_range) over boxes,
      where center=(x, y) in BEV space and max_range is derived from model cfg.

    The final image score is the average of the two. Images with no detections
    are assigned a high uncertainty (1.0).

    Notes:
      - This is a lightweight baseline. You can swap in MC-dropout/TTA-based
        variance by implementing another selector and keeping this interface.
    """

    @torch.no_grad()
    def score_unlabeled(self, model: torch.nn.Module, dataloader, device) -> List[float]:
        model.eval()
        scores_per_image: List[float] = []

        # Range from model config (used to normalize center distance)
        # Fallback if not present
        try:
            pc_range = model.head_conf['train_cfg']['point_cloud_range'] if hasattr(model, 'head_conf') else [0, -51.2, -5, 102.4, 51.2, 3]
        except Exception:
            pc_range = [0, -51.2, -5, 102.4, 51.2, 3]
        max_xy = max(abs(pc_range[1]), abs(pc_range[4]), abs(pc_range[0]), abs(pc_range[3]))
        max_range = float(max_xy) if max_xy > 0 else 60.0

        for batch in dataloader:
            sweep_imgs, mats, _, img_metas, _, _ = batch
            if torch.cuda.is_available():
                for key, value in mats.items():
                    mats[key] = value.to(device, non_blocking=True)
                sweep_imgs = sweep_imgs.to(device, non_blocking=True)

            # Raw preds
            preds = model.model(sweep_imgs, mats)
            # Decode bboxes for each image in batch
            if isinstance(model.model, torch.nn.parallel.DistributedDataParallel):
                dets = model.model.module.get_bboxes(preds, img_metas)
            else:
                dets = model.model.get_bboxes(preds, img_metas)

            # dets[i] = (bboxes[Tensor(N, 9)], scores[Tensor(N)], labels[Tensor(N)],)  
            for i in range(len(dets)):
                bboxes, scores, labels = dets[i][:3]
                # Move to CPU numpy for simple math
                if hasattr(bboxes, 'tensor'):
                    # mmdet3d style sometimes wraps boxes
                    centers = bboxes.tensor[:, :2].detach().cpu().numpy()
                else:
                    centers = bboxes[:, :2].detach().cpu().numpy()
                sc = scores.detach().cpu().numpy()

                if sc.size == 0:
                    scores_per_image.append(1.0)  # maximally uncertain if nothing detected
                    continue

                # Classification uncertainty
                cls_unc = float(1.0 - sc.mean())

                # Regression proxy uncertainty via BEV center distance
                import numpy as np
                dists = (centers[:, 0] ** 2 + centers[:, 1] ** 2) ** 0.5
                reg_unc = float(np.clip(dists / max_range, 0.0, 1.0).mean())

                # Combine
                scores_per_image.append((cls_unc + reg_unc) / 2.0)

        return scores_per_image