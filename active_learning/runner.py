from __future__ import annotations
from typing import List, Optional, Dict
import os, random
import torch
import pytorch_lightning as pl
from torch.utils.data import Subset, DataLoader

from active_learning.methods import get_method

class ActiveLearner:
    """Round-based pool-based active learning orchestrator.

    Orchestrates: init label set -> train -> query -> object-budget check -> repeat.
    """
    def __init__(self, model_cls, args, checkpoint_dir: Optional[str] = None):
        self.model_cls = model_cls
        self.args = args
        self.checkpoint_dir = checkpoint_dir or os.path.join(args.default_root_dir, "checkpoints_al")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Instantiate a single model and keep weights across rounds
        self.model = model_cls(**vars(args)).to(self.device)

        # Build the *full* train dataset once; we'll subset via indices
        self.full_train_dataset = self.model._build_train_dataset()
        self.N = len(self.full_train_dataset)

        # Build initial labeled / unlabeled pools
        self._init_pools()

        # Active method
        Method = get_method(args.al_method)
        self.selector = Method()

        # Object budget (soft-cap support)
        self.obj_budget: Optional[int] = getattr(args, 'al_max_objects', None)
        self.obj_overshoot: int = int(getattr(args, 'al_objects_overshoot', 200))
        self.obj_overshoot_ratio: float = float(getattr(args, 'al_objects_overshoot_ratio', 0.0))
        self._gt_count_cache: Dict[int, int] = {}
        self._budget_exhausted = False

    def _init_pools(self):
        rng = random.Random(self.args.al_pool_seed)
        all_idx = list(range(self.N))
        rng.shuffle(all_idx)
        init_sz = min(self.args.al_init_size, self.N)
        self.labeled_indices = sorted(all_idx[:init_sz])
        self.unlabeled_indices = sorted(all_idx[init_sz:])
        print(f"[AL] Dataset size={self.N} | init labeled={len(self.labeled_indices)} | pool={len(self.unlabeled_indices)}")

    def _make_trainer(self, round_id: int) -> pl.Trainer:
        # Create a fresh trainer each round. Keep it simple & robust.
        ckpt_dir = os.path.join(self.checkpoint_dir, f"round_{round_id}")
        os.makedirs(ckpt_dir, exist_ok=True)
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=ckpt_dir,
            filename="{epoch}",
            every_n_epochs=max(1, self.args.al_epochs_per_round // 2),
            save_last=True, save_top_k=-1
        )
        trainer = pl.Trainer.from_argparse_args(
            self.args,
            max_epochs=self.args.al_epochs_per_round,
            callbacks=[checkpoint_callback],
        )
        return trainer

    def _gt_count_for_index(self, idx: int) -> int:
        if idx in self._gt_count_cache:
            return self._gt_count_cache[idx]
        try:
            sample = self.full_train_dataset[idx]
            # Expected tuple: (sweep_imgs, mats, _, _, gt_boxes, gt_labels)
            gt_labels = None
            if isinstance(sample, (list, tuple)) and len(sample) >= 6:
                gt_labels = sample[5]
            # Count
            if gt_labels is None:
                cnt = 0
            elif isinstance(gt_labels, torch.Tensor):
                cnt = int(gt_labels.shape[0])
            elif hasattr(gt_labels, '__len__'):
                cnt = int(len(gt_labels))
            else:
                cnt = 0
        except Exception:
            cnt = 0
        self._gt_count_cache[idx] = cnt
        return cnt

    def _total_objects(self, indices: List[int]) -> int:
        return sum(self._gt_count_for_index(i) for i in indices)

    def _cap_by_object_budget(self, picked_in_order: List[int]) -> List[int]:
        """Hard cap AFTER whole-image annotation.

        We simulate actual labeling: add images in the method's sorted order.
        After *adding* each image, recompute total labeled objects; if it now
        exceeds the cap, we keep this image (allow slight overshoot) and STOP.
        """
        if not self.obj_budget:
            return picked_in_order

        cap = int(self.obj_budget)
        current = self._total_objects(self.labeled_indices)

        kept: List[int] = []
        added_objs = 0
        for idx in picked_in_order:
            c = self._gt_count_for_index(idx)
            kept.append(idx)
            added_objs += c
            new_total = current + added_objs
            if new_total > cap:
                print(f"[AL] Reached cap after this image: {new_total} > {cap}. Stop querying this round.")
                break
        return kept

    def _train_one_round(self, round_id: int):
        # Tell the model which indices to train on
        self.model.set_labeled_indices(self.labeled_indices)
        trainer = self._make_trainer(round_id)
        try:
            from utils.backup_files import backup_codebase
            backup_codebase(os.path.join(self.args.default_root_dir, 'backup_round_%d' % round_id))
        except Exception:
            pass
        trainer.fit(self.model)

    @torch.no_grad()
    def _query_unlabeled(self, k: int) -> List[int]:
        if k <= 0 or len(self.unlabeled_indices) == 0:
            return []
        subset = Subset(self.full_train_dataset, self.unlabeled_indices)
        loader = DataLoader(
            subset,
            batch_size=self.args.batch_size_per_device,
            shuffle=False,
            num_workers=4,
            collate_fn=self.model._collate_fn_for_train(),
            drop_last=False,
        )
        chosen_global_idx = self.selector.select(
            model=self.model,
            dataloader=loader,
            device=self.device,
            unlabeled_indices=self.unlabeled_indices,
            k=k,
        )
        return chosen_global_idx

    def run(self):
        R = int(self.args.al_rounds)
        Q = int(self.args.al_query_size)
        for r in range(R):
            print(f"[AL] ===== Round {r+1}/{R}: Train on {len(self.labeled_indices)} labeled samples =====")
            # Budget check BEFORE training (optional); most users want to still train current set
            self._train_one_round(r)

            # Last round does not need querying (最后一轮不再 query，但仍已完成训练)
            if r == R - 1:
                continue  # 不再 query，直接进入下一轮（for 会自然结束）

            # 预算检查：不再停止训练，只是关闭后续 query
            if self.obj_budget:
                current_objs = self._total_objects(self.labeled_indices)
                print(f"[AL] Labeled objects so far: {current_objs} (cap={self.obj_budget})")
                if current_objs >= self.obj_budget:
                    self._budget_exhausted = True
                    print(
                        "[AL] Object cap reached. Will SKIP further querying but continue training to keep total epochs constant.")

            # 若预算已用尽或池子空了：跳过查询，但继续下一轮训练（保证总 epoch 恒定）
            if self._budget_exhausted or len(self.unlabeled_indices) == 0:
                print(
                    "[AL] Query skipped (budget exhausted or pool empty). Training will continue with the same labeled set.")
                continue

            k = min(Q, len(self.unlabeled_indices))
            print(f"[AL] Querying top-{k} samples from pool={len(self.unlabeled_indices)}…")
            picked = self._query_unlabeled(k)
            picked = self._cap_by_object_budget(picked)

            # 移动样本（即便 picked 为空也不影响后续继续训练）
            pool_set = set(self.unlabeled_indices)
            picked_set = set(picked)
            self.labeled_indices = sorted(set(self.labeled_indices).union(picked_set))
            self.unlabeled_indices = sorted(list(pool_set - picked_set))

            # 统计并记录预算状态
            tot_objs = self._total_objects(self.labeled_indices)
            print(f"[AL] After round {r + 1}: labeled_images={len(self.labeled_indices)}, labeled_objects≈{tot_objs}")
            if self.obj_budget and tot_objs >= self.obj_budget:
                self._budget_exhausted = True
                print("[AL] Object cap reached after this query. Future rounds will SKIP querying but still train.")

        print("[AL] Finished all rounds.")