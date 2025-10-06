from __future__ import annotations
from typing import List, Optional
import os, random
import torch
import pytorch_lightning as pl
from torch.utils.data import Subset, DataLoader

from active_learning.methods import get_method

class ActiveLearner:
    """Round-based pool-based active learning orchestrator.

    Orchestrates: init label set -> train -> query -> repeat.
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
        # Respect user-specified trainer flags when present
        trainer = pl.Trainer.from_argparse_args(
            self.args,
            max_epochs=self.args.al_epochs_per_round,
            callbacks=[checkpoint_callback],
            # For AL loops, single GPU is safest; DDP works but selection runs on rank0 only.
            # users can still override via CLI.
        )
        return trainer

    def _train_one_round(self, round_id: int):
        # Tell the model which indices to train on
        self.model.set_labeled_indices(self.labeled_indices)
        trainer = self._make_trainer(round_id)
        # Backup codebase per round is optional; reuse user's function if present
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
        # Build a simple DataLoader over the unlabeled subset (no Lightning, no shuffle)
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
            print(f"\n[AL] ===== Round {r+1}/{R}: Train on {len(self.labeled_indices)} labeled samples =====")
            self._train_one_round(r)

            # Last round does not need querying
            if r == R - 1:
                break

            k = min(Q, len(self.unlabeled_indices))
            print(f"[AL] Querying top-{k} most uncertain images from pool={len(self.unlabeled_indices)}…")
            picked = self._query_unlabeled(k)
            print(f"[AL] Picked {len(picked)} new samples.")

            # Move from unlabeled → labeled
            pool_set = set(self.unlabeled_indices)
            picked_set = set(picked)
            assert picked_set.issubset(pool_set)
            self.labeled_indices.extend(picked)
            self.labeled_indices = sorted(set(self.labeled_indices))
            self.unlabeled_indices = sorted(list(pool_set - picked_set))

        print("[AL] Finished all rounds.")

