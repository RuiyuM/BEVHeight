from argparse import ArgumentParser, Namespace
import sys, os
BEVHEIGHT_ROOT = "/people/cs/r/rxm210041/Desktop/test_3d_active/BEVHeight"
if BEVHEIGHT_ROOT not in sys.path:
    sys.path.insert(0, BEVHEIGHT_ROOT)

import mmcv
import pytorch_lightning as pl
import torch
import torch.nn.parallel
import torch.utils.data
import torch.utils.data.distributed
import torchvision.models as models
from pytorch_lightning.core import LightningModule
from torch.optim.lr_scheduler import MultiStepLR

from dataset.nusc_mv_det_dataset import NuscMVDetDataset, collate_fn
from evaluators.det_evaluators import RoadSideEvaluator
from models.bev_height import BEVHeight
from utils.torch_dist import all_gather_object, get_rank, synchronize

# === (Keep your constants/configs as-is) ===
H = 1080
W = 1920
final_dim = (864, 1536)
img_conf = dict(img_mean=[123.675, 116.28, 103.53], img_std=[58.395, 57.12, 57.375], to_rgb=True)

data_root = "/data/rxm210041/dair-v2x-i/"
gt_label_path = "/data/rxm210041/dair-v2x-i-kitti/training/label_2"

backbone_conf = {
    'x_bound': [0, 102.4, 0.8],
    'y_bound': [-51.2, 51.2, 0.8],
    'z_bound': [-5, 3, 8],
    'd_bound': [-2.0, 0.0, 90],
    'final_dim': final_dim,
    'output_channels': 80,
    'downsample_factor': 16,
    'img_backbone_conf': dict(
        type='ResNet', depth=50, frozen_stages=0, out_indices=[0,1,2,3], norm_eval=False,
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'),
    ),
    'img_neck_conf': dict(
        type='SECONDFPN', in_channels=[256,512,1024,2048], upsample_strides=[0.25,0.5,1,2], out_channels=[128,128,128,128],
    ),
    'height_net_conf': dict(in_channels=512, mid_channels=512)
}
ida_aug_conf = {
    'final_dim': final_dim, 'H': H, 'W': W,
    'bot_pct_lim': (0.0, 0.0), 'cams': ['CAM_FRONT'], 'Ncams': 1,
}

bev_backbone = dict(type='ResNet', in_channels=80, depth=18, num_stages=3, strides=(1,2,2),
                    dilations=(1,1,1), out_indices=[0,1,2], norm_eval=False, base_channels=160)
bev_neck = dict(type='SECONDFPN', in_channels=[80,160,320,640], upsample_strides=[1,2,4,8], out_channels=[64,64,64,64])

CLASSES = ['car','truck','construction_vehicle','bus','trailer','barrier','motorcycle','bicycle','pedestrian','traffic_cone']
TASKS = [
    dict(num_class=1, class_names=['car']),
    dict(num_class=2, class_names=['truck','construction_vehicle']),
    dict(num_class=2, class_names=['bus','trailer']),
    dict(num_class=1, class_names=['barrier']),
    dict(num_class=2, class_names=['motorcycle','bicycle']),
    dict(num_class=2, class_names=['pedestrian','traffic_cone']),
]
common_heads = dict(reg=(2,2), height=(1,2), dim=(3,2), rot=(2,2), vel=(2,2))

bbox_coder = dict(
    type='CenterPointBBoxCoder', post_center_range=[0.0,-61.2,-10.0,122.4,61.2,10.0],
    max_num=500, score_threshold=0.1, out_size_factor=4, voxel_size=[0.2,0.2,8],
    pc_range=[0,-51.2,-5,104.4,51.2,3], code_size=9,
)

train_cfg = dict(
    point_cloud_range=[0, -51.2, -5, 102.4, 51.2, 3], grid_size=[512,512,1], voxel_size=[0.2,0.2,8],
    out_size_factor=4, dense_reg=1, gaussian_overlap=0.1, max_objs=500, min_radius=2,
    code_weights=[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,0.5,0.5],
)

test_cfg = dict(
    post_center_limit_range=[0.0,-61.2,-10.0,122.4,61.2,10.0], max_per_img=500, max_pool_nms=False,
    min_radius=[4,12,10,1,0.85,0.175], score_threshold=0.1, out_size_factor=4, voxel_size=[0.2,0.2,8],
    nms_type='circle', pre_max_size=1000, post_max_size=83, nms_thr=0.2,
)

head_conf = {
    'bev_backbone_conf': bev_backbone,
    'bev_neck_conf': bev_neck,
    'tasks': TASKS,
    'common_heads': common_heads,
    'bbox_coder': bbox_coder,
    'train_cfg': train_cfg,
    'test_cfg': test_cfg,
    'in_channels': 256,
    'loss_cls': dict(type='GaussianFocalLoss', reduction='mean'),
    'loss_bbox': dict(type='L1Loss', reduction='mean', loss_weight=0.25),
    'gaussian_overlap': 0.1, 'min_radius': 2,
}

class BEVHeightLightningModel(LightningModule):
    def __init__(self,
                 gpus: int = 1,
                 data_root=data_root,
                 eval_interval=1,
                 batch_size_per_device=8,
                 class_names=CLASSES,
                 backbone_conf=backbone_conf,
                 head_conf=head_conf,
                 ida_aug_conf=ida_aug_conf,
                 default_root_dir='outputs/',
                 **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.gpus = gpus
        self.eval_interval = eval_interval
        self.batch_size_per_device = batch_size_per_device
        self.data_root = data_root
        self.basic_lr_per_img = 2e-4 / 64
        self.class_names = class_names
        self.backbone_conf = backbone_conf
        self.head_conf = head_conf
        self.ida_aug_conf = ida_aug_conf
        mmcv.mkdir_or_exist(default_root_dir)
        self.default_root_dir = default_root_dir
        self.evaluator = RoadSideEvaluator(class_names=self.class_names,
                                           current_classes=["Car","Pedestrian","Cyclist"],
                                           data_root=data_root,
                                           gt_label_path=gt_label_path,
                                           output_dir=self.default_root_dir)
        self.model = BEVHeight(self.backbone_conf, self.head_conf)
        self.mode = 'valid'
        self.img_conf = img_conf
        self.data_use_cbgs = False
        self.num_sweeps = 1
        self.sweep_idxes = list()
        self.key_idxes = list()
        self.up_stride = 8
        self.downsample_factor = self.backbone_conf['downsample_factor'] // self.up_stride
        self.dbound = self.backbone_conf['d_bound']
        self.height_channels = int(self.dbound[2])

        # ### [AL-HOOK] indices for labeled/unlabeled
        self._labeled_indices = None
        self._train_dataset = None

    # ### [AL-HOOK]
    def set_labeled_indices(self, indices):
        self._labeled_indices = sorted(indices) if indices is not None else None

    # ### [AL-HOOK] Build full train dataset once for AL runner
    def _build_train_dataset(self):
        ds = NuscMVDetDataset(
            ida_aug_conf=self.ida_aug_conf,
            classes=self.class_names,
            data_root=self.data_root,
            info_path=os.path.join(self.data_root, 'dair_12hz_infos_train.pkl'),
            is_train=True,
            use_cbgs=self.data_use_cbgs,
            img_conf=self.img_conf,
            num_sweeps=self.num_sweeps,
            sweep_idxes=self.sweep_idxes,
            key_idxes=self.key_idxes,
            return_depth=False,
        )
        self._train_dataset = ds
        return ds

    # ### [AL-HOOK]
    def _collate_fn_for_train(self):
        from functools import partial
        return partial(collate_fn, is_return_depth=False)

    def forward(self, sweep_imgs, mats):
        return self.model(sweep_imgs, mats)

    def training_step(self, batch):
        (sweep_imgs, mats, _, _, gt_boxes, gt_labels) = batch
        if torch.cuda.is_available():
            for key, value in mats.items():
                mats[key] = value.cuda()
            sweep_imgs = sweep_imgs.cuda()
            gt_boxes = [gt_box.cuda() for gt_box in gt_boxes]
            gt_labels = [gt_label.cuda() for gt_label in gt_labels]
        preds = self(sweep_imgs, mats)
        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            targets = self.model.module.get_targets(gt_boxes, gt_labels)
            detection_loss = self.model.module.loss(targets, preds)
        else:
            targets = self.model.get_targets(gt_boxes, gt_labels)
            detection_loss = self.model.loss(targets, preds)
        self.log('detection_loss', detection_loss)
        return detection_loss

    @torch.no_grad()
    def eval_step(self, batch, batch_idx, prefix: str):
        (sweep_imgs, mats, _, img_metas, _, _) = batch
        if torch.cuda.is_available():
            for key, value in mats.items():
                mats[key] = value.cuda()
            sweep_imgs = sweep_imgs.cuda()
        preds = self.model(sweep_imgs, mats)
        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            results = self.model.module.get_bboxes(preds, img_metas)
        else:
            results = self.model.get_bboxes(preds, img_metas)
        for i in range(len(results)):
            results[i][0] = results[i][0].tensor.detach().cpu().numpy()
            results[i][1] = results[i][1].detach().cpu().numpy()
            results[i][2] = results[i][2].detach().cpu().numpy()
            results[i].append(img_metas[i])
        return results

    def validation_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx, 'val')

    def validation_epoch_end(self, validation_step_outputs):
        all_pred_results = list()
        all_img_metas = list()
        for validation_step_output in validation_step_outputs:
            for i in range(len(validation_step_output)):
                all_pred_results.append(validation_step_output[i][:3])
                all_img_metas.append(validation_step_output[i][3])
        synchronize()
        len_dataset = len(self.val_dataloader().dataset)
        all_pred_results = sum(map(list, zip(*all_gather_object(all_pred_results))), [])[:len_dataset]
        all_img_metas = sum(map(list, zip(*all_gather_object(all_img_metas))), [])[:len_dataset]
        if get_rank() == 0:
            self.evaluator.evaluate(all_pred_results, all_img_metas)

    def configure_optimizers(self):
        lr = self.basic_lr_per_img * self.batch_size_per_device * self.gpus
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-7)
        scheduler = MultiStepLR(optimizer, [19, 23])
        return [[optimizer], [scheduler]]

    def train_dataloader(self):
        from torch.utils.data import Subset
        # Build or reuse the full dataset
        ds = self._train_dataset if self._train_dataset is not None else self._build_train_dataset()

        # Subset to labeled indices for AL (if provided)
        if self._labeled_indices is not None:
            train_dataset = Subset(ds, self._labeled_indices)
        else:
            train_dataset = ds

        loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size_per_device,
            num_workers=4,
            drop_last=True,
            shuffle=True,
            collate_fn=self._collate_fn_for_train(),
        )
        return loader

    def val_dataloader(self):
        val_dataset = NuscMVDetDataset(
            ida_aug_conf=self.ida_aug_conf,
            classes=self.class_names,
            data_root=self.data_root,
            info_path=os.path.join(self.data_root, 'dair_12hz_infos_val.pkl'),
            is_train=False,
            img_conf=self.img_conf,
            num_sweeps=self.num_sweeps,
            sweep_idxes=self.sweep_idxes,
            key_idxes=self.key_idxes,
            return_depth=False,
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.batch_size_per_device,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=4,
        )
        return val_loader

    @staticmethod
    def add_model_specific_args(parent_parser):
        return parent_parser

# === Runner ===
from active_learning.runner import ActiveLearner

def main(args: Namespace) -> None:
    if args.seed is not None:
        pl.seed_everything(args.seed)
    print(args)

    if args.al_enabled:
        print("[AL] Active Learning enabled.")
        learner = ActiveLearner(BEVHeightLightningModel, args)
        learner.run()
    else:
        model = BEVHeightLightningModel(**vars(args))
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=os.path.join(args.default_root_dir, 'checkpoints'), filename='{epoch}', every_n_epochs=5,
            save_last=True, save_top_k=-1
        )
        trainer = pl.Trainer.from_argparse_args(args, callbacks=[checkpoint_callback])
        from utils.backup_files import backup_codebase
        backup_codebase(os.path.join(args.default_root_dir, 'backup'))
        trainer.fit(model)


def run_cli():
    parent_parser = ArgumentParser(add_help=False)
    parent_parser = pl.Trainer.add_argparse_args(parent_parser)
    parent_parser.add_argument('-b', '--batch_size_per_device', type=int, default=8)
    parent_parser.add_argument('--seed', type=int, default=0)
    parent_parser.add_argument('--ckpt_path', type=str, default="")

    # ### [AL-HOOK] CLI arguments for active learning
    parent_parser.add_argument('--al_enabled', action='store_true', help='Enable active learning loop')
    parent_parser.add_argument('--al_method', type=str, default='uncertainty', help='Active method name (e.g., uncertainty)')
    parent_parser.add_argument('--al_init_size', type=int, default=100, help='Initial random labeled images')
    parent_parser.add_argument('--al_query_size', type=int, default=100, help='Images to add per AL round')
    parent_parser.add_argument('--al_rounds', type=int, default=5, help='Number of AL query rounds')
    parent_parser.add_argument('--al_epochs_per_round', type=int, default=10, help='Train epochs per AL round')
    parent_parser.add_argument('--al_pool_seed', type=int, default=0, help='Seed for initial pool split')

    parser = BEVHeightLightningModel.add_model_specific_args(parent_parser)
    parser.set_defaults(
        profiler='simple', deterministic=False, max_epochs=100,
        accelerator='gpu', devices=1,  # safer default for AL; override as needed
        num_sanity_val_steps=0, gradient_clip_val=5, limit_val_batches=0, enable_checkpointing=True,
        precision=32, default_root_dir='/data/rxm210041/outputs/bev_height_al'
    )
    args = parser.parse_args()
    main(args)

if __name__ == '__main__':
    run_cli()

# =============================
# USAGE EXAMPLE
# =============================
# 1) Place files:
#    active_learning/
#      __init__.py (optional)
#      base.py
#      runner.py
#      methods/
#        __init__.py
#        uncertainty.py
# 2) Add the new training script at:
#    exps/dair-v2x/bev_height_lss_r50_864_1536_128x128_active.py
# 3) Run (single GPU recommended initially):
#    python exps/dair-v2x/bev_height_lss_r50_864_1536_128x128_active.py \
#      --al_enabled --al_method uncertainty \
#      --al_init_size 100 --al_query_size 100 --al_rounds 5 \
#      --al_epochs_per_round 10 --batch_size_per_device 8 --devices 1
