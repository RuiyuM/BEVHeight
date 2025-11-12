# Copyright (c) Megvii Inc. All rights reserved.
from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F
from mmcv.cnn import build_conv_layer
from mmdet3d.models import build_neck
from mmdet.models import build_backbone
from mmdet.models.backbones.resnet import BasicBlock
from torch import nn

from ops.voxel_pooling import voxel_pooling

__all__ = ['LSSFPN']


# -----------------------------
#   Blocks
# -----------------------------
class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation, BatchNorm):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(
            inplanes, planes, kernel_size=kernel_size, stride=1,
            padding=padding, dilation=dilation, bias=False
        )
        self.bn = BatchNorm(planes)
        self.relu = nn.ReLU()
        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)
        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class ASPP(nn.Module):
    def __init__(self, inplanes, mid_channels=256, BatchNorm=nn.BatchNorm2d):
        super(ASPP, self).__init__()
        dilations = [1, 6, 12, 18]
        self.aspp1 = _ASPPModule(inplanes, mid_channels, 1, padding=0, dilation=dilations[0], BatchNorm=BatchNorm)
        self.aspp2 = _ASPPModule(inplanes, mid_channels, 3, padding=dilations[1], dilation=dilations[1], BatchNorm=BatchNorm)
        self.aspp3 = _ASPPModule(inplanes, mid_channels, 3, padding=dilations[2], dilation=dilations[2], BatchNorm=BatchNorm)
        self.aspp4 = _ASPPModule(inplanes, mid_channels, 3, padding=dilations[3], dilation=dilations[3], BatchNorm=BatchNorm)
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(inplanes, mid_channels, 1, stride=1, bias=False),
            BatchNorm(mid_channels),
            nn.ReLU(),
        )
        self.conv1 = nn.Conv2d(int(mid_channels * 5), mid_channels, 1, bias=False)
        self.bn1 = BatchNorm(mid_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self._init_weight()

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        return self.dropout(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x); x = self.act(x); x = self.drop1(x)
        x = self.fc2(x); x = self.drop2(x)
        return x


class SELayer(nn.Module):
    def __init__(self, channels, act_layer=nn.ReLU, gate_layer=nn.Sigmoid):
        super().__init__()
        self.conv_reduce = nn.Conv2d(channels, channels, 1, bias=True)
        self.act1 = act_layer()
        self.conv_expand = nn.Conv2d(channels, channels, 1, bias=True)
        self.gate = gate_layer()

    def forward(self, x, x_se):
        """
        x:    [B*N_cam, C, H, W]
        x_se: [B*N_cam, C, 1, 1]
        """
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        return x * self.gate(x_se)


# -----------------------------
#   DepthNet（带相机-aware SE）
# -----------------------------
class DepthNet(nn.Module):
    def __init__(self, in_channels, mid_channels, context_channels, depth_channels):
        super(DepthNet, self).__init__()
        self.reduce_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.context_conv = nn.Conv2d(mid_channels, context_channels, kernel_size=1, stride=1, padding=0)
        self.bn = nn.BatchNorm1d(27)
        self.depth_mlp = Mlp(27, mid_channels, mid_channels)
        self.depth_se = SELayer(mid_channels)
        self.context_mlp = Mlp(27, mid_channels, mid_channels)
        self.context_se = SELayer(mid_channels)
        self.depth_conv = nn.Sequential(
            BasicBlock(mid_channels, mid_channels),
            BasicBlock(mid_channels, mid_channels),
            BasicBlock(mid_channels, mid_channels),
            ASPP(mid_channels, mid_channels),
            build_conv_layer(cfg=dict(
                type='DCN', in_channels=mid_channels, out_channels=mid_channels, kernel_size=3,
                padding=1, groups=4, im2col_step=128,
            )),
        )
        self.depth_layer = nn.Conv2d(mid_channels, depth_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x, mats_dict):
        """
        x: [B * N_cam, C_in, Hf, Wf]
        """
        intrins = mats_dict['intrin_mats'][:, 0:1, ..., :3, :3]          # [B,1,N,3,3]
        batch_size = intrins.shape[0]
        num_cams = intrins.shape[2]
        ida = mats_dict['ida_mats'][:, 0:1, ...]                         # [B,1,N,4,4]
        sensor2ego = mats_dict['sensor2ego_mats'][:, 0:1, ..., :3, :]    # [B,1,N,3,4]
        bda = mats_dict['bda_mat'].view(batch_size, 1, 1, 4, 4).repeat(1, 1, num_cams, 1, 1)

        mlp_input = torch.cat(
            [
                torch.stack(
                    [
                        intrins[:, 0:1, ..., 0, 0],
                        intrins[:, 0:1, ..., 1, 1],
                        intrins[:, 0:1, ..., 0, 2],
                        intrins[:, 0:1, ..., 1, 2],
                        ida[:, 0:1, ..., 0, 0],
                        ida[:, 0:1, ..., 0, 1],
                        ida[:, 0:1, ..., 0, 3],
                        ida[:, 0:1, ..., 1, 0],
                        ida[:, 0:1, ..., 1, 1],
                        ida[:, 0:1, ..., 1, 3],
                        bda[:, 0:1, ..., 0, 0],
                        bda[:, 0:1, ..., 0, 1],
                        bda[:, 0:1, ..., 1, 0],
                        bda[:, 0:1, ..., 1, 1],
                        bda[:, 0:1, ..., 2, 2],
                    ],
                    dim=-1,
                ),
                sensor2ego.view(batch_size, 1, num_cams, -1),
            ],
            -1,
        )  # [B,1,N_cam,27]

        # 展平到 [B*N_cam, 27]，过 BN + MLP
        mlp_input = mlp_input.view(batch_size * num_cams, -1)   # [B*N_cam, 27]
        mlp_input = self.bn(mlp_input)                          # [B*N_cam, 27]

        x = self.reduce_conv(x)                                 # [B*N_cam, mid, Hf, Wf]

        # context 分支
        context_se = self.context_mlp(mlp_input).view(batch_size * num_cams, -1, 1, 1)
        context = self.context_se(x, context_se)
        context = self.context_conv(context)

        # depth 分支
        depth_se = self.depth_mlp(mlp_input).view(batch_size * num_cams, -1, 1, 1)
        depth = self.depth_se(x, depth_se)
        depth = self.depth_conv(depth)
        depth = self.depth_layer(depth)

        return torch.cat([depth, context], dim=1)


# -----------------------------
#   HeightNet（原始 BEVHeight 版）
# -----------------------------
class HeightNet(nn.Module):
    def __init__(self, in_channels, mid_channels, context_channels, height_channels):
        super(HeightNet, self).__init__()
        self.reduce_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.context_conv = nn.Conv2d(mid_channels, context_channels, kernel_size=1, stride=1, padding=0)
        self.bn = nn.BatchNorm1d(27)
        self.height_mlp = Mlp(27, mid_channels, mid_channels)
        self.height_se = SELayer(mid_channels)
        self.context_mlp = Mlp(27, mid_channels, mid_channels)
        self.context_se = SELayer(mid_channels)
        self.height_conv = nn.Sequential(
            BasicBlock(mid_channels, mid_channels),
            BasicBlock(mid_channels, mid_channels),
            BasicBlock(mid_channels, mid_channels),
            ASPP(mid_channels, mid_channels),
            build_conv_layer(cfg=dict(
                type='DCN', in_channels=mid_channels, out_channels=mid_channels,
                kernel_size=3, padding=1, groups=4, im2col_step=128,
            )),
        )
        self.height_layer = nn.Conv2d(mid_channels, height_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x, mats_dict):
        """
        x: [B * N_cam, C_in, Hf, Wf]
        """
        intrins = mats_dict['intrin_mats'][:, 0:1, ..., :3, :3]
        batch_size = intrins.shape[0]
        num_cams = intrins.shape[2]
        ida = mats_dict['ida_mats'][:, 0:1, ...]
        sensor2ego = mats_dict['sensor2ego_mats'][:, 0:1, ..., :3, :]
        bda = mats_dict['bda_mat'].view(batch_size, 1, 1, 4, 4).repeat(1, 1, num_cams, 1, 1)

        mlp_input = torch.cat(
            [
                torch.stack(
                    [
                        intrins[:, 0:1, ..., 0, 0],
                        intrins[:, 0:1, ..., 1, 1],
                        intrins[:, 0:1, ..., 0, 2],
                        intrins[:, 0:1, ..., 1, 2],
                        ida[:, 0:1, ..., 0, 0],
                        ida[:, 0:1, ..., 0, 1],
                        ida[:, 0:1, ..., 0, 3],
                        ida[:, 0:1, ..., 1, 0],
                        ida[:, 0:1, ..., 1, 1],
                        ida[:, 0:1, ..., 1, 3],
                        bda[:, 0:1, ..., 0, 0],
                        bda[:, 0:1, ..., 0, 1],
                        bda[:, 0:1, ..., 1, 0],
                        bda[:, 0:1, ..., 1, 1],
                        bda[:, 0:1, ..., 2, 2],
                    ],
                    dim=-1,
                ),
                sensor2ego.view(batch_size, 1, num_cams, -1),
            ],
            -1,
        )  # [B,1,N_cam,27]

        # 展平到 [B*N_cam, 27]，过 BN + MLP
        mlp_input = mlp_input.view(batch_size * num_cams, -1)   # [B*N_cam, 27]
        mlp_input = self.bn(mlp_input)                          # [B*N_cam, 27]

        x = self.reduce_conv(x)                                 # [B*N_cam, mid, Hf, Wf]

        # context 分支
        context_se = self.context_mlp(mlp_input).view(batch_size * num_cams, -1, 1, 1)
        context = self.context_se(x, context_se)
        context = self.context_conv(context)

        # height 分支
        height_se = self.height_mlp(mlp_input).view(batch_size * num_cams, -1, 1, 1)
        height = self.height_se(x, height_se)
        height = self.height_conv(height)
        height = self.height_layer(height)

        return torch.cat([height, context], dim=1)


# -----------------------------
#   BEVSpread LSSFPN backbone
# -----------------------------
class LSSFPN(nn.Module):
    """
    LSSFPN backbone (BEVSpread 版)：
      - 正常前向：返回 BEV 特征（以及可选的 height 概率）
      - 主动学习模式：`return_bin_entropy=True` 时，额外返回 H_img ∈ ℝ[B]
        （来自 key-frame 的 height 概率沿 D 维的分箱熵，空间 & 相机平均）
      - use_spread=True 时，启用 BEVSpread 邻域扩散
    """
    def __init__(self, x_bound, y_bound, z_bound, d_bound, final_dim,
                 downsample_factor, output_channels, img_backbone_conf,
                 img_neck_conf, height_net_conf, use_spread=False, spread_nums=4):
        super(LSSFPN, self).__init__()
        self.downsample_factor = downsample_factor
        self.d_bound = d_bound
        self.final_dim = final_dim
        self.output_channels = output_channels
        self.use_spread = use_spread
        self.spread_nums = spread_nums

        self.register_buffer('voxel_size', torch.Tensor([row[2] for row in [x_bound, y_bound, z_bound]]))
        self.register_buffer('voxel_coord', torch.Tensor([row[0] + row[2] / 2.0 for row in [x_bound, y_bound, z_bound]]))
        self.register_buffer('voxel_num', torch.LongTensor([(row[1] - row[0]) / row[2] for row in [x_bound, y_bound, z_bound]]))
        self.register_buffer('frustum', self.create_frustum())
        self.height_channels, _, _, _ = self.frustum.shape

        self.img_backbone = build_backbone(img_backbone_conf)
        self.img_neck = build_neck(img_neck_conf)
        self.height_net = self._configure_height_net(height_net_conf)

        self.img_neck.init_weights()
        self.img_backbone.init_weights()

        # BEVSpread 衰减因子
        self.beta = nn.Parameter(torch.tensor(1/64), requires_grad=True)

        # 供主动学习选择器自动读取 xyz/d 网格信息
        self.backbone_conf = {
            'x_bound': x_bound,
            'y_bound': y_bound,
            'z_bound': z_bound,
            'd_bound': d_bound,
        }

    # ---- helpers ----
    def _configure_height_net(self, height_net_conf):
        return HeightNet(
            height_net_conf['in_channels'],
            height_net_conf['mid_channels'],
            self.output_channels,
            self.height_channels,
        )

    def create_frustum(self):
        """Generate frustum with non-linear depth spacing (DID)."""
        ogfH, ogfW = self.final_dim
        fH, fW = ogfH // self.downsample_factor, ogfW // self.downsample_factor

        alpha = 1.5
        d_coords = np.arange(self.d_bound[2]) / self.d_bound[2]
        d_coords = np.power(d_coords, alpha)
        d_coords = self.d_bound[0] + d_coords * (self.d_bound[1] - self.d_bound[0])
        d_coords = torch.tensor(d_coords, dtype=torch.float).view(-1, 1, 1).expand(-1, fH, fW)

        D, _, _ = d_coords.shape
        x_coords = torch.linspace(0, ogfW - 1, fW, dtype=torch.float).view(1, 1, fW).expand(D, fH, fW)
        y_coords = torch.linspace(0, ogfH - 1, fH, dtype=torch.float).view(1, fH, 1).expand(D, fH, fW)
        paddings = torch.ones_like(d_coords)

        # D x H x W x 4
        frustum = torch.stack((x_coords, y_coords, d_coords, paddings), -1)
        return frustum

    def height2localtion(self, points, sensor2ego_mat, sensor2virtual_mat, intrin_mat, reference_heights):
        batch_size, num_cams, _, _ = sensor2ego_mat.shape
        reference_heights = reference_heights.view(batch_size, num_cams, 1, 1, 1, 1, 1).repeat(
            1, 1, points.shape[2], points.shape[3], points.shape[4], 1, 1
        )
        height = -1 * points[:, :, :, :, :, 2, :] + reference_heights[:, :, :, :, :, 0, :]

        points_const = points.clone()
        points_const[:, :, :, :, :, 2, :] = 10
        points_const = torch.cat(
            (points_const[:, :, :, :, :, :2] * points_const[:, :, :, :, :, 2:3], points_const[:, :, :, :, :, 2:]), 5)
        combine_virtual = sensor2virtual_mat.matmul(torch.inverse(intrin_mat))
        points_virtual = combine_virtual.view(batch_size, num_cams, 1, 1, 1, 4, 4).matmul(points_const)
        ratio = height[:, :, :, :, :, 0] / points_virtual[:, :, :, :, :, 1, 0]
        ratio = ratio.view(batch_size, num_cams, ratio.shape[2], ratio.shape[3], ratio.shape[4], 1, 1).repeat(
            1, 1, 1, 1, 1, 4, 1)
        points = points_virtual * ratio
        points[:, :, :, :, :, 3, :] = 1
        combine_ego = sensor2ego_mat.matmul(torch.inverse(sensor2virtual_mat))
        points = combine_ego.view(batch_size, num_cams, 1, 1, 1, 4, 4).matmul(points)
        return points

    def get_geometry(self, sensor2ego_mat, sensor2virtual_mat, intrin_mat, ida_mat, reference_heights, bda_mat):
        """Transfer points from camera coord to ego coord, return (xyz, depth)."""
        batch_size, num_cams, _, _ = sensor2ego_mat.shape

        points = self.frustum
        ida_mat = ida_mat.view(batch_size, num_cams, 1, 1, 1, 4, 4)
        points = ida_mat.inverse().matmul(points.unsqueeze(-1))

        coods = points.clone()[:, :, :, :, :, :3, :]
        coods[:, :, :, :, :, 2, :] = 1
        ego2cam = sensor2ego_mat.inverse()
        R = ego2cam[:, :, :3, :3]
        T = ego2cam[:, :, :3, 3:4]
        intrin = intrin_mat[:, :, :3, :3]
        MAT1 = R.inverse().matmul(intrin.inverse()).view(batch_size, num_cams, 1, 1, 1, 3, 3).matmul(coods)
        MAT2 = R.inverse().matmul(T).view(batch_size, num_cams, 1, 1, 1, 3, 1).repeat(
            1, 1, coods.shape[2], coods.shape[3], coods.shape[4], 1, 1)
        depth = ((MAT2[:, :, :, :, :, 2, 0] + points[:, :, :, :, :, 2, 0]) / MAT1[:, :, :, :, :, 2, 0]).view(
            batch_size, num_cams, coods.shape[2], coods.shape[3], coods.shape[4], 1, 1)
        depth = depth.squeeze(-1)  # [B, N, D, Hf, Wf]

        points = self.height2localtion(points, sensor2ego_mat, sensor2virtual_mat, intrin_mat, reference_heights)
        if bda_mat is not None:
            bda_mat = bda_mat.unsqueeze(1).repeat(1, num_cams, 1, 1).view(batch_size, num_cams, 1, 1, 1, 4, 4)
            points = (bda_mat @ points).squeeze(-1)
        else:
            points = points.squeeze(-1)
        return points[..., :3], depth  # xyz, depth

    def get_cam_feats(self, imgs):
        """Get feature maps from images."""
        batch_size, num_sweeps, num_cams, num_channels, imH, imW = imgs.shape
        imgs = imgs.flatten().view(batch_size * num_sweeps * num_cams, num_channels, imH, imW)
        img_feats = self.img_neck(self.img_backbone(imgs))[0]
        img_feats = img_feats.reshape(batch_size, num_sweeps, num_cams,
                                      img_feats.shape[1], img_feats.shape[2], img_feats.shape[3])
        return img_feats

    def _forward_height_net(self, feat, mats_dict):
        return self.height_net(feat, mats_dict)

    def _forward_voxel_net(self, img_feat_with_height):
        return img_feat_with_height

    # 计算图像级的 bin-entropy：对 height 概率沿 D 维求熵，再做空间&相机平均
    @staticmethod
    def _image_bin_entropy_from_height(height_prob: torch.Tensor,
                                       batch_size: int,
                                       num_cams: int,
                                       eps: float = 1e-8) -> torch.Tensor:
        """
        height_prob: [B*num_cams, D, H', W'] (softmax 后)
        return: H_img [B]
        """
        hp = height_prob.clamp_min(eps)
        ent_map = -(hp * hp.log()).sum(dim=1)                 # [B*nc, H', W']
        ent_cam = ent_map.mean(dim=(1, 2))                    # [B*nc]
        H_img = ent_cam.view(batch_size, num_cams).mean(dim=1).contiguous()  # [B]
        return H_img

    def _forward_single_sweep(
        self,
        sweep_index,
        sweep_imgs,
        mats_dict,
        is_return_height: bool = False,
        return_bin_entropy: bool = False,
    ):
        """Forward function for single sweep."""
        batch_size, num_sweeps, num_cams, num_channels, img_height, img_width = sweep_imgs.shape
        img_feats = self.get_cam_feats(sweep_imgs)
        source_features = img_feats[:, 0, ...]  # key-frame feats [B,N,C,Hf,Wf]

        # HeightNet: [B*N, D+C_ctx, Hf, Wf]
        height_feature = self._forward_height_net(
            source_features.reshape(
                batch_size * num_cams,
                source_features.shape[2],
                source_features.shape[3],
                source_features.shape[4],
            ),
            mats_dict,
        )
        height = height_feature[:, :self.height_channels].softmax(1)  # [B*N, D, Hf, Wf]
        context = height_feature[:, self.height_channels:(self.height_channels + self.output_channels)]

        img_feat_with_height = height.unsqueeze(1) * context.unsqueeze(2)
        img_feat_with_height = self._forward_voxel_net(img_feat_with_height)

        img_feat_with_height = img_feat_with_height.reshape(
            batch_size, num_cams,
            img_feat_with_height.shape[1], img_feat_with_height.shape[2],
            img_feat_with_height.shape[3], img_feat_with_height.shape[4],
        )

        geom_xyz, depth = self.get_geometry(
            mats_dict['sensor2ego_mats'][:, sweep_index, ...],
            mats_dict['sensor2virtual_mats'][:, sweep_index, ...],
            mats_dict['intrin_mats'][:, sweep_index, ...],
            mats_dict['ida_mats'][:, sweep_index, ...],
            mats_dict['reference_heights'][:, sweep_index, ...],
            mats_dict.get('bda_mat', None),
        )
        img_feat_with_height = img_feat_with_height.permute(0, 1, 3, 4, 5, 2)  # [B,N,D,Hf,Wf,C]
        geom_xyz = ((geom_xyz - (self.voxel_coord - self.voxel_size / 2.0)) / self.voxel_size)
        geom_xyz_voxel = geom_xyz.int()

        if self.use_spread:
            # BEVSpread 扩散逻辑
            depth_weighted = depth * self.beta  # [B,N,D,Hf,Wf]

            # 2x2 邻域
            geom_xyz_voxels = torch.repeat_interleave(geom_xyz_voxel.unsqueeze(-1), 4, dim=-1)
            geom_xyz_voxels[..., 0, :] += torch.tensor([0, 1, 0, 1], device=geom_xyz_voxels.device)
            geom_xyz_voxels[..., 1, :] += torch.tensor([0, 0, 1, 1], device=geom_xyz_voxels.device)

            dxdy = torch.abs(geom_xyz_voxels - geom_xyz.unsqueeze(-1))
            dist_2 = dxdy[..., 0, :] ** 2 + dxdy[..., 1, :] ** 2

            topk, indices = torch.topk(dist_2, self.spread_nums, dim=-1, largest=False, sorted=True)
            indices = torch.repeat_interleave(indices.unsqueeze(-2), 3, dim=-2)
            geom_xyz_voxels = torch.gather(geom_xyz_voxels, dim=-1, index=indices)

            weight = torch.exp(-topk / depth_weighted)
            img_feat_with_heights = img_feat_with_height.unsqueeze(-1) * weight.unsqueeze(-2)

            img_feat_with_heights = img_feat_with_heights.permute(0, 1, 2, 6, 3, 4, 5).reshape(
                batch_size, num_cams, -1,
                img_feat_with_heights.shape[4], img_feat_with_heights.shape[5], img_feat_with_heights.shape[6]
            )
            geom_xyz_voxels = geom_xyz_voxels.permute(0, 1, 2, 6, 3, 4, 5).reshape(
                batch_size, num_cams, -1,
                geom_xyz_voxels.shape[4], geom_xyz_voxels.shape[5], geom_xyz_voxels.shape[6]
            )
            feature_map = voxel_pooling(geom_xyz_voxels, img_feat_with_heights.contiguous(), self.voxel_num.cuda())
        else:
            feature_map = voxel_pooling(geom_xyz_voxel, img_feat_with_height.contiguous(), self.voxel_num.cuda())

        # --- 主动学习：图像级 bin-entropy（key-frame）
        if return_bin_entropy:
            H_img = self._image_bin_entropy_from_height(height, batch_size, num_cams)  # [B]
            if is_return_height:
                return feature_map.contiguous(), height, H_img
            return feature_map.contiguous(), H_img

        if is_return_height:
            return feature_map.contiguous(), height
        return feature_map.contiguous()

    def forward(
        self,
        sweep_imgs,
        mats_dict,
        timestamps=None,
        is_return_height: bool = False,
        return_bin_entropy: bool = False,
    ):
        """
        Return:
          - 默认: BEV 特征
          - is_return_height=True: (BEV 特征, height 概率)
          - return_bin_entropy=True: (BEV 特征, H_img)
          - 两者都 True: (BEV 特征, height 概率, H_img)   # H_img 始终放在 tuple 最后一位
        """
        batch_size, num_sweeps, num_cams, num_channels, img_height, img_width = sweep_imgs.shape

        # key frame
        key_frame_res = self._forward_single_sweep(
            0,
            sweep_imgs[:, 0:1, ...],
            mats_dict,
            is_return_height=is_return_height,
            return_bin_entropy=return_bin_entropy,
        )

        if is_return_height:
            if return_bin_entropy:
                key_frame_feature, key_frame_height, H_img = key_frame_res
            else:
                key_frame_feature, key_frame_height = key_frame_res
        else:
            if return_bin_entropy:
                key_frame_feature, H_img = key_frame_res
            else:
                key_frame_feature = key_frame_res

        if num_sweeps == 1:
            if is_return_height:
                return (key_frame_feature, key_frame_height, H_img) if return_bin_entropy else (key_frame_feature, key_frame_height)
            else:
                return (key_frame_feature, H_img) if return_bin_entropy else key_frame_feature

        # 其他 sweeps 只用于 BEV 特征累积，不算 AL signal
        ret_feature_list = [key_frame_feature]
        for sweep_index in range(1, num_sweeps):
            with torch.no_grad():
                feature_map = self._forward_single_sweep(
                    sweep_index,
                    sweep_imgs[:, sweep_index:sweep_index + 1, ...],
                    mats_dict,
                    is_return_height=False,
                    return_bin_entropy=False,
                )
                ret_feature_list.append(feature_map)

        feat_all = torch.cat(ret_feature_list, 1)
        if is_return_height:
            return (feat_all, key_frame_height, H_img) if return_bin_entropy else (feat_all, key_frame_height)
        else:
            return (feat_all, H_img) if return_bin_entropy else feat_all
