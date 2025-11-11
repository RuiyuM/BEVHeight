# Copyright (c) Megvii Inc. All rights reserved.
import numpy as np
from vggt_c5_injector import VGGT_C5_Injector  # 保留以避免外部导入影响
import torch
import torch.nn.functional as F
from mmcv.cnn import build_conv_layer
from mmdet3d.models import build_neck
from mmdet.models import build_backbone
from mmdet.models.backbones.resnet import BasicBlock
from torch import nn

from ops.voxel_pooling import voxel_pooling

__all__ = ['LSSFPN']


# --------------------------
# 简洁的 BEVDet 风格 DepthNet
# --------------------------
class DepthNet(nn.Module):
    """
    轻量深度+上下文双头（BEVDet 风格）：
    输入: [B*N, C_in, Hf, Wf]
    输出: cat([depth_logits: D, context_feats: C_ctx], dim=1)
    """
    def __init__(self, in_channels, mid_channels, context_channels, depth_channels):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        # 深度分类头
        self.depth_head = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, depth_channels, 1, bias=True),
        )
        # 上下文特征头（送入体素融合）
        self.context_head = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, context_channels, 1, bias=True),
        )

        # Kaiming 初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.)
                m.bias.data.zero_()

    def forward(self, x, _mats_dict_ignored=None):
        x = self.stem(x)
        depth_logits = self.depth_head(x)      # [B*N, D, Hf, Wf]
        context = self.context_head(x)         # [B*N, C_ctx, Hf, Wf]
        return torch.cat([depth_logits, context], dim=1)


class LSSFPN(nn.Module):
    def __init__(self, x_bound, y_bound, z_bound, d_bound, final_dim,
                 downsample_factor, output_channels, img_backbone_conf,
                 img_neck_conf, height_net_conf):
        """Modified towards BEVDet-style LSS view transformer.

        Args:
            x_bound (list): [x_min, x_max, x_step]
            y_bound (list): [y_min, y_max, y_step]
            z_bound (list): [z_min, z_max, z_step]
            d_bound (list): [d_min, d_max, num_bins]  # 注意第三项为bins数
            final_dim (list): 输入图像尺寸 [H, W]
            downsample_factor (int): 下采样率(输入图到特征图)
            output_channels (int): 体素融合前通道数 (context_channels)
            img_backbone_conf (dict): 图像backbone配置
            img_neck_conf (dict): FPN/neck配置
            height_net_conf (dict): 这里仍沿用名字，但会用于DepthNet
        """
        super(LSSFPN, self).__init__()
        self.downsample_factor = downsample_factor
        self.d_bound = d_bound
        self.final_dim = final_dim
        self.output_channels = output_channels

        # 体素网格参数
        self.register_buffer(
            'voxel_size',
            torch.Tensor([row[2] for row in [x_bound, y_bound, z_bound]]))
        self.register_buffer(
            'voxel_coord',
            torch.Tensor([row[0] + row[2] / 2.0 for row in [x_bound, y_bound, z_bound]]))
        self.register_buffer(
            'voxel_num',
            torch.LongTensor([(row[1] - row[0]) / row[2] for row in [x_bound, y_bound, z_bound]]))

        # 视锥体 (线性深度采样，BEVDet 风格)
        self.register_buffer('frustum', self.create_frustum())
        self.height_channels, _, _, _ = self.frustum.shape  # D

        # 图像主干与颈部
        self.img_backbone = build_backbone(img_backbone_conf)
        self.img_neck = build_neck(img_neck_conf)

        # 用 DepthNet 取代原 HeightNet（仍复用 height_net_conf 字段）
        self.height_net = self._configure_height_net(height_net_conf)

        self.img_neck.init_weights()
        self.img_backbone.init_weights()

    def _configure_height_net(self, height_net_conf):
        # 将原 HeightNet 的 in/mid 配置沿用给 DepthNet
        return DepthNet(
            height_net_conf['in_channels'],
            height_net_conf['mid_channels'],
            self.output_channels,
            self.height_channels,  # depth bins
        )

    # --------------------------
    # BEVDet 风格的视锥体构建
    # --------------------------
    def create_frustum(self):
        """Generate frustum: [D, fH, fW, 4] with (u, v, d, 1)."""
        ogfH, ogfW = self.final_dim
        fH, fW = ogfH // self.downsample_factor, ogfW // self.downsample_factor

        # 线性深度划分（BEVDet 常用）
        D = int(self.d_bound[2])
        d_min, d_max = float(self.d_bound[0]), float(self.d_bound[1])
        d_coords = torch.linspace(d_min, d_max, D, dtype=torch.float32).view(-1, 1, 1).expand(-1, fH, fW)

        x_coords = torch.linspace(0, ogfW - 1, fW, dtype=torch.float32).view(1, 1, fW).expand(D, fH, fW)
        y_coords = torch.linspace(0, ogfH - 1, fH, dtype=torch.float32).view(1, fH, 1).expand(D, fH, fW)
        paddings = torch.ones_like(d_coords)

        # D x H x W x 4  (u, v, d, 1)
        frustum = torch.stack((x_coords, y_coords, d_coords, paddings), dim=-1)
        return frustum

    # --------------------------
    # BEVDet/LSS 几何（不依赖高度平面）
    # --------------------------
    @torch.no_grad()
    def get_geometry(self, sensor2ego_mat, sensor2virtual_mat, intrin_mat, ida_mat, reference_heights, bda_mat):
        """
        将像素+深度投影到ego系的3D点：
            uv1 --IDA^-1--> uv1(pre) --K^-1--> ray_cam --*d--> p_cam --T_cam2ego--> p_ego --(BDA)--> p_ego'
        NOTE: 保留形参以兼容外部调用，但不再使用 sensor2virtual_mat/reference_heights。
        返回: [B, N, D, fH, fW, 3]
        """
        device = intrin_mat.device
        dtype = intrin_mat.dtype

        B, N, _, _ = sensor2ego_mat.shape
        D, fH, fW, _ = self.frustum.shape

        # 准备 uv1 与 d
        frustum = self.frustum.to(device=device, dtype=dtype)
        # uv1: [D, H, W, 3]
        uv1 = torch.stack(
            [frustum[..., 0], frustum[..., 1], torch.ones_like(frustum[..., 0])],
            dim=-1
        )

        # reshape & broadcast: [B, N, D, H, W, 3, 1]
        uv1 = uv1.view(1, 1, D, fH, fW, 3, 1).repeat(B, N, 1, 1, 1, 1, 1)
        d = frustum[..., 2].view(1, 1, D, fH, fW, 1, 1)

        # IDA^-1 (3x3 + 平移)，将增强后的像素坐标还原
        ida_inv = torch.inverse(ida_mat)  # [B, N, 4, 4]
        R_ida = ida_inv[..., :3, :3].view(B, N, 1, 1, 1, 3, 3)
        t_ida = ida_inv[..., :3, 3].view(B, N, 1, 1, 1, 3, 1)
        uv1_pre = R_ida @ uv1 + t_ida  # [B, N, D, H, W, 3, 1]

        # K^-1
        K_inv = torch.inverse(intrin_mat[..., :3, :3]).view(B, N, 1, 1, 1, 3, 3)
        rays_cam = K_inv @ uv1_pre  # [B, N, D, H, W, 3, 1]

        # 乘深度得到相机坐标点
        points_cam = rays_cam * d.to(dtype=dtype, device=device)  # [..., 3, 1]
        ones = torch.ones_like(points_cam[..., :1, :])
        points_cam_4 = torch.cat([points_cam, ones], dim=-2)  # [..., 4, 1]

        # 相机到自车
        T = sensor2ego_mat.view(B, N, 1, 1, 1, 4, 4)
        points_ego = T @ points_cam_4  # [..., 4, 1]

        # BDA（可选）
        if bda_mat is not None:
            BDA = bda_mat.view(B, 1, 1, 1, 1, 4, 4)
            points_ego = BDA @ points_ego

        points_ego = points_ego.squeeze(-1)[..., :3]  # [B, N, D, H, W, 3]
        return points_ego

    # --------------------------
    # 常规图像特征
    # --------------------------
    def get_cam_feats(self, imgs):
        """Get feature maps from images."""
        B, S, N, C, H, W = imgs.shape
        imgs = imgs.flatten().view(B * S * N, C, H, W)
        img_feats = self.img_neck(self.img_backbone(imgs))[0]
        img_feats = img_feats.reshape(B, S, N, img_feats.shape[1], img_feats.shape[2], img_feats.shape[3])
        return img_feats

    def _forward_height_net(self, feat, mats_dict):
        # 兼容原接口；内部已改为 BEVDet-style DepthNet
        return self.height_net(feat, mats_dict)

    def _forward_voxel_net(self, img_feat_with_depth):
        # 与 BEVDet 一致：直接体素池化前的特征，不做额外处理
        return img_feat_with_depth

    # --------------------------
    # 单帧（key-frame）前向
    # --------------------------
    def _forward_single_sweep(self,
                              sweep_index,
                              sweep_imgs,
                              mats_dict,
                              is_return_height=False,
                              return_bin_entropy: bool = False,
                              return_depth_profile=False, depth_profile_mode="hard",
                              depth_profile_weight="none"):
        """
        BEVDet 风格：深度分布 * 上下文特征 -> 体素池化
        仍保留 AL 的可选返回，便于兼容，但实现不再依赖任何 AL 模块。
        """
        B, S, N, C, H, W = sweep_imgs.shape

        # 1) 图像特征
        img_feats = self.get_cam_feats(sweep_imgs)            # [B,S,N,Cf,Hf,Wf]
        source_features = img_feats[:, 0, ...]                # [B,N,Cf,Hf,Wf]

        # 2) 深度+上下文（BEVDet 风格）
        depth_context = self._forward_height_net(
            source_features.reshape(B * N, source_features.shape[2], source_features.shape[3], source_features.shape[4]),
            mats_dict,
        )                                                     # [B*N, D+C_ctx, Hf, Wf]
        D = self.height_channels
        depth_logits = depth_context[:, :D]                   # [B*N, D, Hf, Wf]
        context = depth_context[:, D:(D + self.output_channels)]  # [B*N, C_ctx, Hf, Wf]
        depth_prob = depth_logits.softmax(dim=1)              # 概率化

        # 可选: 信息熵/深度profile（兼容外部接口需求）
        if return_bin_entropy:
            eps = 1e-12
            H_map = -(depth_prob * (depth_prob.clamp_min(eps).log())).sum(dim=1) / float(np.log(D))
            H_cam = H_map.mean(dim=(1, 2))                    # [B*N]
            H_img = H_cam.view(B, N).mean(dim=1)              # [B]
        else:
            H_img = None

        if return_depth_profile:
            _, D, fH, fW = depth_prob.shape
            if depth_profile_mode == "hard":
                top1_prob, top1_idx = depth_prob.max(dim=1)   # [B*N, Hf, Wf]
                if depth_profile_weight == "top1":
                    w = top1_prob.reshape(B * N, -1)
                elif depth_profile_weight == "margin":
                    top2_prob = depth_prob.topk(k=2, dim=1).values[:, 1, :, :]
                    w = (top1_prob - top2_prob).clamp_min(0).reshape(B * N, -1)
                else:
                    w = torch.ones(B * N, fH * fW, device=depth_prob.device, dtype=depth_prob.dtype)
                idx_flat = top1_idx.reshape(B * N, -1)
                m_flat = torch.zeros(B * N, D, device=depth_prob.device, dtype=depth_prob.dtype)
                m_flat.scatter_add_(1, idx_flat, w)
                m_img = m_flat.view(B, N, D).sum(dim=1)       # [B, D]
            else:
                m_cam = depth_prob.mean(dim=(2, 3))           # [B*N, D]
                m_img = m_cam.view(B, N, D).mean(dim=1)       # [B, D]
            m_depth_img = m_img / (m_img.sum(dim=1, keepdim=True).clamp_min(1e-6))
        else:
            m_depth_img = None

        # 3) 外积融合 (B*N, C_ctx, D, Hf, Wf)
        img_feat_with_depth = depth_prob.unsqueeze(1) * context.unsqueeze(2)
        img_feat_with_depth = self._forward_voxel_net(img_feat_with_depth)

        # 4) 几何 (BEVDet/LSS)
        img_feat_with_depth = img_feat_with_depth.reshape(B, N,
                                                          img_feat_with_depth.shape[1],
                                                          img_feat_with_depth.shape[2],
                                                          img_feat_with_depth.shape[3],
                                                          img_feat_with_depth.shape[4])  # [B,N,C,D,Hf,Wf]

        geom_xyz = self.get_geometry(
            mats_dict['sensor2ego_mats'][:, sweep_index, ...],
            mats_dict.get('sensor2virtual_mats', None),  # 兼容键，内部不会使用
            mats_dict['intrin_mats'][:, sweep_index, ...],
            mats_dict['ida_mats'][:, sweep_index, ...],
            mats_dict.get('reference_heights', None),
            mats_dict.get('bda_mat', None),
        )                                                     # [B,N,D,Hf,Wf,3]

        # 5) 体素池化到 BEV
        feats_for_pool = img_feat_with_depth.permute(0, 1, 3, 4, 5, 2).contiguous()  # [B,N,D,Hf,Wf,C]
        # 量化到体素索引
        geom_xyz = ((geom_xyz - (self.voxel_coord - self.voxel_size / 2.0)) / self.voxel_size).int()
        feature_map = voxel_pooling(geom_xyz, feats_for_pool, self.voxel_num.to(geom_xyz.device))

        # 6) 兼容不同返回模式
        if return_bin_entropy and return_depth_profile:
            return feature_map.contiguous(), H_img, m_depth_img
        if is_return_height and return_bin_entropy:
            return feature_map.contiguous(), depth_prob, H_img
        if is_return_height:
            return feature_map.contiguous(), depth_prob
        if return_bin_entropy:
            return feature_map.contiguous(), H_img
        return feature_map.contiguous()

    def forward(self,
                sweep_imgs,
                mats_dict,
                timestamps=None,
                is_return_height: bool = False,
                return_bin_entropy: bool = False,
                return_depth_profile: bool = False,
                depth_profile_mode: str = "hard",
                depth_profile_weight: str = "none"):
        # 只用 key-frame（index 0），与原逻辑一致
        out = self._forward_single_sweep(
            0,
            sweep_imgs[:, 0:1, ...],  # (B,1,N,C,H,W)
            mats_dict,
            is_return_height=is_return_height,
            return_bin_entropy=return_bin_entropy,
            return_depth_profile=return_depth_profile,
            depth_profile_mode=depth_profile_mode,
            depth_profile_weight=depth_profile_weight,
        )
        return out
