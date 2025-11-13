# Copyright (c) Megvii Inc. All rights reserved.
import numpy as np
from vggt_c5_injector import VGGT_C5_Injector
import torch
import torch.nn.functional as F
from mmcv.cnn import build_conv_layer
from mmdet3d.models import build_neck
from mmdet.models import build_backbone
from mmdet.models.backbones.resnet import BasicBlock
from torch import nn

from ops.voxel_pooling import voxel_pooling

__all__ = ['LSSFPN']


class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation,
                 BatchNorm):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes,
                                     planes,
                                     kernel_size=kernel_size,
                                     stride=1,
                                     padding=padding,
                                     dilation=dilation,
                                     bias=False)
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

        self.aspp1 = _ASPPModule(inplanes,
                                 mid_channels,
                                 1,
                                 padding=0,
                                 dilation=dilations[0],
                                 BatchNorm=BatchNorm)
        self.aspp2 = _ASPPModule(inplanes,
                                 mid_channels,
                                 3,
                                 padding=dilations[1],
                                 dilation=dilations[1],
                                 BatchNorm=BatchNorm)
        self.aspp3 = _ASPPModule(inplanes,
                                 mid_channels,
                                 3,
                                 padding=dilations[2],
                                 dilation=dilations[2],
                                 BatchNorm=BatchNorm)
        self.aspp4 = _ASPPModule(inplanes,
                                 mid_channels,
                                 3,
                                 padding=dilations[3],
                                 dilation=dilations[3],
                                 BatchNorm=BatchNorm)

        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(inplanes, mid_channels, 1, stride=1, bias=False),
            BatchNorm(mid_channels),
            nn.ReLU(),
        )
        self.conv1 = nn.Conv2d(int(mid_channels * 5),
                               mid_channels,
                               1,
                               bias=False)
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
        x5 = F.interpolate(x5,
                           size=x4.size()[2:],
                           mode='bilinear',
                           align_corners=True)
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
    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.ReLU,
                 drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class SELayer(nn.Module):
    def __init__(self, channels, act_layer=nn.ReLU, gate_layer=nn.Sigmoid):
        super().__init__()
        self.conv_reduce = nn.Conv2d(channels, channels, 1, bias=True)
        self.act1 = act_layer()
        self.conv_expand = nn.Conv2d(channels, channels, 1, bias=True)
        self.gate = gate_layer()

    def forward(self, x, x_se):
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        return x * self.gate(x_se)


# --------- 新的 LSS 风格 DepthNet（替代原 HeightNet，仅结构变化，接口保持不变）---------
class DepthNet(nn.Module):
    """
    LSS-style depth + context head:
      - 输入: image feature map
      - 输出: [B, D + C, H, W]，前 D 个通道为 depth logits，后 C 个通道为 context 特征

    与原 HeightNet 不同的是：
      * 不再用 camera-aware MLP / SELayer / ASPP
      * 仅使用 Conv-BN-ReLU 堆叠，尽量贴近 Lift, Splat, Shoot 的 DepthNet 形式
    """

    def __init__(self, in_channels, mid_channels, depth_channels, context_channels):
        super().__init__()
        self.depth_channels = depth_channels
        self.context_channels = context_channels

        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)

        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)

        # 输出 D + C 个通道：前 D 为 depth logits，后 C 为 BEV context 特征
        self.conv_out = nn.Conv2d(mid_channels,
                                  depth_channels + context_channels,
                                  kernel_size=1,
                                  padding=0,
                                  bias=True)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, mats_dict=None):
        """
        输入:
            x: [B, C_in, H, W]
            mats_dict: 为了接口兼容保留，但在 LSS 风格 DepthNet 中不使用
        输出:
            feat: [B, D + C, H, W]
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv_out(x)
        return x


class LSSFPN(nn.Module):
    def __init__(self, x_bound, y_bound, z_bound, d_bound, final_dim,
                 downsample_factor, output_channels, img_backbone_conf,
                 img_neck_conf, height_net_conf):
        """Modified from `https://github.com/nv-tlabs/lift-splat-shoot`.

        Args:
            x_bound (list): Boundaries for x.
            y_bound (list): Boundaries for y.
            z_bound (list): Boundaries for z.
            d_bound (list): Boundaries for d.
            final_dim (list): Dimension for input images.
            downsample_factor (int): Downsample factor between feature map
                and input image.
            output_channels (int): Number of channels for the output
                feature map.
            img_backbone_conf (dict): Config for image backbone.
            img_neck_conf (dict): Config for image neck.
            height_net_conf (dict): Config for depth/height net.
        """

        super(LSSFPN, self).__init__()
        self.downsample_factor = downsample_factor
        self.d_bound = d_bound
        self.final_dim = final_dim
        self.output_channels = output_channels

        # voxel grid
        self.register_buffer(
            'voxel_size',
            torch.Tensor([row[2] for row in [x_bound, y_bound, z_bound]]))
        self.register_buffer(
            'voxel_coord',
            torch.Tensor([
                row[0] + row[2] / 2.0 for row in [x_bound, y_bound, z_bound]
            ]))
        self.register_buffer(
            'voxel_num',
            torch.LongTensor([(row[1] - row[0]) / row[2]
                              for row in [x_bound, y_bound, z_bound]]))

        # LSS-style frustum (均匀深度)
        self.register_buffer('frustum', self.create_frustum())
        self.height_channels, _, _, _ = self.frustum.shape  # 这里的 "height_channels" 实际上就是 depth_bins D

        self.img_backbone = build_backbone(img_backbone_conf)
        self.img_neck = build_neck(img_neck_conf)
        self.height_net = self._configure_height_net(height_net_conf)

        self.img_neck.init_weights()
        self.img_backbone.init_weights()

    def _configure_height_net(self, height_net_conf):
        # 使用新的 DepthNet，保持接口: forward(feat, mats_dict) -> [B, D+C, H, W]
        return DepthNet(
            in_channels=height_net_conf['in_channels'],
            mid_channels=height_net_conf['mid_channels'],
            depth_channels=self.height_channels,
            context_channels=self.output_channels,
        )

    def create_frustum(self):
        """Generate frustum (LSS-style: uniform metric depth bins)."""
        # image plane grid
        ogfH, ogfW = self.final_dim
        fH, fW = ogfH // self.downsample_factor, ogfW // self.downsample_factor

        # depth bins: [d_min, d_max], num_bins = d_bound[2]
        d_min, d_max, d_num = self.d_bound
        d_num = int(d_num)

        # 均匀深度采样：与 LSS 一致使用 metric depth 等间距
        depth_values = torch.linspace(d_min, d_max, d_num, dtype=torch.float32)
        d_coords = depth_values.view(-1, 1, 1).expand(-1, fH, fW)  # [D, fH, fW]

        D, _, _ = d_coords.shape

        # 像素坐标网格
        x_coords = torch.linspace(0, ogfW - 1, fW, dtype=torch.float32).view(
            1, 1, fW).expand(D, fH, fW)
        y_coords = torch.linspace(0, ogfH - 1, fH,
                                  dtype=torch.float32).view(1, fH,
                                                            1).expand(D, fH, fW)
        paddings = torch.ones_like(d_coords)

        # D x H x W x 4 (u, v, depth, 1)
        frustum = torch.stack((x_coords, y_coords, d_coords, paddings), -1)
        return frustum

    def height2localtion(self, points, sensor2ego_mat, sensor2virtual_mat, intrin_mat, reference_heights):
        # 保持原 BEVHeight 几何逻辑不变，以确保仅靠本文件就能跑通
        batch_size, num_cams, _, _ = sensor2ego_mat.shape
        reference_heights = reference_heights.view(batch_size, num_cams, 1, 1, 1, 1,
                                                   1).repeat(1, 1, points.shape[2], points.shape[3], points.shape[4], 1,
                                                             1)
        height = -1 * points[:, :, :, :, :, 2, :] + reference_heights[:, :, :, :, :, 0, :]

        points_const = points.clone()
        points_const[:, :, :, :, :, 2, :] = 10
        points_const = torch.cat(
            (points_const[:, :, :, :, :, :2] * points_const[:, :, :, :, :, 2:3],
             points_const[:, :, :, :, :, 2:]), 5)
        combine_virtual = sensor2virtual_mat.matmul(torch.inverse(intrin_mat))
        points_virtual = combine_virtual.view(batch_size, num_cams, 1, 1, 1, 4, 4).matmul(points_const)
        ratio = height[:, :, :, :, :, 0] / points_virtual[:, :, :, :, :, 1, 0]
        ratio = ratio.view(batch_size, num_cams, ratio.shape[2], ratio.shape[3], ratio.shape[4], 1, 1).repeat(1, 1, 1,
                                                                                                              1, 1, 4,
                                                                                                              1)
        points = points_virtual * ratio
        points[:, :, :, :, :, 3, :] = 1
        combine_ego = sensor2ego_mat.matmul(torch.inverse(sensor2virtual_mat))
        points = combine_ego.view(batch_size, num_cams, 1, 1, 1, 4,
                                  4).matmul(points)
        return points

    def get_geometry(self, sensor2ego_mat, sensor2virtual_mat, intrin_mat, ida_mat, reference_heights, bda_mat):
        """Transfer points from camera coord to ego coord.

        Args:
            sensor2ego_mat(Tensor): (B, num_cams, 4, 4).
            intrin_mat(Tensor): (B, num_cams, 4, 4).
            ida_mat(Tensor): (B, num_cams, 4, 4).
            bda_mat(Tensor or None): (B, 4, 4).

        Returns:
            Tensors: points in ego coord, shape (B, num_cams, D, fH, fW, 3).
        """
        batch_size, num_cams, _, _ = sensor2ego_mat.shape

        # undo ida
        points = self.frustum  # [D, fH, fW, 4]
        ida_mat = ida_mat.view(batch_size, num_cams, 1, 1, 1, 4, 4)
        points = ida_mat.inverse().matmul(points.unsqueeze(-1))

        # 使用原始 BEVHeight 的 height2localtion 几何变换
        points = self.height2localtion(points, sensor2ego_mat, sensor2virtual_mat, intrin_mat, reference_heights)

        if bda_mat is not None:
            bda_mat = bda_mat.unsqueeze(1).repeat(1, num_cams, 1, 1).view(
                batch_size, num_cams, 1, 1, 1, 4, 4)
            points = (bda_mat @ points).squeeze(-1)
        else:
            points = points.squeeze(-1)
        return points[..., :3]

    def get_cam_feats(self, imgs):
        """Get feature maps from images."""
        batch_size, num_sweeps, num_cams, num_channels, imH, imW = imgs.shape

        imgs = imgs.flatten().view(batch_size * num_sweeps * num_cams,
                                   num_channels, imH, imW)
        img_feats = self.img_neck(self.img_backbone(imgs))[0]
        img_feats = img_feats.reshape(batch_size, num_sweeps, num_cams,
                                      img_feats.shape[1], img_feats.shape[2],
                                      img_feats.shape[3])
        return img_feats

    def _forward_height_net(self, feat, mats_dict):
        # 这里的 height_net 已经是 LSS 风格 DepthNet，但保持相同调用方式
        return self.height_net(feat, mats_dict)

    def _forward_voxel_net(self, img_feat_with_height):
        # 这里保持为 identity，方便以后在 BEV 空间上再接 neck / head
        return img_feat_with_height

    def _forward_single_sweep(self,
                              sweep_index,
                              sweep_imgs,
                              mats_dict,
                              is_return_height=False,
                              return_bin_entropy: bool = False,
                              return_depth_profile=False, depth_profile_mode="hard",
                              depth_profile_weight="none"
                              ):
        """Forward function for single sweep.

        Args:
            sweep_index (int): Index of sweeps.
            sweep_imgs (Tensor): Input images.
            mats_dict (dict): camera & augmentation matrices.
            is_return_height (bool, optional): Whether to return depth/height volume.

        Returns:
            Tensor or tuple: BEV feature map (+ 可选的 height, H_img, m_depth_img).
        """
        batch_size, num_sweeps, num_cams, num_channels, img_height, \
        img_width = sweep_imgs.shape
        img_feats = self.get_cam_feats(sweep_imgs)
        source_features = img_feats[:, 0, ...]  # [B, num_cams, C, fH, fW]

        # -------- LSS 风格 depth + context 预测 --------
        height_feature = self._forward_height_net(
            source_features.reshape(batch_size * num_cams,
                                    source_features.shape[2],
                                    source_features.shape[3],
                                    source_features.shape[4]),
            mats_dict,
        )  # [B*num_cams, D+C, fH, fW]

        height = height_feature[:, :self.height_channels].softmax(1)  # [B*num_cams, D, fH, fW]

        # ---- 计算 H_img（与现在相同）----
        if return_bin_entropy:
            eps = 1e-12
            D = height.shape[1]
            H_map = -(height * (height.clamp_min(eps).log())).sum(dim=1) / float(np.log(D))  # [B*num_cams, fH, fW]
            H_cam = H_map.mean(dim=(1, 2))  # [B*num_cams]
            B = sweep_imgs.shape[0]
            num_cams = sweep_imgs.shape[2]
            H_img = H_cam.view(B, num_cams).mean(dim=1)  # [B]
        else:
            H_img = None

        # ---- 计算 m_depth_img: [B, D] ----
        if return_depth_profile:
            B = sweep_imgs.shape[0]
            num_cams = sweep_imgs.shape[2]
            _, D, fH, fW = height.shape

            if depth_profile_mode == "hard":
                top1_prob, top1_idx = height.max(dim=1)  # [B*num_cams, fH, fW]
                if depth_profile_weight == "top1":
                    w = top1_prob.reshape(B * num_cams, -1)  # [B*num_cams, fH*fW]
                elif depth_profile_weight == "margin":
                    top2_prob = height.topk(k=2, dim=1).values[:, 1, :, :]  # [B*num_cams, fH, fW]
                    w = (top1_prob - top2_prob).clamp_min(0).reshape(B * num_cams, -1)
                else:
                    w = torch.ones(B * num_cams, fH * fW, device=height.device, dtype=height.dtype)

                idx_flat = top1_idx.reshape(B * num_cams, -1)  # [L, Npix]
                m_flat = torch.zeros(B * num_cams, D, device=height.device, dtype=height.dtype)
                m_flat.scatter_add_(1, idx_flat, w)  # 按 top-1 累加权重
                m_img = m_flat.view(B, num_cams, D).sum(dim=1)  # 相机上求和 -> [B, D]
            else:
                # soft-mode：空间平均概率
                m_cam = height.mean(dim=(2, 3))  # [B*num_cams, D]
                m_img = m_cam.view(B, num_cams, D).mean(dim=1)  # [B, D]

            m_depth_img = m_img / (m_img.sum(dim=1, keepdim=True).clamp_min(1e-6))  # L1 归一
        else:
            m_depth_img = None

        # -------- Lift: depth × context -> 3D features --------
        img_feat_with_height = height.unsqueeze(
            1) * height_feature[:, self.height_channels:(
                self.height_channels + self.output_channels)].unsqueeze(2)
        img_feat_with_height = self._forward_voxel_net(img_feat_with_height)

        img_feat_with_height = img_feat_with_height.reshape(
            batch_size,
            num_cams,
            img_feat_with_height.shape[1],
            img_feat_with_height.shape[2],
            img_feat_with_height.shape[3],
            img_feat_with_height.shape[4],
        )

        geom_xyz = self.get_geometry(
            mats_dict['sensor2ego_mats'][:, sweep_index, ...],
            mats_dict['sensor2virtual_mats'][:, sweep_index, ...],
            mats_dict['intrin_mats'][:, sweep_index, ...],
            mats_dict['ida_mats'][:, sweep_index, ...],
            mats_dict['reference_heights'][:, sweep_index, ...],
            mats_dict.get('bda_mat', None),
        )
        img_feat_with_height = img_feat_with_height.permute(0, 1, 3, 4, 5, 2)
        geom_xyz = ((geom_xyz - (self.voxel_coord - self.voxel_size / 2.0)) /
                    self.voxel_size).int()

        feature_map = voxel_pooling(geom_xyz, img_feat_with_height.contiguous(),
                                    self.voxel_num.cuda())

        # -------- 输出保持与你之前一致，方便 active learning 调用 --------
        if return_bin_entropy and return_depth_profile:
            return feature_map.contiguous(), H_img, m_depth_img
        if is_return_height and return_bin_entropy:
            return (feature_map.contiguous(), height, H_img)
        if is_return_height:
            return feature_map.contiguous(), height
        if return_bin_entropy:
            return feature_map.contiguous(), H_img
        return feature_map.contiguous()

    def forward(
            self,
            sweep_imgs,
            mats_dict,
            timestamps=None,
            is_return_height: bool = False,
            return_bin_entropy: bool = False,
            return_depth_profile: bool = False,
            depth_profile_mode: str = "hard",
            depth_profile_weight: str = "none",
    ):
        # 只跑 key-frame（index 0）
        out = self._forward_single_sweep(
            0,
            sweep_imgs[:, 0:1, ...],  # (B,1,num_cams,C,H,W)
            mats_dict,
            is_return_height=is_return_height,
            return_bin_entropy=return_bin_entropy,
            return_depth_profile=return_depth_profile,
            depth_profile_mode=depth_profile_mode,
            depth_profile_weight=depth_profile_weight,
        )
        # 直接透传 _forward_single_sweep 的返回
        return out
