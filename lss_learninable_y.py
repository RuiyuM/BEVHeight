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


class HeightNet(nn.Module):
    def __init__(self, in_channels, mid_channels, context_channels,
                 height_channels):
        super(HeightNet, self).__init__()
        self.reduce_conv = nn.Sequential(
            nn.Conv2d(in_channels,
                      mid_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.context_conv = nn.Conv2d(mid_channels,
                                      context_channels,
                                      kernel_size=1,
                                      stride=1,
                                      padding=0)
        self.bn = nn.BatchNorm1d(27)
        self.height_mlp = Mlp(27, mid_channels, mid_channels)
        self.height_se = SELayer(mid_channels)  # NOTE: add camera-aware
        self.context_mlp = Mlp(27, mid_channels, mid_channels)
        self.context_se = SELayer(mid_channels)  # NOTE: add camera-aware
        self.height_conv = nn.Sequential(
            BasicBlock(mid_channels, mid_channels),
            BasicBlock(mid_channels, mid_channels),
            BasicBlock(mid_channels, mid_channels),
            ASPP(mid_channels, mid_channels),
            build_conv_layer(cfg=dict(
                type='DCN',
                in_channels=mid_channels,
                out_channels=mid_channels,
                kernel_size=3,
                padding=1,
                groups=4,
                im2col_step=128,
            )),

        )
        self.height_layer = nn.Conv2d(mid_channels,
                                      height_channels,
                                      kernel_size=1,
                                      stride=1,
                                      padding=0)

    def forward(self, x, mats_dict):
        intrins = mats_dict['intrin_mats'][:, 0:1, ..., :3, :3]
        batch_size = intrins.shape[0]
        num_cams = intrins.shape[2]
        ida = mats_dict['ida_mats'][:, 0:1, ...]
        sensor2ego = mats_dict['sensor2ego_mats'][:, 0:1, ..., :3, :]
        bda = mats_dict['bda_mat'].view(batch_size, 1, 1, 4,
                                        4).repeat(1, 1, num_cams, 1, 1)
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
        )
        mlp_input = self.bn(mlp_input.reshape(-1, mlp_input.shape[-1]))
        x = self.reduce_conv(x)
        context_se = self.context_mlp(mlp_input)[..., None, None]
        context = self.context_se(x, context_se)
        context = self.context_conv(context)
        height_se = self.height_mlp(mlp_input)[..., None, None]
        height = self.height_se(x, height_se)
        height = self.height_conv(height)
        height = self.height_layer(height)
        return torch.cat([height, context], dim=1)


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
            height_net_conf (dict): Config for height net.
        """

        super(LSSFPN, self).__init__()
        self.downsample_factor = downsample_factor
        self.d_bound = d_bound
        self.final_dim = final_dim
        self.output_channels = output_channels

        self.use_ipm = True
        self.use_ida_in_H = False  # 你的可视化表明无需乘 IDA
        self.align_corners = True
        self._init_bev_grid(x_bound, y_bound, True)  # 或者给个 beta=2.0 试远密近疏
        self.adapter = None
        self.learn_y = True
        self.apply_bev_mask = True  # 开：把可见扇形mask乘到BEV特征上
        self.mask_dilate_ks = 5  # 可见区做一点膨胀，避免边缘漏掉（奇数：3/5/7）
        self.mask_blur_ks = 0  # 若想软边界，可设为 3/5（=avg pool 模糊）；0 表示不用
        self._vis_mask_bev = None  # 训练时可选记录一下（给你看 coverage）

        def pop_vis_mask_bev(self):
            m = self._vis_mask_bev
            self._vis_mask_bev = None
            return m

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
        self.register_buffer('frustum', self.create_frustum())
        self.height_channels, _, _, _ = self.frustum.shape

        self.img_backbone = build_backbone(img_backbone_conf)
        self.img_neck = build_neck(img_neck_conf)
        self.height_net = self._configure_height_net(height_net_conf)

        self.img_neck.init_weights()
        self.img_backbone.init_weights()

    def _build_learned_y(self):
        """将可学习的 dy_raw 累加成单调 y 位置，端点对齐 [ymin, ymax]"""
        dy = F.softplus(self.dy_raw) + 1e-3  # 保证正数步长
        c = torch.cumsum(dy, dim=0)  # 单调累加
        ys = self.ymin + (c / c[-1]) * (self.ymax - self.ymin)
        return ys

    def _init_bev_grid(self, x_bound, y_bound, learn_y=True):
        xmin, xmax, dx = x_bound
        ymin, ymax, dy = y_bound
        nx = int(round((xmax - xmin) / dx))
        ny = int(round((ymax - ymin) / dy))

        # x 轴固定均匀，作为 buffer 保存
        xs = torch.linspace(xmin + dx / 2, xmax - dx / 2, nx)
        self.HB, self.WB = ny, nx
        self.register_buffer('bev_xs', xs)  # (WB,)

        if learn_y:
            self.learn_y = True
            # 记录端点与“线性参考曲线”，便于正则化/可视化
            self.ymin = float(ymin);
            self.ymax = float(ymax)
            ys_lin = torch.linspace(ymin + dy / 2, ymax - dy / 2, ny)
            self.register_buffer('bev_ys_lin', ys_lin)  # (HB,)
            # 可学习的增量参数（初始为 0 ⇒ 近似线性）
            self.dy_raw = nn.Parameter(torch.zeros(ny))
            # 不在这里固定 bev_xy1；训练/推理时按当前 dy_raw 动态生成
        else:
            self.learn_y = False
            ys = torch.linspace(ymin + dy / 2, ymax - dy / 2, ny)
            yy, xx = torch.meshgrid(ys, xs)  # 1.9 默认 'ij'
            XY1 = torch.stack([xx, yy, torch.ones_like(xx)], -1)  # (HB,WB,3)
            self.register_buffer('bev_xy1', XY1)

        # ---------- 3) 计算每相机的像素采样网格 ----------
    def _compute_pixel_grid(self, sensor2ego, intrin, ida, img_hw, feat_hw):
        """
        sensor2ego: (B,N,4,4) cam->ego
        intrin:     (B,N,4,4) 仅用前3x3
        ida:        (B,N,4,4) 若 self.use_ida_in_H=False，将被忽略
        img_hw:     (Himg, Wimg)
        feat_hw:    (Hf, Wf)
        return:     grid (B,N,HB,WB,2)  [-1,1] 供 grid_sample
        """
        device = sensor2ego.device
        B, N = sensor2ego.shape[:2]
        HB, WB = self.HB, self.WB
        Himg, Wimg = img_hw
        Hf, Wf = feat_hw

        # 取 ego->cam
        ego2cam = torch.inverse(sensor2ego)  # (B,N,4,4)
        R = ego2cam[..., :3, :3]  # (B,N,3,3)
        t = ego2cam[..., :3, 3]  # (B,N,3)

        K = intrin[..., :3, :3]  # (B,N,3,3)
        fx = K[..., 0, 0];
        fy = K[..., 1, 1]
        cx = K[..., 0, 2];
        cy = K[..., 1, 2]

        # BEV 网格 (X,Y,1)
        device = sensor2ego.device
        B, N = sensor2ego.shape[:2]
        HB, WB = self.HB, self.WB

        # --- 动态或固定的 BEV 网格 ---
        if getattr(self, 'learn_y', False):
            ys = self._build_learned_y().to(device)  # (HB,)
            xs = self.bev_xs.to(device)  # (WB,)
            yy, xx = torch.meshgrid(ys, xs)  # 1.9 兼容
            XY1 = torch.stack([xx, yy, torch.ones_like(xx)], -1)  # (HB,WB,3)
        else:
            XY1 = self.bev_xy1.to(device)  # 固定网格

        # 展开成 (1,1,HB,WB,3) 方便广播
        XY1 = XY1.view(1, 1, HB, WB, 3)
        X = XY1[..., 0];
        Y = XY1[..., 1]

        # Pc = R * [X,Y,0]^T + t
        # 展开：X_cam = r00*X + r01*Y + t0, 同理 Y_cam, Z_cam
        r00 = R[..., 0, 0].unsqueeze(-1).unsqueeze(-1)  # (B,N,1,1)
        r01 = R[..., 0, 1].unsqueeze(-1).unsqueeze(-1)
        r10 = R[..., 1, 0].unsqueeze(-1).unsqueeze(-1)
        r11 = R[..., 1, 1].unsqueeze(-1).unsqueeze(-1)
        r20 = R[..., 2, 0].unsqueeze(-1).unsqueeze(-1)
        r21 = R[..., 2, 1].unsqueeze(-1).unsqueeze(-1)
        t0 = t[..., 0].unsqueeze(-1).unsqueeze(-1)
        t1 = t[..., 1].unsqueeze(-1).unsqueeze(-1)
        t2 = t[..., 2].unsqueeze(-1).unsqueeze(-1)

        Xc = r00 * X + r01 * Y + t0  # (B,N,HB,WB)
        Yc = r10 * X + r11 * Y + t1
        Zc = r20 * X + r21 * Y + t2

        # 像素坐标（不含 IDA）
        u = fx.unsqueeze(-1).unsqueeze(-1) * (Xc / (Zc + 1e-6)) + cx.unsqueeze(-1).unsqueeze(-1)
        v = fy.unsqueeze(-1).unsqueeze(-1) * (Yc / (Zc + 1e-6)) + cy.unsqueeze(-1).unsqueeze(-1)

        # 可选：把 IDA 线性作用到 [u,v,d,1]
        if self.use_ida_in_H:
            # ida: (B,N,4,4)
            a = ida[..., 0, 0].unsqueeze(-1).unsqueeze(-1);
            b = ida[..., 0, 1].unsqueeze(-1).unsqueeze(-1)
            c = ida[..., 0, 2].unsqueeze(-1).unsqueeze(-1);
            d4 = ida[..., 0, 3].unsqueeze(-1).unsqueeze(-1)
            e = ida[..., 1, 0].unsqueeze(-1).unsqueeze(-1);
            f = ida[..., 1, 1].unsqueeze(-1).unsqueeze(-1)
            g = ida[..., 1, 2].unsqueeze(-1).unsqueeze(-1);
            h = ida[..., 1, 3].unsqueeze(-1).unsqueeze(-1)
            up = a * u + b * v + c * Zc + d4
            vp = e * u + f * v + g * Zc + h
        else:
            up, vp = u, v

        # 像素 -> 特征坐标
        sx = Wimg / float(Wf)  # 注意浮点
        sy = Himg / float(Hf)
        uf = up / sx
        vf = vp / sy

        # 归一化到 [-1,1]
        if self.align_corners:
            xn = 2.0 * (uf / max(Wf - 1, 1)) - 1.0
            yn = 2.0 * (vf / max(Hf - 1, 1)) - 1.0
        else:
            xn = (uf + 0.5) / Wf * 2.0 - 1.0
            yn = (vf + 0.5) / Hf * 2.0 - 1.0

        return torch.stack([xn, yn], dim=-1)  # (B,N,HB,WB,2)

    def y_mapping_regularization(self, lambda_smooth=1e-4, lambda_shape=1e-3):
        """光滑项 + 形状项（相对于线性参考曲线）"""
        if not getattr(self, 'learn_y', False):
            return torch.tensor(0.0, device=self.bev_xs.device)
        ys = self._build_learned_y()
        ys_lin = self.bev_ys_lin.to(ys.device)
        # 平滑：对 dy_raw 的一阶差分
        smooth = (self.dy_raw[1:] - self.dy_raw[:-1]).pow(2).mean()
        # 形变幅度：避免整体过度偏离线性
        shape = (ys - ys_lin).abs().mean()
        return lambda_smooth * smooth + lambda_shape * shape
    # ---------- 4) 从多相机特征拉到 BEV ----------
    def _ipm_from_feats(self, feats, sensor2ego, intrin, ida, img_hw):
        import torch.nn.functional as F
        B, N, C, Hf, Wf = feats.shape
        grid = self._compute_pixel_grid(sensor2ego, intrin, ida, img_hw, (Hf, Wf))  # (B,N,HB,WB,2)

        # --- grid_sample 多相机融合 ---
        bev = None
        for n in range(N):
            grid_n = grid[:, n]  # (B,HB,WB,2)
            feat_n = feats[:, n]  # (B,C,Hf,Wf)
            bev_n = F.grid_sample(
                feat_n, grid_n, mode='bilinear',
                padding_mode='zeros', align_corners=self.align_corners
            )  # (B,C,HB,WB)
            bev = bev_n if bev is None else (bev + bev_n)

        # --- 计算可见区域 mask（多相机并集）---
        with torch.no_grad():
            vis_list = []
            for n in range(N):
                g = grid[:, n]
                inb = ((g[..., 0] >= -1) & (g[..., 0] <= 1) &
                       (g[..., 1] >= -1) & (g[..., 1] <= 1)).float()  # (B,HB,WB)
                vis_list.append(inb)
            vis = torch.stack(vis_list, dim=1).amax(dim=1)  # (B,HB,WB)

            # 轻微膨胀，避免边缘 BEV 栅格被误判不可见
            if self.mask_dilate_ks and self.mask_dilate_ks > 1:
                pad = self.mask_dilate_ks // 2
                vis = F.max_pool2d(vis.unsqueeze(1), kernel_size=self.mask_dilate_ks,
                                   stride=1, padding=pad).squeeze(1)  # 仍是 0/1
            # 可选：做个小模糊，得到软边界（不是必须）
            if self.mask_blur_ks and self.mask_blur_ks > 1:
                pad = self.mask_blur_ks // 2
                vis = F.avg_pool2d(vis.unsqueeze(1).float(),
                                   kernel_size=self.mask_blur_ks,
                                   stride=1, padding=pad).squeeze(1).clamp(0, 1)

            self._vis_mask_bev = vis  # 训练时可在 Lightning 里 log coverage

        # --- 通道适配 ---
        if self.adapter is None:
            self.adapter = nn.Conv2d(C, self.output_channels, kernel_size=1, bias=False).to(feats.device)
        bev = self.adapter(bev)  # (B, Cout, HB, WB)

        # --- 直接把 mask 乘到 BEV 特征上（最关键的一行） ---
        if self.apply_bev_mask and (self._vis_mask_bev is not None):
            bev = bev * self._vis_mask_bev.unsqueeze(1)  # (B,1,HB,WB) broadcast

        return bev

    def _configure_height_net(self, height_net_conf):
        return HeightNet(
            height_net_conf['in_channels'],
            height_net_conf['mid_channels'],
            self.output_channels,
            self.height_channels,
        )

    def create_frustum(self):
        """Generate frustum"""
        # make grid in image plane
        ogfH, ogfW = self.final_dim
        fH, fW = ogfH // self.downsample_factor, ogfW // self.downsample_factor

        # DID
        alpha = 1.5
        d_coords = np.arange(self.d_bound[2]) / self.d_bound[2]
        d_coords = np.power(d_coords, alpha)
        d_coords = self.d_bound[0] + d_coords * (self.d_bound[1] - self.d_bound[0])
        d_coords = torch.tensor(d_coords, dtype=torch.float).view(-1, 1, 1).expand(-1, fH, fW)

        D, _, _ = d_coords.shape
        x_coords = torch.linspace(0, ogfW - 1, fW, dtype=torch.float).view(
            1, 1, fW).expand(D, fH, fW)
        y_coords = torch.linspace(0, ogfH - 1, fH,
                                  dtype=torch.float).view(1, fH,
                                                          1).expand(D, fH, fW)
        paddings = torch.ones_like(d_coords)

        # D x H x W x 3
        frustum = torch.stack((x_coords, y_coords, d_coords, paddings), -1)
        return frustum

    def height2localtion(self, points, sensor2ego_mat, sensor2virtual_mat, intrin_mat, reference_heights):
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
            rots(Tensor): Rotation matrix from camera to ego.
            trans(Tensor): Translation matrix from camera to ego.
            intrins(Tensor): Intrinsic matrix.
            post_rots_ida(Tensor): Rotation matrix for ida.
            post_trans_ida(Tensor): Translation matrix for ida
            post_rot_bda(Tensor): Rotation matrix for bda.

        Returns:
            Tensors: points ego coord.
        """
        batch_size, num_cams, _, _ = sensor2ego_mat.shape

        # undo post-transformation
        # B x N x D x H x W x 3
        points = self.frustum
        ida_mat = ida_mat.view(batch_size, num_cams, 1, 1, 1, 4, 4)
        points = ida_mat.inverse().matmul(points.unsqueeze(-1))
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
        return self.height_net(feat, mats_dict)

    def _forward_voxel_net(self, img_feat_with_height):
        return img_feat_with_height

    def _forward_single_sweep(self, sweep_index, sweep_imgs, mats_dict, is_return_height=False):
        """
        现在用 IPM → grid_sample；不再返回 height。
        sweep_imgs: (B,1,N,3,H,W)
        """
        assert self.use_ipm, "self.use_ipm=False 时请改回原始路径"
        B, _, N, Cimg, Himg, Wimg = sweep_imgs.shape

        # 1) 提取特征（保持你已有的 get_cam_feats）
        img_feats = self.get_cam_feats(sweep_imgs)  # (B,1,N,Cf,Hf,Wf)
        feats = img_feats[:, 0]  # (B,N,Cf,Hf,Wf)

        # 2) 取当帧的标定矩阵
        K = mats_dict['intrin_mats'][:, sweep_index]  # (B,N,4,4)
        s2e = mats_dict['sensor2ego_mats'][:, sweep_index]  # (B,N,4,4)
        ida = mats_dict['ida_mats'][:, sweep_index]  # (B,N,4,4)

        # 3) IPM 到 BEV
        bev = self._ipm_from_feats(feats, s2e, K, ida, (Himg, Wimg))  # (B,C_out,HB,WB)
        return bev.contiguous()

    def forward(self, sweep_imgs, mats_dict, timestamps=None, is_return_height=False):
        B, S, N, C, H, W = sweep_imgs.shape
        if self.use_ipm:
            key = self._forward_single_sweep(0, sweep_imgs[:, 0:1, ...], mats_dict, is_return_height=False)
            if S == 1:
                return key
            outs = [key]
            for s in range(1, S):
                with torch.no_grad():
                    outs.append(
                        self._forward_single_sweep(s, sweep_imgs[:, s:s + 1, ...], mats_dict, is_return_height=False))
            return torch.cat(outs, dim=1)  # (B, S*C_out, HB, WB)
        else:
            # 如需兼容老路径，可保留原实现
            raise NotImplementedError("Set self.use_ipm=True to use IPM pipeline.")
