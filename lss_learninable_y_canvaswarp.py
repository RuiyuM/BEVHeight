# Copyright (c) Megvii Inc. All rights reserved.
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from mmcv.cnn import build_conv_layer
from mmdet3d.models import build_neck
from mmdet.models import build_backbone
from mmdet.models.backbones.resnet import BasicBlock

from ops.voxel_pooling import voxel_pooling

__all__ = ['LSSFPN']


# ---------------- CanvasWarpXY（可学习画布映射，可选） ----------------
class CanvasWarpXY(nn.Module):
    """
    把扇形有效区（按极坐标 r,theta 展开）拉伸铺满整个 (HB,WB) 画布的可学习小形变。
    - 列方向 WB 对应 theta；行方向 HB 对应 r
    - 学习的是每列的 theta 偏移 & 每行的 r 偏移（tanh 限幅，再按区间缩放）
    - 输入:  bev_src  (B,C,HB,WB)  —— 你原始的 BEV 特征（扇形区域非零，其它零）
             vis_mask (B,HB,WB)    —— 可见扇形区域（多相机并集），可为 None
             bev_xy1  (HB,WB,3)    —— 每个 BEV 像素对应的物理坐标 [X,Y,1]
    - 输出:  bev_warp (B,C,HB,WB)  —— 铺满后的 BEV（不再乘 mask）
    """
    def __init__(self, HB, WB, theta_scale=0.2, r_scale=0.2, reg_w=1e-4):
        super().__init__()
        self.HB = HB
        self.WB = WB
        self.theta_scale = float(theta_scale)  # 相对跨度的最大偏移比例
        self.r_scale = float(r_scale)
        self.reg_w = float(reg_w)

        # 可学习偏移：列方向 WB 个，行方向 HB 个
        self.theta_offset = nn.Parameter(torch.zeros(WB))
        self.r_offset = nn.Parameter(torch.zeros(HB))

        # 正则缓存
        self.reg_loss = torch.tensor(0.0)

    @torch.no_grad()
    def _infer_sector_bounds(self, bev_xy1, vis_mask=None):
        """
        从 bev_xy1 推出扇形的 (theta_min, theta_max, r_min, r_max)
        若给了 vis_mask，仅在可见区域内统计。theta 做 wrap-aware 的最小跨度选择。
        返回 Python float（兼容 torch==1.9 的 linspace）
        """
        # (HB,WB,3) -> (HB,WB)
        X = bev_xy1[..., 0]
        Y = bev_xy1[..., 1]
        theta = torch.atan2(Y, X)              # [-pi, pi]
        theta = (theta + 2*math.pi) % (2*math.pi)  # [0, 2pi)
        r = torch.sqrt(X * X + Y * Y)

        if vis_mask is None:
            th_min_t = theta.min()
            th_max_t = theta.max()
            r_min_t = r.min()
            r_max_t = r.max()
        else:
            m = vis_mask > 0
            if m.sum() < 1:
                th_min_t = theta.min()
                th_max_t = theta.max()
                r_min_t = r.min()
                r_max_t = r.max()
            else:
                thv = theta[m]
                rv  = r[m]
                th0 = thv
                th1 = (thv + math.pi) % (2*math.pi)
                # 选跨度更小的角度表，避免跨 0/2pi 带来的大跨度
                span0 = th0.max() - th0.min()
                span1 = th1.max() - th1.min()
                th_use = th1 if span1 < span0 else th0
                th_min_t = th_use.min()
                th_max_t = th_use.max()
                r_min_t = rv.min()
                r_max_t = rv.max()

        # 兼容 linspace：转 python float
        th_min = float(th_min_t.detach().cpu().item())
        th_max = float(th_max_t.detach().cpu().item())
        r_min  = float(r_min_t.detach().cpu().item())
        r_max  = float(r_max_t.detach().cpu().item())
        return th_min, th_max, r_min, r_max

    def _build_base_axes(self, device, dtype, th_min, th_max, r_min, r_max):
        """构造基础等距轴（兼容 1.9），返回 (theta_base[WB], r_base[HB])"""
        theta_base = torch.linspace(th_min, th_max, self.WB, device=device, dtype=dtype)
        r_base = torch.linspace(r_min, r_max, self.HB, device=device, dtype=dtype)
        return theta_base, r_base

    def _make_sampling_grid(self, theta_new, r_new, th_min, th_max, r_min, r_max):
        """
        把目标画布上的 (theta_new, r_new) 映射回源图的栅格索引，并转成 grid_sample 需要的 [-1,1] 归一化坐标。
        返回: grid (1,HB,WB,2)
        """
        # 归一化到 [0,1]
        dth = max(th_max - th_min, 1e-6)
        dr  = max(r_max - r_min,  1e-6)
        s_th = (theta_new - th_min) / dth  # (HB,WB)
        s_r  = (r_new    - r_min)  / dr    # (HB,WB)
        s_th = s_th.clamp(0.0, 1.0)
        s_r  = s_r.clamp(0.0, 1.0)

        # 源图像素索引
        u = s_th * (self.WB - 1)  # 列
        v = s_r  * (self.HB - 1)  # 行

        # 转成 [-1,1]
        x = 2.0 * (u / max(self.WB - 1, 1)) - 1.0
        y = 2.0 * (v / max(self.HB - 1, 1)) - 1.0
        grid = torch.stack([x, y], dim=-1).unsqueeze(0)  # (1,HB,WB,2)
        return grid

    def _regularization(self):
        """
        对偏移做一阶差分 L2，避免剧烈振荡；按 reg_w 缩放。
        """
        reg = 0.0
        if self.theta_offset.numel() > 1:
            dth = self.theta_offset[1:] - self.theta_offset[:-1]
            reg = reg + (dth.pow(2).mean())
        if self.r_offset.numel() > 1:
            dr = self.r_offset[1:] - self.r_offset[:-1]
            reg = reg + (dr.pow(2).mean())
        return reg * self.reg_w

    def forward(self, bev_src, vis_mask, bev_xy1):
        """
        bev_src: (B,C,HB,WB)
        vis_mask: (B,HB,WB) or None
        bev_xy1: (HB,WB,3) 物理坐标
        """
        B, C, HB, WB = bev_src.shape
        assert HB == self.HB and WB == self.WB, f"CanvasWarpXY size mismatch: got {(HB,WB)} expect {(self.HB,self.WB)}"
        dev = bev_src.device
        dtype = bev_src.dtype

        # 1) 扇形边界（只需一套边界；使用可见区域求并集）
        if vis_mask is not None:
            vis_any = (vis_mask.float().amax(dim=0) > 0).float()  # (HB,WB)
        else:
            vis_any = None
        th_min, th_max, r_min, r_max = self._infer_sector_bounds(bev_xy1.to(dev), vis_any)

        # 2) 基础轴 + 可学习偏移（tanh 限幅，按跨度缩放）
        theta_base, r_base = self._build_base_axes(dev, dtype, th_min, th_max, r_min, r_max)  # (WB),(HB)
        dth = max(th_max - th_min, 1e-6)
        dr  = max(r_max - r_min,  1e-6)
        # 形状对齐
        th_off = torch.tanh(self.theta_offset.to(dev, dtype)) * (self.theta_scale * dth)  # (WB,)
        r_off  = torch.tanh(self.r_offset.to(dev, dtype))     * (self.r_scale * dr)       # (HB,)

        # 网格 (HB,WB)
        r_new, theta_new = torch.meshgrid(r_base + r_off, theta_base + th_off)  # 1.9 默认 'ij'
        # 3) 生成采样网格（去源扇形矩形中采样）
        grid = self._make_sampling_grid(theta_new, r_new, th_min, th_max, r_min, r_max)  # (1,HB,WB,2)
        grid = grid.to(dev, dtype)

        # 4) 对每个 batch 使用同一采样网格
        grid_rep = grid.expand(B, -1, -1, -1)  # (B,HB,WB,2)

        bev_warp = F.grid_sample(
            bev_src, grid_rep, mode='bilinear',
            padding_mode='zeros', align_corners=True
        )  # (B,C,HB,WB)

        # 5) 更新正则
        self.reg_loss = self._regularization().to(dev)

        # 6) 一定要 return
        return bev_warp


# ---------------- 下面是你原来的模块们（无改动或小改动） ----------------
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
        x1 = self.aspp1(x); x2 = self.aspp2(x); x3 = self.aspp3(x); x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        x = self.conv1(x); x = self.bn1(x); x = self.relu(x)
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
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        return x * self.gate(x_se)


class HeightNet(nn.Module):
    def __init__(self, in_channels, mid_channels, context_channels,
                 height_channels):
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


# ---------------- LSSFPN（含 learn-y 与可选 CanvasWarpXY） ----------------
class LSSFPN(nn.Module):
    def __init__(self, x_bound, y_bound, z_bound, d_bound, final_dim,
                 downsample_factor, output_channels, img_backbone_conf,
                 img_neck_conf, height_net_conf):
        super(LSSFPN, self).__init__()
        # 保存边界
        self.x_bound = x_bound  # [xmin, xmax, dx]
        self.y_bound = y_bound  # [ymin, ymax, dy]

        self.downsample_factor = downsample_factor
        self.d_bound = d_bound
        self.final_dim = final_dim
        self.output_channels = output_channels

        self.use_ipm = True
        self.use_ida_in_H = False
        self.align_corners = True

        # learn-y：动态 Y 轴；不注册固定 bev_xy1
        self._init_bev_grid(x_bound, y_bound, learn_y=True)

        self.adapter = None
        self.apply_bev_mask = False       # 乘扇形可见 mask（建议：当 use_canvas_warp=True 时设为 False）
        self.mask_dilate_ks = 5
        self.mask_blur_ks = 0
        self._vis_mask_bev = None

        # （可选）可学习画布映射：默认关
        self.use_canvas_warp = True
        self.canvas_warp = CanvasWarpXY(self.HB, self.WB, theta_scale=0.2, r_scale=0.2, reg_w=1e-4)

        # 体素/高度等（保留原始）
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

    # ---------- learn-y：把可学习的 dy_raw 累加成单调 y（单位米） ----------
    def _build_learned_y(self):
        dy = F.softplus(self.dy_raw) + 1e-3
        c = torch.cumsum(dy, dim=0)
        ys = self.ymin + (c / c[-1]) * (self.ymax - self.ymin)
        return ys  # (HB,)

    # ---------- （新）根据当前 learn-y 构建 BEV XY1（动态） ----------
    def build_bev_xy1(self, device=None):
        dev = device or (self.bev_xs.device if hasattr(self, 'bev_xs') else None)
        xs = self.bev_xs.to(dev)                        # (WB,)
        ys = self._build_learned_y().to(dev)            # (HB,)
        yy, xx = torch.meshgrid(ys, xs)                 # PyTorch 1.9 默认 'ij'
        XY1 = torch.stack([xx, yy, torch.ones_like(xx)], dim=-1)  # (HB,WB,3)
        return XY1

    def _init_bev_grid(self, x_bound, y_bound, learn_y=True):
        xmin, xmax, dx = x_bound
        ymin, ymax, dy = y_bound
        nx = int(round((xmax - xmin) / dx))
        ny = int(round((ymax - ymin) / dy))

        self.WB = nx; self.HB = ny
        xs = torch.linspace(xmin + dx / 2, xmax - dx / 2, nx)
        self.register_buffer('bev_xs', xs)  # (WB,)

        if learn_y:
            self.ymin = float(ymin); self.ymax = float(ymax)
            ys_lin = torch.linspace(ymin + dy / 2, ymax - dy / 2, ny)
            self.register_buffer('bev_ys_lin', ys_lin)  # (HB,)
            self.dy_raw = nn.Parameter(torch.zeros(ny)) # 可学习增量
            self.learn_y = True
        else:
            self.learn_y = False
            yy, xx = torch.meshgrid(
                torch.linspace(ymin + dy / 2, ymax - dy / 2, ny),
                xs
            )
            XY1 = torch.stack([xx, yy, torch.ones_like(xx)], -1)
            self.register_buffer('bev_xy1', XY1)

    # ---------- 计算每相机的像素采样网格（特征级 IPM） ----------
    def _compute_pixel_grid(self, sensor2ego, intrin, ida, img_hw, feat_hw):
        device = sensor2ego.device
        B, N = sensor2ego.shape[:2]
        HB, WB = self.HB, self.WB
        Himg, Wimg = img_hw
        Hf, Wf = feat_hw

        ego2cam = torch.inverse(sensor2ego)
        R = ego2cam[..., :3, :3]
        t = ego2cam[..., :3, 3]

        K = intrin[..., :3, :3]
        fx = K[..., 0, 0]; fy = K[..., 1, 1]
        cx = K[..., 0, 2]; cy = K[..., 1, 2]

        # 动态 BEV 网格
        if getattr(self, 'learn_y', False):
            XY1 = self.build_bev_xy1(device).view(1, 1, HB, WB, 3)
        else:
            XY1 = self.bev_xy1.to(device).view(1, 1, HB, WB, 3)

        X = XY1[..., 0]; Y = XY1[..., 1]

        r00 = R[..., 0, 0].unsqueeze(-1).unsqueeze(-1)
        r01 = R[..., 0, 1].unsqueeze(-1).unsqueeze(-1)
        r10 = R[..., 1, 0].unsqueeze(-1).unsqueeze(-1)
        r11 = R[..., 1, 1].unsqueeze(-1).unsqueeze(-1)
        r20 = R[..., 2, 0].unsqueeze(-1).unsqueeze(-1)
        r21 = R[..., 2, 1].unsqueeze(-1).unsqueeze(-1)
        t0 = t[..., 0].unsqueeze(-1).unsqueeze(-1)
        t1 = t[..., 1].unsqueeze(-1).unsqueeze(-1)
        t2 = t[..., 2].unsqueeze(-1).unsqueeze(-1)

        Xc = r00 * X + r01 * Y + t0
        Yc = r10 * X + r11 * Y + t1
        Zc = r20 * X + r21 * Y + t2

        u = fx.unsqueeze(-1).unsqueeze(-1) * (Xc / (Zc + 1e-6)) + cx.unsqueeze(-1).unsqueeze(-1)
        v = fy.unsqueeze(-1).unsqueeze(-1) * (Yc / (Zc + 1e-6)) + cy.unsqueeze(-1).unsqueeze(-1)

        if self.use_ida_in_H:
            a = ida[..., 0, 0].unsqueeze(-1).unsqueeze(-1); b = ida[..., 0, 1].unsqueeze(-1).unsqueeze(-1)
            c = ida[..., 0, 2].unsqueeze(-1).unsqueeze(-1); d4 = ida[..., 0, 3].unsqueeze(-1).unsqueeze(-1)
            e = ida[..., 1, 0].unsqueeze(-1).unsqueeze(-1); f = ida[..., 1, 1].unsqueeze(-1).unsqueeze(-1)
            g = ida[..., 1, 2].unsqueeze(-1).unsqueeze(-1); h = ida[..., 1, 3].unsqueeze(-1).unsqueeze(-1)
            up = a * u + b * v + c * Zc + d4
            vp = e * u + f * v + g * Zc + h
        else:
            up, vp = u, v

        sx = Wimg / float(Wf)
        sy = Himg / float(Hf)
        uf = up / sx
        vf = vp / sy

        if self.align_corners:
            xn = 2.0 * (uf / max(Wf - 1, 1)) - 1.0
            yn = 2.0 * (vf / max(Hf - 1, 1)) - 1.0
        else:
            xn = (uf + 0.5) / Wf * 2.0 - 1.0
            yn = (vf + 0.5) / Hf * 2.0 - 1.0

        return torch.stack([xn, yn], dim=-1)  # (B,N,HB,WB,2)

    # ---------- learn-y 的正则（可加到总损失里） ----------
    def y_mapping_regularization(self, lambda_smooth=1e-4, lambda_shape=1e-3):
        if not getattr(self, 'learn_y', False):
            return torch.tensor(0.0, device=self.bev_xs.device)
        ys = self._build_learned_y()
        ys_lin = self.bev_ys_lin.to(ys.device)
        smooth = (self.dy_raw[1:] - self.dy_raw[:-1]).pow(2).mean()
        shape = (ys - ys_lin).abs().mean()
        return lambda_smooth * smooth + lambda_shape * shape

    # ---------- 特征级 IPM & 多相机融合 ----------
    def _ipm_from_feats(self, feats, sensor2ego, intrin, ida, img_hw):
        """
        feats:      (B,N,C,Hf,Wf)
        sensor2ego: (B,N,4,4)
        intrin:     (B,N,4,4)
        ida:        (B,N,4,4)
        img_hw:     (Himg, Wimg)
        return:     (B, C_out, HB, WB)
        """
        import torch.nn.functional as F

        # ---- 形状检查 ----
        assert feats.dim() == 5, f"[ipm] feats.dim={feats.dim()}, expect 5D (B,N,C,Hf,Wf)"
        B, N, C, Hf, Wf = feats.shape
        assert int(N) >= 1, f"[ipm] N(cams)={N}，没有相机特征；请检查 dataloader / cams 列表"

        Himg, Wimg = img_hw
        HB, WB = self.HB, self.WB

        # ---- 第一次打印关键形状（仅一次）----
        if not hasattr(self, "_ipm_debug_once"):
            self._ipm_debug_once = True
            print(f"[ipm] feats={tuple(feats.shape)}, HimgxWimg=({Himg},{Wimg}), "
                  f"HfxWf=({Hf},{Wf}), BEV=({HB},{WB}), C_out={self.output_channels}")

        # ---- 采样网格 (B,N,HB,WB,2) ----
        grid = self._compute_pixel_grid(sensor2ego, intrin, ida, img_hw, (Hf, Wf))  # (B,N,HB,WB,2)
        assert grid.shape[:2] == (B, N) and grid.shape[-1] == 2, f"[ipm] bad grid shape: {tuple(grid.shape)}"

        # ---- 向量化 grid_sample：把相机维折叠到 batch ----
        grid_bn = grid.contiguous().view(B * N, HB, WB, 2)  # (B*N,HB,WB,2)
        feat_bn = feats.contiguous().view(B * N, C, Hf, Wf)  # (B*N,C,Hf,Wf)

        bev_bn = F.grid_sample(
            feat_bn, grid_bn, mode='bilinear',
            padding_mode='zeros', align_corners=self.align_corners
        )  # (B*N,C,HB,WB)

        # 聚合相机（求和/求均值均可，先保持求和）
        bev = bev_bn.view(B, N, C, HB, WB).sum(dim=1)  # (B,C,HB,WB)

        # ---- 可见区域 mask（多相机并集）----
        with torch.no_grad():
            inb_bn = ((grid_bn[..., 0] >= -1) & (grid_bn[..., 0] <= 1) &
                      (grid_bn[..., 1] >= -1) & (grid_bn[..., 1] <= 1)).float()  # (B*N,HB,WB)
            inb = inb_bn.view(B, N, HB, WB).amax(dim=1)  # (B,HB,WB)

            # 膨胀 / 模糊（可选）
            if self.mask_dilate_ks and self.mask_dilate_ks > 1:
                pad = self.mask_dilate_ks // 2
                inb = F.max_pool2d(inb.unsqueeze(1), kernel_size=self.mask_dilate_ks,
                                   stride=1, padding=pad).squeeze(1)
            if self.mask_blur_ks and self.mask_blur_ks > 1:
                pad = self.mask_blur_ks // 2
                inb = F.avg_pool2d(inb.unsqueeze(1).float(),
                                   kernel_size=self.mask_blur_ks,
                                   stride=1, padding=pad).squeeze(1).clamp(0, 1)
            self._vis_mask_bev = inb  # 记录下来（可视化/统计 coverage）

        # ---- 1x1 adapter ----
        if self.adapter is None:
            self.adapter = nn.Conv2d(C, self.output_channels, kernel_size=1, bias=False).to(feats.device)
        bev = self.adapter(bev)  # (B,C_out,HB,WB)

        # ---- 分支：扇形 or 铺满（CanvasWarpXY）----
        if getattr(self, "use_canvas_warp", False):
            assert hasattr(self, "canvas_warp"), "[ipm] use_canvas_warp=True 但未构造 self.canvas_warp"
            bev_xy1 = self.build_bev_xy1(device=feats.device)  # (HB,WB,3) 基于 learn-y 动态生成
            bev = self.canvas_warp(bev, self._vis_mask_bev, bev_xy1)  # 铺满画布；此时不要再乘 mask
        else:
            if self.apply_bev_mask and (self._vis_mask_bev is not None):
                bev = bev * self._vis_mask_bev.unsqueeze(1)

        # ---- 兜底检查 & 返回 ----
        if torch.isnan(bev).any():
            raise RuntimeError("[ipm] bev contains NaN")
        return bev.contiguous()

    def _configure_height_net(self, height_net_conf):
        return HeightNet(
            height_net_conf['in_channels'],
            height_net_conf['mid_channels'],
            self.output_channels,
            self.height_channels,
        )

    # ---------------- 以下保持你原始实现（未动） ----------------
    def create_frustum(self):
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
        frustum = torch.stack((x_coords, y_coords, d_coords, paddings), -1)
        return frustum

    def height2localtion(self, points, sensor2ego_mat, sensor2virtual_mat, intrin_mat, reference_heights):
        batch_size, num_cams, _, _ = sensor2ego_mat.shape
        reference_heights = reference_heights.view(batch_size, num_cams, 1, 1, 1, 1, 1)\
                                               .repeat(1, 1, points.shape[2], points.shape[3], points.shape[4], 1, 1)
        height = -1 * points[:, :, :, :, :, 2, :] + reference_heights[:, :, :, :, :, 0, :]
        points_const = points.clone()
        points_const[:, :, :, :, :, 2, :] = 10
        points_const = torch.cat(
            (points_const[:, :, :, :, :, :2] * points_const[:, :, :, :, :, 2:3],
             points_const[:, :, :, :, :, 2:]), 5)
        combine_virtual = sensor2virtual_mat.matmul(torch.inverse(intrin_mat))
        points_virtual = combine_virtual.view(batch_size, num_cams, 1, 1, 1, 4, 4).matmul(points_const)
        ratio = height[:, :, :, :, :, 0] / points_virtual[:, :, :, :, :, 1, 0]
        ratio = ratio.view(batch_size, num_cams, ratio.shape[2], ratio.shape[3], ratio.shape[4], 1, 1)\
                     .repeat(1, 1, 1, 1, 1, 4, 1)
        points = points_virtual * ratio
        points[:, :, :, :, :, 3, :] = 1
        combine_ego = sensor2ego_mat.matmul(torch.inverse(sensor2virtual_mat))
        points = combine_ego.view(batch_size, num_cams, 1, 1, 1, 4, 4).matmul(points)
        return points

    def get_geometry(self, sensor2ego_mat, sensor2virtual_mat, intrin_mat, ida_mat, reference_heights, bda_mat):
        batch_size, num_cams, _, _ = sensor2ego_mat.shape
        points = self.frustum
        ida_mat = ida_mat.view(batch_size, num_cams, 1, 1, 1, 4, 4)
        points = ida_mat.inverse().matmul(points.unsqueeze(-1))
        points = self.height2localtion(points, sensor2ego_mat, sensor2virtual_mat, intrin_mat, reference_heights)
        if bda_mat is not None:
            bda_mat = bda_mat.unsqueeze(1).repeat(1, num_cams, 1, 1).view(batch_size, num_cams, 1, 1, 1, 4, 4)
            points = (bda_mat @ points).squeeze(-1)
        else:
            points = points.squeeze(-1)
        return points[..., :3]

    def get_cam_feats(self, imgs):
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

    def _forward_single_sweep(self, sweep_index, sweep_imgs, mats_dict, is_return_height=False):
        assert self.use_ipm, "self.use_ipm=False 时请改回原始路径"
        B, _, N, Cimg, Himg, Wimg = sweep_imgs.shape

        img_feats = self.get_cam_feats(sweep_imgs)     # (B,1,N,Cf,Hf,Wf)
        feats = img_feats[:, 0]                        # (B,N,Cf,Hf,Wf)

        K   = mats_dict['intrin_mats'][:, sweep_index]     # (B,N,4,4)
        s2e = mats_dict['sensor2ego_mats'][:, sweep_index] # (B,N,4,4)
        ida = mats_dict['ida_mats'][:, sweep_index]        # (B,N,4,4)

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
                    outs.append(self._forward_single_sweep(s, sweep_imgs[:, s:s + 1, ...], mats_dict, is_return_height=False))
            return torch.cat(outs, dim=1)  # (B, S*C_out, HB, WB)
        else:
            raise NotImplementedError("Set self.use_ipm=True to use IPM pipeline.")
