import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from mmcv.runner import BaseModule
from .wavelet_process import DWTC


class FusionLayer(nn.Module):
    def __init__(
            self, 
            in_channels,
            reduction=16,
            num_layers=5,
            fs_type='cat',
            use_om=True,
            use_grid=True,
            use_fixed_ga=False,
            use_msf=True,
            om_kernels=[9, 7, 5, 3, 1],
            msf_kernels=[7, 5, 3],
            wf_loss=False,
            wf_loss_mode='kl_v1',
            wf_loss_weight=0.1,
            use_clfm=[],
            clfm_mode='up_new',
            usepoolup=['v'],
            aam_align_corners=True,
            aam_warp=True,
            tdr_levels=(0,),
            tdr_loss_weight=0.0,
            tdr_hm_min_sigma=1.0):
        super(FusionLayer, self).__init__()
        self.in_channels = in_channels
        self.reduction = reduction
        self.num_layers = num_layers
        self.fs_type = fs_type
        self.use_om = use_om
        self.use_grid = use_grid
        self.use_msf = use_msf
        self.om_kernels = om_kernels
        self.msf_kernels = msf_kernels
        self.wf_loss = wf_loss
        self.wf_loss_mode = wf_loss_mode
        self.wf_loss_weight = wf_loss_weight
        self.use_clfm = use_clfm
        self.usepoolup = usepoolup
        self.tdr_loss_weight = tdr_loss_weight
        self.tdr_hm_min_sigma = tdr_hm_min_sigma
        self.tdr_loss = None

        if 'v2' in self.use_clfm:
            self.idwt_layers = nn.ModuleList()
            for i in range(num_layers):
                self.idwt_layers.append(
                    DWTC(
                        in_channel=in_channels, 
                        out_channel=in_channels,
                        mode='upsample_v4',))
                
        if 'v3' in self.use_clfm or 't3' in self.use_clfm:
            self.idwt_layers = nn.ModuleList()
            for i in range(num_layers):
                self.idwt_layers.append(
                    DWTC(
                        in_channel=in_channels, 
                        out_channel=in_channels,
                        mode=clfm_mode,
                        # Levels 1-3 fall back to the champion path: on the
                        # trained champion their CLFM gradient is 1e4 to 3e5
                        # times smaller than level 0's, because ATSS assigns
                        # every tiny person to P3.  A module there would carry
                        # parameters that never train.
                        tdr_on=(i in tdr_levels),))
        if fs_type == 'cat' or fs_type == 'clfm':
            self.conv = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)
        elif fs_type == 'add':
            assert in_channels % 2 == 0, "in_channels should be divisible by 2 for add fusion type"
        elif fs_type == 'fusionnet-xo':
            self.hofm_layers = nn.ModuleList()
            for i in range(num_layers):
                self.hofm_layers.append(
                    HOFM(
                        in_channels=in_channels,
                        reduction=reduction,
                        num_stages=i,
                        use_om=use_om,
                        use_grid=use_grid,
                        use_fixed_ga=use_fixed_ga,
                        use_msf=use_msf,
                        om_kernels=om_kernels,
                        msf_kernels=msf_kernels,
                        aam_align_corners=aam_align_corners,
                        aam_warp=aam_warp))
        else:
            raise ValueError(f"Unsupported fusion type: {fs_type}")

        if 'v' in self.usepoolup:
            self.poolupsample_v = PoolingUpsample(in_channels)
        if 't' in self.usepoolup:
            self.poolupsample_t = PoolingUpsample(in_channels)

    def forward(self, v_feats, t_feats, gt_bboxes=None, img_metas=None):
        fused_feats = []
        for i in range(self.num_layers):
            v_feat = v_feats[i]
            t_feat = t_feats[i]
            if 'v' in self.usepoolup:
                v_feat = self.poolupsample_v(v_feat)
            if 't' in self.usepoolup:
                t_feat = self.poolupsample_t(t_feat)

            if self.fs_type == 'cat':
                fused_feat = torch.cat([v_feat, t_feat], dim=1)
                fused_feats.append(self.conv(fused_feat))

            elif self.fs_type == 'add':
                fused_feat = v_feat + t_feat
                fused_feats.append(fused_feat)
            
            elif self.fs_type == 'fusionnet-xo':
                if len(self.use_clfm) != 0:
                    if 'v' == self.use_clfm[0]:
                        v_feat = F.interpolate(v_feat, size=t_feat.shape[2:], mode='bilinear', align_corners=True)
                    elif 'v2' in self.use_clfm or 'v3' in self.use_clfm:
                        out = self.idwt_layers[i](t_feat, v_feat)
                        if isinstance(out, tuple):
                            # up_tdr also hands back a refined thermal feature;
                            # t_feats[] itself is untouched, so wf_loss keeps
                            # targeting the raw thermal and stays comparable.
                            v_feat, t_feat = out
                        else:
                            v_feat = out
                    elif 't2' in self.use_clfm or 't3' in self.use_clfm:
                        t_feat = self.idwt_layers[i](v_feat, t_feat)
                    elif 't' in self.use_clfm:
                        t_feat = F.interpolate(t_feat, size=v_feat.shape[2:], mode='bilinear', align_corners=True)
                fused_feat = self.hofm_layers[i](v_feat, t_feat)
                fused_feats.append(fused_feat)

        if self.training:
            self.tdr_loss = (self.compute_tdr_loss(gt_bboxes, img_metas)
                             if self.tdr_loss_weight > 0 else None)
            if self.wf_loss:
                if self.wf_loss_mode == 'kl_v1':
                    wf_loss = 0
                    for i in range(self.num_layers):
                        wf_loss += self.compute_kl_loss_near_objects(fused_feats[i], t_feats[i], gt_bboxes, img_metas)
                    return fused_feats, wf_loss
                elif self.wf_loss_mode == 'kl_v2':
                    wf_loss = 0
                    for i in range(self.num_layers):
                        wf_loss += self.compute_kl_loss_near_objects(fused_feats[i], t_feats[i], gt_bboxes, img_metas, weight=self.wf_loss_weight)
                    return fused_feats, wf_loss
                elif self.wf_loss_mode == 'kl_v2_t':
                    wf_loss = 0
                    for i in range(self.num_layers):
                        wf_loss += self.compute_kl_loss_near_objects(fused_feats[i], v_feats[i], gt_bboxes, img_metas, weight=self.wf_loss_weight)
                    return fused_feats, wf_loss
        return fused_feats

    def _gaussian_heatmap(self, boxes, h, w, H, W, device):
        """A CenterNet-style target map on the band grid.

        A binary box mask is the wrong target here: one band cell covers 16
        image pixels at level 0 and the objects are 8-12 pixels, so a box lands
        on one or two cells and the supervision is almost all background.  A
        Gaussian spread over a few cells gives the head something to regress and
        tolerates the fact that a tiny object's useful detail does not stop at
        the annotated boundary.
        """
        hm = torch.zeros(h, w, device=device)
        for b in boxes:
            # The peak sits on the ROUNDED centre, as CenterNet does.  Evaluating
            # the Gaussian at a fractional centre leaves every cell slightly below
            # 1, and the focal loss counts positives with gt >= 0.999 -- so the
            # positive set comes out empty and the loss is all negative terms.
            ci = int(round(float(b[0] + b[2]) / 2 * w / W))
            cj = int(round(float(b[1] + b[3]) / 2 * h / H))
            if not (0 <= ci < w and 0 <= cj < h):
                continue
            sz = max(float(b[2] - b[0]) * w / W, float(b[3] - b[1]) * h / H)
            sig = max(self.tdr_hm_min_sigma, sz / 3.0)
            r = int(3 * sig) + 1
            x0, x1 = max(ci - r, 0), min(ci + r + 1, w)
            y0, y1 = max(cj - r, 0), min(cj + r + 1, h)
            ys = torch.arange(y0, y1, device=device, dtype=torch.float32)
            xs = torch.arange(x0, x1, device=device, dtype=torch.float32)
            g = torch.exp(-(((ys[:, None] - cj) ** 2 + (xs[None, :] - ci) ** 2)
                            / (2 * sig ** 2)))
            hm[y0:y1, x0:x1] = torch.maximum(hm[y0:y1, x0:x1], g)
        return hm

    @staticmethod
    def _focal_heatmap(pred, gt, alpha=2.0, beta=4.0):
        """CenterNet's penalty-reduced focal loss.

        Chosen over plain cross-entropy for two reasons that both matter here:
        it takes a soft target, so cells near an object are not punished as hard
        as cells far from one, and the (1 - y)^beta term handles a 30:1 class
        imbalance without a hand-set class weight.
        """
        pred = pred.clamp(1e-4, 1 - 1e-4)
        pos = gt.ge(0.999).float()
        neg = 1.0 - pos
        pos_loss = torch.log(pred) * (1 - pred).pow(alpha) * pos
        neg_loss = torch.log(1 - pred) * pred.pow(alpha) * (1 - gt).pow(beta) * neg
        n = pos.sum()
        return -(pos_loss.sum() + neg_loss.sum()) / n.clamp(min=1.0)

    def compute_tdr_loss(self, bboxes, img_metas):
        """Teach R to separate objects from background.

        Left to the detection loss alone the map cannot learn: its convolution
        receives a gradient two to four orders of magnitude below the layers
        around it, and an unsupervised run converged to a constant (sd 0.0002).
        A per-image margin between object and background means was tried and
        left 11.4% of background cells above 0.5 -- it constrains the means and
        says nothing about positions.  The Gaussian-heatmap focal target below
        gives every cell an answer and brought that to 0.5%.
        """
        loss, n = 0.0, 0
        for dwtc in self.idwt_layers:
            tdr = getattr(dwtc, 'tdr', None)
            if tdr is None or tdr.target is None:
                continue
            p = tdr.target                              # (B, 1, h, w)
            B, _, h, w = p.shape
            for i in range(B):
                if bboxes[i].numel() == 0:
                    continue
                H, W = img_metas[i]['pad_shape'][:2]
                hm = self._gaussian_heatmap(bboxes[i], h, w, H, W, p.device)
                loss = loss + self._focal_heatmap(p[i, 0], hm)
                n += 1
        return self.tdr_loss_weight * loss / max(n, 1)
    def compute_kl_loss_near_objects(self, fused_x, x, bboxes, img_metas, weight=1.0):
        kl_loss = 0
        b, _, h, w = fused_x.shape

        for i in range(b):
            H, W = img_metas[i]['pad_shape'][:2]
            scale_h = h / H
            scale_w = w / W

            _bboxes = bboxes[i].clone()
            for bbox in _bboxes:
                x_center = (bbox[0] + bbox[2]) / 2
                y_center = (bbox[1] + bbox[3]) / 2
                
                new_width = (bbox[2] - bbox[0]) * 1.5
                new_height = (bbox[3] - bbox[1]) * 1.5
                
                bbox[0] = max(0, x_center - new_width / 2)
                bbox[2] = min(W, x_center + new_width / 2)
                bbox[1] = max(0, y_center - new_height / 2)
                bbox[3] = min(H, y_center + new_height / 2)

                bbox[[0, 2]] *= scale_w  # x 坐标
                bbox[[1, 3]] *= scale_h  # y 坐标

                # 取整并确保在特征图范围内
                x_min, y_min, x_max, y_max = bbox.int()
                x_min, x_max = x_min.clamp(0, w-1), x_max.clamp(0, w-1)
                y_min, y_max = y_min.clamp(0, h-1), y_max.clamp(0, h-1)

                # 提取目标区域的特征
                fused_x_region = fused_x[i, :, y_min:y_max, x_min:x_max]
                x_region = x[i, :, y_min:y_max, x_min:x_max]

                # 将目标区域特征转换为概率分布
                fused_x_region_log = F.log_softmax(fused_x_region, dim=0)  # 使用 log_softmax 作为预测分布
                x_region_soft = F.softmax(x_region, dim=0)  # 使用 softmax 作为目标分布

                # 计算 KL 散度
                kl_loss += weight * F.kl_div(fused_x_region_log, x_region_soft, reduction='batchmean')
        
        kl_loss /= b
        return kl_loss


class HOFM(BaseModule):
    def __init__(
            self, 
            in_channels,
            reduction=16,
            num_stages=0,
            use_om=True,
            use_grid=True,
            use_fixed_ga=False,
            use_msf=True,
            om_kernels=[9, 7, 5, 3, 1],
            om_range_factor =[1, 1, 1, 1, 1], 
            msf_kernels=[7, 5, 3],
            aam_align_corners=True,
            aam_warp=True):
        super(HOFM, self).__init__()
        # The reference grid below is built as (i+0.5)/W*2-1.  Under
        # align_corners=True that maps to pixel index (i+0.5)*(W-1)/W, which is
        # off-grid at every column except the centre -- so a predicted offset of
        # zero still resamples, and AAM cannot express "leave this where it is".
        # Measured on the trained champion at level 0: every column lands an
        # average of 0.25 px off, max|grid_sample(x, ref) - x| = 2.71, and the
        # head input loses 0.021 AUC on adjacent-cell separation plus 0.215 px of
        # sub-cell localisation.  align_corners=False maps the same grid to
        # exactly i (residual 5e-06), which keeps the alignment and drops the
        # blur -- it measures better than skipping the resampling altogether,
        # so the warp is doing real work once it stops smearing.
        # Both flags default to the shipped behaviour; only a config turns them on.
        self.aam_align_corners = aam_align_corners
        self.aam_warp = aam_warp
        self.in_channels = in_channels
        self.reduction = reduction
        self.num_stages = num_stages
        self.use_om = use_om
        self.use_grid = use_grid
        self.use_fixed_ga = use_fixed_ga
        self.use_msf = use_msf
        self.om_kernels = om_kernels
        self.om_range_factor = om_range_factor[num_stages]
        self.msf_kernels = msf_kernels

        self.conv = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)

        if self.use_om:
            if self.use_grid:
                self.conv_offset = nn.Sequential(
                    nn.Conv2d(
                        in_channels=in_channels, 
                        out_channels=in_channels, 
                        kernel_size=om_kernels[num_stages], 
                        stride=1, 
                        padding=om_kernels[num_stages] // 2, 
                        groups=in_channels),
                    LayerNormProxy(in_channels),
                    nn.GELU(),
                    nn.Conv2d(
                        in_channels=in_channels, 
                        out_channels=2, 
                        kernel_size=1, 
                        stride=1, 
                        padding=0, 
                        bias=False))
            else:
                self.conv_offset = nn.Sequential(
                    nn.Conv2d(
                        in_channels=in_channels, 
                        out_channels=in_channels, 
                        kernel_size=om_kernels[num_stages], 
                        stride=1, 
                        padding=om_kernels[num_stages] // 2, 
                        groups=in_channels),
                    LayerNormProxy(in_channels),
                    nn.GELU(),
                    nn.Conv2d(
                        in_channels=in_channels, 
                        out_channels=in_channels, 
                        kernel_size=1, 
                        stride=1, 
                        padding=0, 
                        bias=False))
            self.proj_offset = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)
        
        if self.use_msf:
            self.msf_conv_layers = nn.ModuleList()
            for kernel in msf_kernels:
                msf_conv = nn.Conv2d(
                    in_channels=in_channels, 
                    out_channels=in_channels // reduction, 
                    kernel_size=kernel, 
                    padding=kernel // 2)
                self.msf_conv_layers.append(msf_conv)
            self.msf_gap = nn.AdaptiveAvgPool2d(1)
            self.relu = nn.ReLU(inplace=True)
            self.msf_fc = nn.Conv2d(
                in_channels=in_channels // reduction, 
                out_channels=in_channels // reduction * 3, 
                kernel_size=1)
            self.proj_msf = nn.Conv2d(
                in_channels=in_channels // reduction * 3, 
                out_channels=in_channels, 
                kernel_size=1)
        
        self.iter = -1

    @torch.no_grad()
    def _get_ref_points(self, H_key, W_key, B, dtype, device):

        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H_key - 0.5, H_key, dtype=dtype, device=device),
            torch.linspace(0.5, W_key - 0.5, W_key, dtype=dtype, device=device)
        )
        ref = torch.stack((ref_y, ref_x), -1)
        ref[..., 1].div_(W_key).mul_(2).sub_(1)
        ref[..., 0].div_(H_key).mul_(2).sub_(1)
        ref = ref[None, ...].expand(B, -1, -1, -1)  # B * g H W 2

        return ref

    def forward(self, feat_v, feat_t):
        self.iter += 1
        H = feat_t.shape[2]
        feat_c = self.conv(torch.cat([feat_v, feat_t], dim=1))


        if self.use_om:
            proj_offset = feat_c
            offset = self.conv_offset(proj_offset)
            Hk, Wk = offset.size(2), offset.size(3)

            if self.use_grid:
                if self.om_range_factor > 0:
                    offset_range = torch.tensor(
                        [1.0 / Hk, 1.0 / Wk], 
                        device=offset.device).reshape(1, 2, 1, 1)
                    offset = offset.tanh().mul(offset_range).mul(self.om_range_factor)
            
                offset = einops.rearrange(offset, 'b c h w -> b h w c')

                vis_reference = self._get_ref_points(
                    Hk, Wk, feat_v.shape[0], feat_v.dtype, feat_v.device)
                lwir_reference = self._get_ref_points(
                    Hk, Wk, feat_v.shape[0], feat_v.dtype, feat_v.device)
                
                if self.use_fixed_ga is not True:
                    if self.om_range_factor >= 0:
                        vis_pos = vis_reference + offset
                        lwir_pos = lwir_reference
                    else:
                        vis_pos = (vis_reference + offset).tanh()
                        lwir_pos = lwir_reference.tanh()
                else:
                    vis_pos = vis_reference
                    lwir_pos = lwir_reference
                
                if self.aam_warp:
                    feat_v = F.grid_sample(
                        input=feat_v,
                        grid=vis_pos[..., (1, 0)],
                        mode='bilinear',
                        align_corners=self.aam_align_corners)

                    feat_t = F.grid_sample(
                        input=feat_t,
                        grid=lwir_pos[..., (1, 0)],
                        mode='bilinear',
                        align_corners=self.aam_align_corners)
                
                feat_c = self.proj_offset(torch.cat([feat_v, feat_t], dim=1))
            
            else:
                feat_t = feat_t + offset
                feat_c = self.proj_offset(torch.cat([feat_v, feat_t], dim=1))

        if self.use_msf:
            u_feats = []
            for msf_conv in self.msf_conv_layers:
                u_feats.append(msf_conv(feat_c))
            u_feat = sum(u_feats)

            wc = self.relu(self.msf_gap(u_feat))

            ws = self.msf_fc(wc)
            z_feats = torch.chunk(ws, len(self.msf_conv_layers), dim=1)
            a_feats = []
            for i in range(len(z_feats)):
                a_feats.append(u_feats[i] * F.softmax(z_feats[i], dim=1))
            a = self.proj_msf(torch.cat(a_feats, dim=1))

            feat_c = feat_c + a
            # if H == 128:
            #     draw_feature_map(feat_c, name='feats', iter=self.iter)
        
        return feat_c


class PoolingUpsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.maxpooling = nn.MaxPool2d(2, 2, dilation=1)
        # self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv1x1 = nn.Conv2d(in_channels*2, in_channels, 1)
    
    def forward(self, x):
        x_ = self.maxpooling(x)
        # x_ = self.upsample(x_)
        x_ = F.interpolate(x_, mode='bilinear', size=x.shape[-2:], align_corners=True)
        # import pdb; pdb.set_trace()
        x = self.conv1x1(torch.cat((x, x_), 1))
        return x


class LayerNormProxy(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = einops.rearrange(x, 'b c h w -> b h w c')
        x = self.norm(x)
        return einops.rearrange(x, 'b h w c -> b c h w')


class PoolingUpsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.maxpooling = nn.MaxPool2d(2, 2, dilation=1)
        self.conv1x1 = nn.Conv2d(in_channels*2, in_channels, 1)
    
    def forward(self, x):
        x_ = self.maxpooling(x)
        x_ = F.interpolate(x_, mode='bilinear', size=x.shape[-2:], align_corners=True)
        x = self.conv1x1(torch.cat((x, x_), 1))
        return x


import cv2
import mmcv
import numpy as np
import os
import torch
import matplotlib.pyplot as plt


def featuremap_2_heatmap(feature_map):
    assert isinstance(feature_map, torch.Tensor)
    feature_map = feature_map.detach()
    heatmap = feature_map[:,0,:,:]*0
    heatmaps = []
    for c in range(feature_map.shape[1]):
        heatmap+=feature_map[:,c,:,:]
    heatmap = heatmap.cpu().numpy()
    heatmap = np.mean(heatmap, axis=0)

    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    heatmaps.append(heatmap)

    return heatmaps


def draw_feature_map(features,save_dir = 'work_dir/rgbtdroneperson/vis_results/wavelet',name = None, iter=0):
    i=0
    file_name_list = [
        {"id": 0, "file_name": "05120.jpg", "height": 512, "width": 640}, 
        {"id": 1, "file_name": "05563.jpg", "height": 512, "width": 640}, 
        {"id": 2, "file_name": "05569.jpg", "height": 512, "width": 640}, 
        {"id": 3, "file_name": "05650.jpg", "height": 512, "width": 640}, 
        {"id": 4, "file_name": "05661.jpg", "height": 512, "width": 640}, 
        {"id": 5, "file_name": "06006.jpg", "height": 512, "width": 640}
    ]
    gt_name = file_name_list[iter]['file_name']
    img = cv2.imread(f'/data3/pengpeiran/datasets/RGBTDronePerson/val/visible/{gt_name[:-4]}.jpg', cv2.IMREAD_COLOR)
    if isinstance(features,torch.Tensor):
        for heat_maps in features:
            heat_maps=heat_maps.unsqueeze(0)
            heatmaps = featuremap_2_heatmap(heat_maps)
            # 这里的h,w指的是你想要把特征图resize成多大的尺寸
            
            for heatmap in heatmaps:
                heatmap = np.uint8(255 * heatmap)
                heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0])) 
                # 下面这行将热力图转换为RGB格式 ，如果注释掉就是灰度图
                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

                superimposed_img = heatmap
                plt.imshow(superimposed_img,cmap='gray')
                plt.show()
                cv2.imwrite(os.path.join(save_dir,name +str(i)+f'_{gt_name[:-4]}.png'), superimposed_img)

                superimposed_img = heatmap * 0.5 + img*0.3
                plt.imshow(superimposed_img,cmap='gray')
                plt.show()
                cv2.imwrite(os.path.join(save_dir,name +str(i)+f'overlay_{gt_name[:-4]}.png'), superimposed_img)
                i=i+1

    else:
        for featuremap in features:
            heatmaps = featuremap_2_heatmap(featuremap)
            # heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))  # 将热力图的大小调整为与原始图像相同
            for heatmap in heatmaps:
                heatmap = np.uint8(255 * heatmap)  # 将热力图转换为RGB格式
                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                # superimposed_img = heatmap * 0.5 + img*0.3
                superimposed_img = heatmap
                # plt.imshow(superimposed_img,cmap='gray')
                # plt.show()
                # 下面这些是对特征图进行保存，使用时取消注释
                # cv2.imshow("1",superimposed_im