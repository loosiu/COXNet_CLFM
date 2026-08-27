"""AAM 정렬을 이미지에 적용 — 원본 RGB/thermal vs AAM warp된 RGB (전체 val, 추론).

AAM(HOFM)은 추론 시 visible feature를 grid_sample(vis_pos)로 warp함(offset 예측).
그 P3 offset을 이미지 해상도로 올려 RGB에 적용 → "AAM이 가하는 공간 보정"을 이미지로 확인.
thermal은 AAM에서 identity(lwir_pos=reference)라 안 바뀜.

패널: [RGB 원본 | thermal 원본 | RGB(AAM warp) | thermal(불변)]
라벨에 평균 offset(input-px).

usage:
  python tools/misc/vis_aam_warp.py --config ... --checkpoint ... \
     --level 0 --all --output-dir work_dir/aam_warp
"""
import argparse
import os.path as osp

import cv2
import mmcv
import numpy as np
import torch
import torch.nn.functional as F
from mmcv import Config
from mmcv.runner import load_checkpoint
from mmdet.models import build_detector
from mmdet.datasets import build_dataset


def lab(img, t):
    bar = np.full((26, img.shape[1], 3), 40, np.uint8)
    cv2.putText(bar, t, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    return np.vstack([bar, img])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True); p.add_argument('--checkpoint', required=True)
    p.add_argument('--images', nargs='+'); p.add_argument('--all', action='store_true')
    p.add_argument('--level', type=int, default=0)
    p.add_argument('--amp', type=float, default=3.0, help='변위 시각화 증폭배율')
    p.add_argument('--output-dir', required=True)
    args = p.parse_args()

    cfg = Config.fromfile(args.config); cfg.model.backbone.init_cfg = None
    m = build_detector(cfg.model).cuda(); load_checkpoint(m, args.checkpoint, map_location='cpu'); m.eval()
    ds = build_dataset(cfg.data.test, dict(test_mode=True))
    idx_of = {ds.data_infos[i]['filename']: i for i in range(len(ds))}
    img_prefix = cfg.data.test.img_prefix
    tr = cfg.data.test.pipeline[1]['transforms'][2]
    mean = np.array(tr['mean_list'][0]); std = np.array(tr['std_list'][0])
    mean_t = np.array(tr['mean_list'][1]); std_t = np.array(tr['std_list'][1])
    mmcv.mkdir_or_exist(args.output_dir)
    L = args.level
    layer = m.fuse_layer.hofm_layers[L]
    omf = float(getattr(layer, 'om_range_factor', 1.0))

    def find(pp, fn):
        for pth in (osp.join(osp.dirname(img_prefix), pp[0], 'images',
                             osp.basename(img_prefix), fn), osp.join(img_prefix, pp[1], fn)):
            if osp.exists(pth):
                return pth

    names = ([ds.data_infos[i]['filename'][:-4] for i in range(len(ds))]
             if args.all else args.images)
    for name in (mmcv.track_iter_progress(names) if args.all else names):
        fn = name + '.jpg'
        rgb = cv2.imread(find(('visible', 'visible'), fn))
        tp = find(('infrared', 'thermal'), fn)
        th = cv2.resize(cv2.imread(tp), (rgb.shape[1], rgb.shape[0])) if tp else rgb
        H0, W0 = rgb.shape[:2]
        v = cv2.resize(rgb, (640, 512))[:, :, ::-1].astype(np.float32)
        t = cv2.resize(th, (640, 512))[:, :, ::-1].astype(np.float32)
        v = torch.from_numpy(((v - mean) / std).transpose(2, 0, 1)[None]).float().cuda()
        t = torch.from_numpy(((t - mean_t) / std_t).transpose(2, 0, 1)[None]).float().cuda()

        raw = {}
        ho = layer.conv_offset.register_forward_hook(lambda md, i, o: raw.__setitem__('o', o.detach()))
        with torch.no_grad():
            m.extract_feat([v, t])
        ho.remove()
        ro = raw['o']; Hk, Wk = ro.shape[2], ro.shape[3]
        tt = ro.tanh()
        off_y = tt[:, 0:1] * (1.0 / Hk) * omf      # (1,1,Hk,Wk) normalized grid
        off_x = tt[:, 1:2] * (1.0 / Wk) * omf
        # 이미지 해상도로 업샘플
        off_y = F.interpolate(off_y, size=(H0, W0), mode='bilinear', align_corners=True)[0, 0]
        off_x = F.interpolate(off_x, size=(H0, W0), mode='bilinear', align_corners=True)[0, 0]
        # 이미지 기준 identity grid + offset
        ys = torch.linspace(-1, 1, H0, device=off_y.device)
        xs = torch.linspace(-1, 1, W0, device=off_x.device)
        gy, gx = torch.meshgrid(ys, xs)
        grid = torch.stack([gx + off_x, gy + off_y], dim=-1)[None]   # (1,H0,W0,2) (x,y)
        rgb_t = torch.from_numpy(rgb[:, :, ::-1].transpose(2, 0, 1)[None].copy()).float().cuda()
        warped = F.grid_sample(rgb_t, grid, mode='bilinear', align_corners=True,
                               padding_mode='border')[0].cpu().numpy().transpose(1, 2, 0)
        warped = warped[:, :, ::-1].astype(np.uint8)   # RGB→BGR

        offmag = torch.sqrt((tt[:, 0] * omf * 0.5) ** 2 + (tt[:, 1] * omf * 0.5) ** 2)
        mpx = float(offmag.mean()) * 8   # input-px (P3 stride 8)
        sy = (tt[:, 0] * omf * 0.5)[0].cpu().numpy()   # (Hk,Wk) feature-px
        sx = (tt[:, 1] * omf * 0.5)[0].cpu().numpy()

        gt = ds.get_ann_info(idx_of[fn])['bboxes']
        AMP = args.amp

        def draw_gt(img):
            o = img.copy()
            for x1, y1, x2, y2 in gt.astype(int):
                cv2.rectangle(o, (x1, y1), (x2, y2), (0, 255, 0), 1)
            return o

        def draw_shift(img):
            # GT중심(=정렬 목표, white) ← AAM이 RGB를 샘플링하는 원위치(yellow) 에서 이동
            o = img.copy()
            for x1, y1, x2, y2 in gt:
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                fx = min(Wk - 1, max(0, int(cx / W0 * Wk)))
                fy = min(Hk - 1, max(0, int(cy / H0 * Hk)))
                dpx = sx[fy, fx] * (W0 / Wk) * AMP     # 원본px 변위 (증폭)
                dpy = sy[fy, fx] * (H0 / Hk) * AMP
                yx, yy = int(cx + dpx), int(cy + dpy)   # yellow = 원래 RGB 위치(AAM 기준)
                cv2.rectangle(o, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)
                cv2.line(o, (yx, yy), (cx, cy), (255, 255, 0), 1)   # cyan 이동선
                cv2.circle(o, (yx, yy), 3, (0, 255, 255), -1)       # yellow = orig
                cv2.circle(o, (cx, cy), 3, (255, 255, 255), -1)     # white = AAM 정렬후(GT)
            return o
        panel = np.hstack([
            lab(draw_shift(rgb), f'{name} RGB  GT=green yellow=orig white=AAM(shift x{AMP:.0f})'),
            lab(draw_gt(th), 'thermal (orig)'),
            lab(draw_shift(warped), f'RGB warped (~{mpx:.1f}px)'),
            lab(draw_gt(th), 'thermal (unchanged by AAM)')])
        cv2.imwrite(osp.join(args.output_dir, fn), panel, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not args.all:
            print(f'{name}: mean offset {mpx:.2f} input-px')
    print('저장:', args.output_dir)


if __name__ == '__main__':
    main()
