"""AAM 전후를 객체별 확대(zoom)로 확실히 보기.

작은 객체(2~20px) + AAM warp(~5px)는 전체 이미지에선 안 보이므로, GT 객체마다
주변을 crop해 크게 확대. 객체당 한 행:
  [thermal 확대 | RGB 정렬前 확대 | RGB 정렬後(AAM warp) 확대]
각 crop에 GT 초록박스 + 중심 십자선(고정 기준). RGB前→後에서 객체가 이동하는지 확인.

usage:
  python tools/misc/vis_aam_zoom.py --config ... --checkpoint ... \
     --images 05540 05661 --level 0 --output-dir work_dir/aam_zoom [--all] [--max-obj 6]
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

Z = 200   # 확대 crop 표시 크기


def lab(img, t):
    bar = np.full((24, img.shape[1], 3), 40, np.uint8)
    cv2.putText(bar, t, (5, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return np.vstack([bar, img])


def crop_zoom(img, cx, cy, half, gt_box, cross=True, yellow_box=None):
    H, W = img.shape[:2]
    x0, y0 = max(0, cx - half), max(0, cy - half)
    x1, y1 = min(W, cx + half), min(H, cy + half)
    c = img[y0:y1, x0:x1].copy()
    if c.size == 0:
        c = np.zeros((Z, Z, 3), np.uint8)
    s = Z / max(c.shape[0], c.shape[1])
    c = cv2.resize(c, (int(c.shape[1] * s), int(c.shape[0] * s)),
                   interpolation=cv2.INTER_NEAREST)
    c = cv2.copyMakeBorder(c, 0, max(0, Z - c.shape[0]), 0, max(0, Z - c.shape[1]),
                           cv2.BORDER_CONSTANT, value=(30, 30, 30))[:Z, :Z]

    def box(b, col):
        cv2.rectangle(c, (int((b[0] - x0) * s), int((b[1] - y0) * s)),
                      (int((b[2] - x0) * s), int((b[3] - y0) * s)), col, 1)
    box(gt_box, (0, 255, 0))                 # 초록 = GT(thermal 위치)
    if yellow_box is not None:               # 노랑 = AAM이 본 RGB 위치(GT+offset)
        box(yellow_box, (0, 255, 255))
    if cross:  # 객체 중심 고정 십자선(빨강)
        px, py = int((cx - x0) * s), int((cy - y0) * s)
        cv2.line(c, (px - 8, py), (px + 8, py), (0, 0, 255), 1)
        cv2.line(c, (px, py - 8), (px, py + 8), (0, 0, 255), 1)
    return c


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True); p.add_argument('--checkpoint', required=True)
    p.add_argument('--images', nargs='+'); p.add_argument('--all', action='store_true')
    p.add_argument('--level', type=int, default=0)
    p.add_argument('--max-obj', type=int, default=6)
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
        gt = ds.get_ann_info(idx_of[fn])['bboxes']
        if not len(gt):
            continue
        v = cv2.resize(rgb, (640, 512))[:, :, ::-1].astype(np.float32)
        t = cv2.resize(th, (640, 512))[:, :, ::-1].astype(np.float32)
        v = torch.from_numpy(((v - mean) / std).transpose(2, 0, 1)[None]).float().cuda()
        t = torch.from_numpy(((t - mean_t) / std_t).transpose(2, 0, 1)[None]).float().cuda()

        raw = {}
        ho = layer.conv_offset.register_forward_hook(lambda md, i, o: raw.__setitem__('o', o.detach()))
        with torch.no_grad():
            m.extract_feat([v, t])
        ho.remove()
        ro = raw['o']; Hk, Wk = ro.shape[2], ro.shape[3]; tt = ro.tanh()
        off_y = F.interpolate(tt[:, 0:1] * (1.0 / Hk) * omf, size=(H0, W0),
                              mode='bilinear', align_corners=True)[0, 0]
        off_x = F.interpolate(tt[:, 1:2] * (1.0 / Wk) * omf, size=(H0, W0),
                              mode='bilinear', align_corners=True)[0, 0]
        ys = torch.linspace(-1, 1, H0, device=off_y.device)
        xs = torch.linspace(-1, 1, W0, device=off_x.device)
        gy, gx = torch.meshgrid(ys, xs)
        grid = torch.stack([gx + off_x, gy + off_y], dim=-1)[None]
        rgb_t = torch.from_numpy(rgb[:, :, ::-1].transpose(2, 0, 1)[None].copy()).float().cuda()
        warped = F.grid_sample(rgb_t, grid, mode='bilinear', align_corners=True,
                               padding_mode='border')[0].cpu().numpy().transpose(1, 2, 0)
        warped = warped[:, :, ::-1].astype(np.uint8)
        oxn = off_x.cpu().numpy(); oyn = off_y.cpu().numpy()   # 정규화 offset (H0,W0)

        rows = []
        for k, (x1, y1, x2, y2) in enumerate(gt[:args.max_obj]):
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            half = int(max(24, (x2 - x1) * 2, (y2 - y1) * 2))
            # 노랑 = GT + AAM offset (AAM이 본 RGB 위치)
            iy, ix = min(H0 - 1, max(0, cy)), min(W0 - 1, max(0, cx))
            dpx = float(oxn[iy, ix]) * (W0 / 2); dpy = float(oyn[iy, ix]) * (H0 / 2)
            ybox = (x1 + dpx, y1 + dpy, x2 + dpx, y2 + dpy)
            tc = crop_zoom(cv2.convertScaleAbs(th, alpha=1.3), cx, cy, half, (x1, y1, x2, y2))
            bc = crop_zoom(rgb, cx, cy, half, (x1, y1, x2, y2), yellow_box=ybox)
            ac = crop_zoom(warped, cx, cy, half, (x1, y1, x2, y2))
            row = np.hstack([lab(tc, f'obj{k+1} thermal (GT green)'),
                             lab(bc, 'RGB before (green=GT, yellow=AAM-est)'),
                             lab(ac, 'RGB after warp (green=GT)')])
            rows.append(row)
        panel = np.vstack(rows)
        cv2.imwrite(osp.join(args.output_dir, fn), panel, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if not args.all:
            print(f'{name}: {min(len(gt), args.max_obj)} objects zoomed')
    print('저장:', args.output_dir)


if __name__ == '__main__':
    main()
