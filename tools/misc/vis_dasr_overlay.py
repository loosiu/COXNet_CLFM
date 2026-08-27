"""객체 정렬 확인 — visible/thermal 활성을 색으로 겹쳐보기 (registration 표준).

빨강=visible 활성, 초록=thermal 활성. 객체에서 겹치면 노랑(정렬), 어긋나면 분리.
AAM 전(F_V+F_T) vs 후(F'_V+F_T): 후에 노랑↑이면 정렬 성공.

패널: [thermal+GT | overlay 전(F_V+F_T) | overlay 후(F'_V+F_T)]

usage:
  python tools/misc/vis_dasr_overlay.py --config ... --checkpoint ... \
     --images 05540 --level 0 --output-dir work_dir/dasr_overlay [--all]
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


def act(feat, disp, gamma=1.6):
    a = feat.abs().mean(1)[0].detach().cpu().numpy()
    a = (a - a.min()) / (a.max() - a.min() + 1e-9)
    return cv2.resize(a ** gamma, disp)


def overlay(va, ta, gt, s):
    o = np.zeros((va.shape[0], va.shape[1], 3), np.float32)
    o[..., 2] = va * 255      # R = visible
    o[..., 1] = ta * 255      # G = thermal
    o = o.astype(np.uint8)
    for x1, y1, x2, y2 in gt:
        cv2.rectangle(o, (int(x1 * s[0]), int(y1 * s[1])),
                      (int(x2 * s[0]), int(y2 * s[1])), (255, 255, 255), 1)
    return o


def lab(img, t):
    bar = np.full((26, img.shape[1], 3), 40, np.uint8)
    cv2.putText(bar, t, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    return np.vstack([bar, img])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True); p.add_argument('--checkpoint', required=True)
    p.add_argument('--images', nargs='+'); p.add_argument('--all', action='store_true')
    p.add_argument('--level', type=int, default=0)
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
        v = cv2.resize(rgb, (640, 512))[:, :, ::-1].astype(np.float32)
        t = cv2.resize(th, (640, 512))[:, :, ::-1].astype(np.float32)
        v = torch.from_numpy(((v - mean) / std).transpose(2, 0, 1)[None]).float().cuda()
        t = torch.from_numpy(((t - mean_t) / std_t).transpose(2, 0, 1)[None]).float().cuda()

        gs = []
        orig = F.grid_sample
        def patched(input, grid, **kw):
            out = orig(input, grid, **kw); gs.append((input.detach(), out.detach())); return out
        F.grid_sample = patched
        with torch.no_grad():
            m.extract_feat([v, t])
        F.grid_sample = orig

        fv_pre, fv_al = gs[2 * L]; ft_pre, ft_al = gs[2 * L + 1]
        Hk, Wk = fv_pre.shape[2], fv_pre.shape[3]
        disp = (Wk * 8, Hk * 8); s = (disp[0] / W0, disp[1] / H0)
        va0, va1 = act(fv_pre, disp), act(fv_al, disp)
        ta = act(ft_pre, disp)
        ref = cv2.resize(cv2.convertScaleAbs(th, alpha=1.3), disp)
        for x1, y1, x2, y2 in gt:
            cv2.rectangle(ref, (int(x1 * s[0]), int(y1 * s[1])),
                          (int(x2 * s[0]), int(y2 * s[1])), (0, 255, 0), 1)
        panel = np.hstack([lab(ref, f'{name} thermal+GT'),
                           lab(overlay(va0, ta, gt, s), 'BEFORE  R=visible G=thermal (yellow=aligned)'),
                           lab(overlay(va1, ta, gt, s), 'AFTER AAM  R=F\'_V G=F_T')])
        cv2.imwrite(osp.join(args.output_dir, fn), panel, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if not args.all:
            print(f'{name}: overlay saved')
    print('저장:', args.output_dir)


if __name__ == '__main__':
    main()
