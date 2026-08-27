"""COXNet Fig 3 재현 — DASR(HOFM) feature: F_V, F_T (정렬前) vs F'_V, F'_T (정렬後).

논문 Fig 3처럼 픽셀이 아니라 feature map을 본다. HOFM.forward의 grid_sample을 패치해
정렬 전후 feat_v/feat_t를 캡처, 채널평균 |활성|을 heatmap으로. GT는 초록 박스.

패널: [thermal+GT | F_V(前) | F'_V(後) | F_T(前) | F'_T(後)]
F_V→F'_V로 visible이 정렬되는지, cross-modal cos-sim(前→後) 확인.

usage:
  python tools/misc/vis_dasr_feat.py --config ... --checkpoint ... \
     --images 05540 05661 --level 0 --output-dir work_dir/dasr_feat  [--all]
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


def cos_sim(a, b):
    a = a.flatten(2); b = b.flatten(2)
    a = a / (a.norm(dim=1, keepdim=True) + 1e-6)
    b = b / (b.norm(dim=1, keepdim=True) + 1e-6)
    return float((a * b).sum(1).mean())


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

        fv_pre, fv_al = gs[2 * L]       # F_V, F'_V
        ft_pre, ft_al = gs[2 * L + 1]   # F_T, F'_T
        Hk, Wk = fv_pre.shape[2], fv_pre.shape[3]
        disp = (Wk * 8, Hk * 8)
        sx, sy = disp[0] / W0, disp[1] / H0
        gt = ds.get_ann_info(idx_of[fn])['bboxes']

        def heat(feat):
            mm = feat.abs().mean(1)[0].detach().cpu().numpy()
            mm = (mm - mm.min()) / (mm.max() - mm.min() + 1e-9)
            hm = cv2.applyColorMap(np.uint8(255 * cv2.resize(mm, disp)), cv2.COLORMAP_JET)
            for x1, y1, x2, y2 in gt:
                cv2.rectangle(hm, (int(x1 * sx), int(y1 * sy)),
                              (int(x2 * sx), int(y2 * sy)), (0, 255, 0), 1)
            return hm

        ref = cv2.resize(cv2.convertScaleAbs(th, alpha=1.4), disp)
        for x1, y1, x2, y2 in gt:
            cv2.rectangle(ref, (int(x1 * sx), int(y1 * sy)),
                          (int(x2 * sx), int(y2 * sy)), (0, 255, 0), 1)
        before = cos_sim(fv_pre, ft_pre); after = cos_sim(fv_al, ft_al)
        panel = np.hstack([lab(ref, f'{name} thermal+GT'),
                           lab(heat(fv_pre), 'F_V (before)'),
                           lab(heat(fv_al), "F'_V (after align)"),
                           lab(heat(ft_pre), 'F_T (before)'),
                           lab(heat(ft_al), "F'_T (after)")])
        s = np.full((28, panel.shape[1], 3), 30, np.uint8)
        col = (0, 255, 0) if after > before else (0, 0, 255)
        cv2.putText(s, f'P{L+3}  cross-modal cos-sim  F_V,F_T {before:.4f} -> '
                    f"F'_V,F'_T {after:.4f}  (delta {after-before:+.4f})",
                    (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 1)
        panel = np.vstack([s, panel])
        cv2.imwrite(osp.join(args.output_dir, fn), panel, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if not args.all:
            print(f'{name}: cos-sim {before:.4f} -> {after:.4f} ({after-before:+.4f})')
    print('저장:', args.output_dir)


if __name__ == '__main__':
    main()
