"""COXNet 논문 "Feature Visualization" 스타일 — 회색조 장면 + 활성 주황 glow.

JET heatmap이 아니라, 같은 thermal 장면 위에 각 feature(F_V,F'_V,F_T,F'_T)의 활성도를
따뜻한 colormap(HOT)으로 오버레이 → 장면 구조가 보이고 객체에서 glow.
AAM 전(F_V)→후(F'_V)에서 visible glow가 thermal 객체 위치로 오는지(정렬) 확인.

패널: [thermal+GT | F_V(전) | F'_V(후) | F_T(전) | F'_T(후)]

usage:
  python tools/misc/vis_dasr_paper.py --config ... --checkpoint ... \
     --images 05540 --level 0 --output-dir work_dir/dasr_paper [--all]
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


def glow(feat, base_gray, disp, gt, s, gamma=1.6, alpha=0.8):
    """활성도를 회색조 장면 위에 HOT colormap으로 오버레이 (논문식)."""
    a = feat.abs().mean(1)[0].detach().cpu().numpy()
    a = (a - a.min()) / (a.max() - a.min() + 1e-9)
    a = a ** gamma                                   # 강한 활성만 남김
    a = cv2.resize(a, disp)
    base = cv2.cvtColor(cv2.resize(base_gray, disp), cv2.COLOR_GRAY2BGR).astype(np.float32)
    hot = cv2.applyColorMap(np.uint8(255 * a), cv2.COLORMAP_HOT).astype(np.float32)
    am = (a[..., None] * alpha)
    out = (base * (1 - am) + hot * am).astype(np.uint8)
    for x1, y1, x2, y2 in gt:
        cv2.rectangle(out, (int(x1 * s[0]), int(y1 * s[1])),
                      (int(x2 * s[0]), int(y2 * s[1])), (0, 255, 0), 1)
    return out


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
        base = cv2.cvtColor(cv2.convertScaleAbs(th, alpha=1.3), cv2.COLOR_BGR2GRAY)
        ref = cv2.resize(cv2.convertScaleAbs(th, alpha=1.3), disp)
        for x1, y1, x2, y2 in gt:
            cv2.rectangle(ref, (int(x1 * s[0]), int(y1 * s[1])),
                          (int(x2 * s[0]), int(y2 * s[1])), (0, 255, 0), 1)
        panel = np.hstack([lab(ref, f'{name} thermal+GT'),
                           lab(glow(fv_pre, base, disp, gt, s), 'F_V (before AAM)'),
                           lab(glow(fv_al, base, disp, gt, s), "F'_V (after AAM)"),
                           lab(glow(ft_pre, base, disp, gt, s), 'F_T (before AAM)'),
                           lab(glow(ft_al, base, disp, gt, s), "F'_T (after AAM)")])
        cv2.imwrite(osp.join(args.output_dir, fn), panel, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if not args.all:
            print(f'{name}: paper-style saved')
    print('저장:', args.output_dir)


if __name__ == '__main__':
    main()
