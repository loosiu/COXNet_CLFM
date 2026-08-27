"""COXNet Fig(c) DASR — AAM 직전(F_V,F_T)·직후(F'_V,F'_T) feature를 PCA-RGB로.

feature는 256채널 추상 텐서라 입력 이미지처럼 못 봄. PCA로 256→3채널 투영해
컬러 이미지로 표시(구조가 보임, 파란 heatmap 아님). 모달별로 before에서 PCA basis를
학습해 after에 동일 적용 → before/after 색 비교 가능.

패널: [thermal+GT | F_V(전) | F'_V(후) | F_T(전) | F'_T(후)]  (PCA-RGB)

usage:
  python tools/misc/vis_dasr_pca.py --config ... --checkpoint ... \
     --images 05540 05661 --level 0 --output-dir work_dir/dasr_pca [--all]
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


def pca_rgb(feat, disp, gt, s, basis=None):
    """feat (1,C,H,W) → PCA 3채널 RGB 이미지. basis 없으면 학습, 있으면 적용."""
    C, H, W = feat.shape[1], feat.shape[2], feat.shape[3]
    x = feat[0].reshape(C, H * W).T.detach().cpu().numpy().astype(np.float32)  # (HW,C)
    mu = x.mean(0, keepdims=True)
    xc = x - mu
    if basis is None:
        _, _, Vt = np.linalg.svd(xc, full_matrices=False)
        basis = Vt[:3]                       # (3,C)
    y = xc @ basis.T                          # (HW,3)
    lo = np.percentile(y, 2, axis=0); hi = np.percentile(y, 98, axis=0)
    y = np.clip((y - lo) / (hi - lo + 1e-9), 0, 1)
    img = (y.reshape(H, W, 3) * 255).astype(np.uint8)
    img = cv2.resize(img, disp, interpolation=cv2.INTER_NEAREST)
    for x1, y1, x2, y2 in gt:
        cv2.rectangle(img, (int(x1 * s[0]), int(y1 * s[1])),
                      (int(x2 * s[0]), int(y2 * s[1])), (0, 255, 0), 1)
    return img, basis


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
        # 모달별 basis: before에서 학습해 after에 적용
        pv_pre, bv = pca_rgb(fv_pre, disp, gt, s)
        pv_al, _ = pca_rgb(fv_al, disp, gt, s, basis=bv)
        pt_pre, bt = pca_rgb(ft_pre, disp, gt, s)
        pt_al, _ = pca_rgb(ft_al, disp, gt, s, basis=bt)
        ref = cv2.resize(cv2.convertScaleAbs(th, alpha=1.3), disp)
        for x1, y1, x2, y2 in gt:
            cv2.rectangle(ref, (int(x1 * s[0]), int(y1 * s[1])),
                          (int(x2 * s[0]), int(y2 * s[1])), (0, 255, 0), 1)
        panel = np.hstack([lab(ref, f'{name} thermal+GT'),
                           lab(pv_pre, 'F_V (before AAM)'),
                           lab(pv_al, "F'_V (after AAM)"),
                           lab(pt_pre, 'F_T (before AAM)'),
                           lab(pt_al, "F'_T (after AAM)")])
        cv2.imwrite(osp.join(args.output_dir, fn), panel, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if not args.all:
            print(f'{name}: PCA-RGB saved')
    print('저장:', args.output_dir)


if __name__ == '__main__':
    main()
