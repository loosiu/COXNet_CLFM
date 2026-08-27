"""AAM 전후 원본 feature(F_V,F_T,F'_V,F'_T)를 회색조로 충실하게 시각화 + 저장.

색 트릭 없이 채널평균 활성 크기를 회색조(밝을수록 활성)로, percentile(2~98%)
정규화해 구조가 보이게. "출력 그대로"에 가장 가까운 중립 렌더.
--dump-npy: 원본 텐서(F_V,F_T,F'_V,F'_T).npy도 저장.

패널: [thermal+GT | F_V(전) | F'_V(후) | F_T(전) | F'_T(후)]

usage:
  python tools/misc/vis_dasr_raw.py --config ... --checkpoint ... \
     --level 0 --all --output-dir work_dir/dasr_raw [--dump-npy]
"""
import argparse
import os
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


def gray(feat, disp, gt, s):
    a = feat.abs().mean(1)[0].detach().cpu().numpy()
    lo, hi = np.percentile(a, 2), np.percentile(a, 98)
    a = np.clip((a - lo) / (hi - lo + 1e-9), 0, 1)
    g = cv2.cvtColor((cv2.resize(a, disp) * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    for x1, y1, x2, y2 in gt:
        cv2.rectangle(g, (int(x1 * s[0]), int(y1 * s[1])),
                      (int(x2 * s[0]), int(y2 * s[1])), (0, 255, 0), 1)
    return g


def lab(img, t):
    bar = np.full((26, img.shape[1], 3), 40, np.uint8)
    cv2.putText(bar, t, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    return np.vstack([bar, img])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True); p.add_argument('--checkpoint', required=True)
    p.add_argument('--images', nargs='+'); p.add_argument('--all', action='store_true')
    p.add_argument('--level', type=int, default=0)
    p.add_argument('--dump-npy', action='store_true', help='원본 텐서 .npy 저장')
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
    if args.dump_npy:
        mmcv.mkdir_or_exist(osp.join(args.output_dir, 'npy'))
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
        ref = cv2.resize(cv2.convertScaleAbs(th, alpha=1.3), disp)
        for x1, y1, x2, y2 in gt:
            cv2.rectangle(ref, (int(x1 * s[0]), int(y1 * s[1])),
                          (int(x2 * s[0]), int(y2 * s[1])), (0, 255, 0), 1)
        panel = np.hstack([lab(ref, f'{name} thermal+GT'),
                           lab(gray(fv_pre, disp, gt, s), 'F_V (before AAM)'),
                           lab(gray(fv_al, disp, gt, s), "F'_V (after AAM)"),
                           lab(gray(ft_pre, disp, gt, s), 'F_T (before AAM)'),
                           lab(gray(ft_al, disp, gt, s), "F'_T (after AAM)")])
        cv2.imwrite(osp.join(args.output_dir, fn), panel, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if args.dump_npy:
            np.savez_compressed(osp.join(args.output_dir, 'npy', name + '.npz'),
                                F_V=fv_pre[0].cpu().numpy(), F_V_after=fv_al[0].cpu().numpy(),
                                F_T=ft_pre[0].cpu().numpy(), F_T_after=ft_al[0].cpu().numpy())
        if not args.all:
            print(f'{name}: raw feature saved (shape {tuple(fv_pre.shape)})')
    print('저장:', args.output_dir)


if __name__ == '__main__':
    main()
