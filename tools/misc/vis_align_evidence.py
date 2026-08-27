"""AAM 정렬 증거 (#1 feature heatmap + #2 offset field + MI 정량).

논문식 alignment 입증: 이미지 warp가 아니라 feature로.
패널: [thermal+GT | F_T | F_V(前) | F'_V(後) | offset quiver]
상단바: MI(F_V,F_T) -> MI(F'_V,F_T)  (증가하면 정렬 개선)
F'_V의 hotspot이 F_T(객체) 위치로 모이면 정렬 성공.

usage:
  python tools/misc/vis_align_evidence.py --config ... --checkpoint ... \
     --images 05540 05661 --level 0 --output-dir work_dir/align_evidence [--all]
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


def mag_map(feat):
    m = feat.abs().mean(1)[0].detach().cpu().numpy()
    return (m - m.min()) / (m.max() - m.min() + 1e-9)


def mutual_info(a, b, bins=16):
    a = (a.flatten() * (bins - 1)).astype(int)
    b = (b.flatten() * (bins - 1)).astype(int)
    j = np.histogram2d(a, b, bins=bins, range=[[0, bins - 1], [0, bins - 1]])[0]
    pj = j / (j.sum() + 1e-12)
    pa, pb = pj.sum(1, keepdims=True), pj.sum(0, keepdims=True)
    mask = pj > 0
    return float((pj[mask] * np.log(pj[mask] / (pa @ pb)[mask] + 1e-12)).sum())


def heat(mnorm, disp, gt, s):
    hm = cv2.applyColorMap(np.uint8(255 * cv2.resize(mnorm, disp)), cv2.COLORMAP_JET)
    for x1, y1, x2, y2 in gt:
        cv2.rectangle(hm, (int(x1 * s[0]), int(y1 * s[1])),
                      (int(x2 * s[0]), int(y2 * s[1])), (0, 255, 0), 1)
    return hm


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
    layer = m.fuse_layer.hofm_layers[L]
    omf = float(getattr(layer, 'om_range_factor', 1.0))
    mis = []

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

        gs = []; raw = {}
        ho = layer.conv_offset.register_forward_hook(lambda md, i, o: raw.__setitem__('o', o.detach()))
        orig = F.grid_sample
        def patched(input, grid, **kw):
            out = orig(input, grid, **kw); gs.append((input.detach(), out.detach())); return out
        F.grid_sample = patched
        with torch.no_grad():
            m.extract_feat([v, t])
        F.grid_sample = orig; ho.remove()

        fv_pre, fv_al = gs[2 * L]; ft = gs[2 * L + 1][0]
        Hk, Wk = fv_pre.shape[2], fv_pre.shape[3]
        disp = (Wk * 8, Hk * 8); s = (disp[0] / W0, disp[1] / H0)
        mv, mt, mva = mag_map(fv_pre), mag_map(ft), mag_map(fv_al)
        mi_b, mi_a = mutual_info(mv, mt), mutual_info(mva, mt)
        mis.append((mi_b, mi_a))

        # #2 offset quiver (downsampled)
        ro = raw['o']; tt = ro.tanh()
        sy = (tt[:, 0] * omf * 0.5)[0].cpu().numpy(); sx = (tt[:, 1] * omf * 0.5)[0].cpu().numpy()
        of = cv2.resize(cv2.convertScaleAbs(th, alpha=1.3), disp)
        for yy in range(0, Hk, 3):
            for xx in range(0, Wk, 3):
                cyv, cxv = int((yy + .5) * 8), int((xx + .5) * 8)
                cv2.arrowedLine(of, (cxv, cyv),
                                (int(cxv + sx[yy, xx] * 8 * 6), int(cyv + sy[yy, xx] * 8 * 6)),
                                (0, 255, 255), 1, tipLength=0.4)
        ref = cv2.resize(cv2.convertScaleAbs(th, alpha=1.3), disp)
        for x1, y1, x2, y2 in gt:
            cv2.rectangle(ref, (int(x1 * s[0]), int(y1 * s[1])),
                          (int(x2 * s[0]), int(y2 * s[1])), (0, 255, 0), 1)
        panel = np.hstack([lab(ref, f'{name} thermal+GT'),
                           lab(heat(mt, disp, gt, s), 'F_T (thermal)'),
                           lab(heat(mv, disp, gt, s), 'F_V before'),
                           lab(heat(mva, disp, gt, s), "F'_V after"),
                           lab(of, 'offset field x6')])
        bar = np.full((30, panel.shape[1], 3), 30, np.uint8)
        col = (0, 255, 0) if mi_a > mi_b else (0, 0, 255)
        cv2.putText(bar, f'P{L+3}  MI(F_V,F_T) {mi_b:.4f} -> MI(F\'_V,F_T) {mi_a:.4f}'
                    f'  (delta {mi_a-mi_b:+.4f})  [>0 = alignment raised correspondence]',
                    (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.52, col, 1)
        cv2.imwrite(osp.join(args.output_dir, fn), np.vstack([bar, panel]),
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
        if not args.all:
            print(f'{name}: MI {mi_b:.4f} -> {mi_a:.4f} ({mi_a-mi_b:+.4f})')
    if mis:
        mb = np.mean([x[0] for x in mis]); ma = np.mean([x[1] for x in mis])
        up = sum(a > b for b, a in mis)
        print(f'\n=== 전체 {len(mis)}장 MI: before {mb:.4f} -> after {ma:.4f} '
              f'(delta {ma-mb:+.4f}) | 개선된 이미지 {up}/{len(mis)} ({up/len(mis)*100:.0f}%) ===')
    print('저장:', args.output_dir)


if __name__ == '__main__':
    main()
