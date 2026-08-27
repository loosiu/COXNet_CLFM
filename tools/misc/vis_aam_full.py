"""미정렬 + AAM 효과를 한 패널에 (train dual-GT).

[Thermal+Thermal-GT | RGB+Visible-GT | F_V(전) | F'_V(후) | diff |F'_V-F_V|]
앞2: 객체가 두 모달에서 다른 위치(미정렬). 뒤3: AAM이 feature를 실제로 뭘 바꾸나.
feature엔 thermal-GT(초록)·visible-GT(파랑) 박스 오버레이 → 정렬 목표/실제 대비.

usage:
  python tools/misc/vis_aam_full.py --config ... --checkpoint ... \
     --root data/RGBTDronePerson --images 00003 00005 --level 0 --output-dir work_dir/aam_full [--all]
"""
import argparse
import json
import os.path as osp

import cv2
import mmcv
import numpy as np
import torch
import torch.nn.functional as F
from mmcv import Config
from mmcv.runner import load_checkpoint
from mmdet.models import build_detector


def gray(feat, disp):
    a = feat.abs().mean(1)[0].detach().cpu().numpy()
    lo, hi = np.percentile(a, 2), np.percentile(a, 98)
    a = np.clip((a - lo) / (hi - lo + 1e-9), 0, 1)
    return cv2.cvtColor((cv2.resize(a, disp) * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)


def lab(img, t):
    bar = np.full((26, img.shape[1], 3), 40, np.uint8)
    cv2.putText(bar, t, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1)
    return np.vstack([bar, img])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True); p.add_argument('--checkpoint', required=True)
    p.add_argument('--root', default='data/RGBTDronePerson')
    p.add_argument('--images', nargs='+'); p.add_argument('--all', action='store_true')
    p.add_argument('--level', type=int, default=0)
    p.add_argument('--output-dir', required=True)
    args = p.parse_args()

    cfg = Config.fromfile(args.config); cfg.model.backbone.init_cfg = None
    m = build_detector(cfg.model).cuda(); load_checkpoint(m, args.checkpoint, map_location='cpu'); m.eval()
    tr = cfg.data.test.pipeline[1]['transforms'][2]
    mean = np.array(tr['mean_list'][0]); std = np.array(tr['std_list'][0])
    mean_t = np.array(tr['mean_list'][1]); std_t = np.array(tr['std_list'][1])
    mmcv.mkdir_or_exist(args.output_dir)
    L = args.level

    Vj = json.load(open(osp.join(args.root, 'sub_train_visible.json')))
    Tj = json.load(open(osp.join(args.root, 'sub_train_thermal.json')))
    vimg = {i['id']: i['file_name'] for i in Vj['images']}
    timg = {i['id']: i['file_name'] for i in Tj['images']}

    def by_file(js, img):
        d = {}
        for a in js['annotations']:
            d.setdefault(img[a['image_id']], []).append(a['bbox'])
        return d
    vb, tb = by_file(Vj, vimg), by_file(Tj, timg)
    common = sorted(set(vb) & set(tb))
    names = common if args.all else [n if n.endswith('.jpg') else n + '.jpg' for n in (args.images or [])]

    def draw_box(im, b, col, sc=(1, 1)):
        cv2.rectangle(im, (int(b[0] * sc[0]), int(b[1] * sc[1])),
                      (int((b[0] + b[2]) * sc[0]), int((b[1] + b[3]) * sc[1])), col, 1)

    for fn in (mmcv.track_iter_progress(names) if args.all else names):
        if fn not in vb or fn not in tb:
            continue
        rgb = cv2.imread(osp.join(args.root, 'train', 'visible', fn))
        th = cv2.imread(osp.join(args.root, 'train', 'thermal', fn))
        if rgb is None or th is None:
            continue
        th = cv2.resize(th, (rgb.shape[1], rgb.shape[0]))
        H0, W0 = rgb.shape[:2]
        v = cv2.resize(rgb, (640, 512))[:, :, ::-1].astype(np.float32)
        t = cv2.resize(th, (640, 512))[:, :, ::-1].astype(np.float32)
        vt = torch.from_numpy(((v - mean) / std).transpose(2, 0, 1)[None]).float().cuda()
        tt = torch.from_numpy(((t - mean_t) / std_t).transpose(2, 0, 1)[None]).float().cuda()
        gs = []
        orig = F.grid_sample
        def patched(input, grid, **kw):
            out = orig(input, grid, **kw); gs.append((input.detach(), out.detach())); return out
        F.grid_sample = patched
        with torch.no_grad():
            m.extract_feat([vt, tt])
        F.grid_sample = orig
        fv_pre, fv_al = gs[2 * L]
        Hk, Wk = fv_pre.shape[2], fv_pre.shape[3]
        disp = (Wk * 8, Hk * 8)
        sc = (disp[0] / W0, disp[1] / H0)

        # 1) thermal + thermal GT(초록)
        thp = cv2.convertScaleAbs(th, alpha=1.3)
        for b in tb[fn]:
            draw_box(thp, b, (0, 255, 0))
        # 2) RGB + visible GT(초록) — 밝기 적응형(대낮 과다노출 방지)
        _br = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY).mean() / 255
        _gain = float(np.clip(0.45 / max(_br, 1e-3), 0.6, 4.0))
        rgp = cv2.convertScaleAbs(rgb, alpha=_gain)
        for b in vb[fn]:
            draw_box(rgp, b, (0, 255, 0))
        # 3-4) F_V, F'_V (회색) + thermal GT(초록)·visible GT(파랑)
        fvp = gray(fv_pre, disp); fap = gray(fv_al, disp)
        for im in (fvp, fap):
            for b in tb[fn]:
                draw_box(im, b, (0, 255, 0), sc)
            for b in vb[fn]:
                draw_box(im, b, (255, 120, 0), sc)
        # 5) diff |F'_V - F_V|
        d = (fv_al - fv_pre).abs().mean(1)[0].detach().cpu().numpy()
        d = (d - d.min()) / (d.max() - d.min() + 1e-9)
        dp = cv2.applyColorMap((cv2.resize(d, disp) * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
        for b in tb[fn]:
            draw_box(dp, b, (0, 255, 0), sc)

        panel = np.hstack([lab(thp, f'{fn[:-4]} thermal + Thermal-GT'),
                           lab(rgp, 'RGB + Visible-GT'),
                           lab(fvp, 'F_V (before AAM)  green=TGT blue=VGT'),
                           lab(fap, "F'_V (after AAM)"),
                           lab(dp, "diff |F'_V - F_V|")])
        cv2.imwrite(osp.join(args.output_dir, fn), panel, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if not args.all:
            print(f'{fn[:-4]}: saved')
    print('저장:', args.output_dir)


if __name__ == '__main__':
    main()
