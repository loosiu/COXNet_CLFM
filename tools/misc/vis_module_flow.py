"""모듈별 feature 흐름 정성 검증 — 각 모듈이 feature를 어떻게 바꾸나.

파이프라인 경계마다 feature 캡처:
  Visible FPN → [CLFM] → F_V → [AAM warp] → F'_V → [MSF] → Fused
각 단계 회색조 + GT(초록) 오버레이. 객체 구조가 단계별로 좋아지나/나빠지나로
어느 모듈이 제대로 작동/약점인지 판단.

패널: [thermal+GT | Visible FPN | F_V (CLFM) | F'_V (AAM) | Fused (MSF)]

usage:
  python tools/misc/vis_module_flow.py --config ... --checkpoint ... \
     --images 05540 05661 --level 0 --output-dir work_dir/module_flow [--all]
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
    cv2.putText(bar, t, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
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

        cap = {}
        hs = []
        # CLFM (idwt_layers[L]): pre → visible FPN 입력, post → CLFM 출력 F_V
        hs.append(m.fuse_layer.idwt_layers[L].register_forward_pre_hook(
            lambda mod, inp: cap.__setitem__('vfpn', inp[1].detach())))
        hs.append(m.fuse_layer.idwt_layers[L].register_forward_hook(
            lambda mod, inp, out: cap.__setitem__('fv', out.detach())))
        # HOFM 출력 = MSF 후 fused
        hs.append(m.fuse_layer.hofm_layers[L].register_forward_hook(
            lambda mod, inp, out: cap.__setitem__('fused', out.detach())))
        # grid_sample patch → F'_V
        gs = []
        orig = F.grid_sample
        def patched(input, grid, **kw):
            o = orig(input, grid, **kw); gs.append((input.detach(), o.detach())); return o
        F.grid_sample = patched
        with torch.no_grad():
            m.extract_feat([v, t])
        F.grid_sample = orig
        for h in hs:
            h.remove()

        fpv = gs[2 * L][1]   # F'_V
        Hk, Wk = cap['fv'].shape[2], cap['fv'].shape[3]
        disp = (Wk * 8, Hk * 8); s = (disp[0] / W0, disp[1] / H0)
        ref = cv2.resize(cv2.convertScaleAbs(th, alpha=1.3), disp)
        for x1, y1, x2, y2 in gt:
            cv2.rectangle(ref, (int(x1 * s[0]), int(y1 * s[1])),
                          (int(x2 * s[0]), int(y2 * s[1])), (0, 255, 0), 1)
        panel = np.hstack([lab(ref, f'{name} thermal+GT'),
                           lab(gray(cap['vfpn'], disp, gt, s), 'Visible FPN (input)'),
                           lab(gray(cap['fv'], disp, gt, s), 'F_V  after CLFM'),
                           lab(gray(fpv, disp, gt, s), "F'_V  after AAM"),
                           lab(gray(cap['fused'], disp, gt, s), 'Fused  after MSF')])
        cv2.imwrite(osp.join(args.output_dir, fn), panel, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if not args.all:
            print(f'{name}: module flow saved')
    print('저장:', args.output_dir)


if __name__ == '__main__':
    main()
