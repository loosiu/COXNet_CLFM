"""IGBS 조명 prior(LLR)·밴드 게이트 시각화 — 학습 전/후 설계 검증용.

설계가 맞다면: 야간 이미지의 어두운 곳 + 조명 포화 지점(가로등/헤드라이트)에서
LLR이 밝게(→thermal HF 선택), 주간 이미지는 전반적으로 낮게(→visible HF) 나온다.

usage:
  python tools/misc/vis_igbs_gate.py \
     --config configs/coxnet/coxnet_igbs_r50_fpn_1x_rgbtdroneperson.py \
     [--checkpoint <pth>] --images 06007 06025 --output-dir work_dir/vis_igbs
  (--checkpoint 생략 시 init 가중치 — LLR은 해석적이라 학습 전에도 유효)
"""
import argparse
import os.path as osp

import cv2
import mmcv
import numpy as np
import torch
from mmcv import Config
from mmcv.runner import load_checkpoint
from mmdet.models import build_detector
from mmdet.datasets import build_dataset


def build(cfg_path, ckpt):
    cfg = Config.fromfile(cfg_path)
    cfg.model.backbone.init_cfg = None
    m = build_detector(cfg.model).cuda()
    if ckpt:
        load_checkpoint(m, ckpt, map_location='cpu')
    m.eval()
    return m, cfg


def overlay(img, heat, alpha=0.5):
    """heat: (H,W) in [0,1] -> jet overlay on img."""
    h = cv2.resize((heat * 255).astype(np.uint8), (img.shape[1], img.shape[0]))
    h = cv2.applyColorMap(h, cv2.COLORMAP_JET)
    return cv2.addWeighted(img, 1 - alpha, h, alpha, 0)


def lab(img, t):
    bar = np.full((26, img.shape[1], 3), 40, np.uint8)
    cv2.putText(bar, t, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return np.vstack([bar, img])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--checkpoint', default=None)
    p.add_argument('--images', nargs='*')
    p.add_argument('--all', action='store_true')
    p.add_argument('--output-dir', required=True)
    args = p.parse_args()

    m, cfg = build(args.config, args.checkpoint)
    ds = build_dataset(cfg.data.test, dict(test_mode=True))
    idx_of = {ds.data_infos[i]['filename']: i for i in range(len(ds))}
    img_prefix = cfg.data.test.img_prefix
    tr = cfg.data.test.pipeline[1]['transforms'][2]
    mean = np.array(tr['mean_list'][0]); std = np.array(tr['std_list'][0])
    mean_t = np.array(tr['mean_list'][1]); std_t = np.array(tr['std_list'][1])
    mmcv.mkdir_or_exist(args.output_dir)

    def find(pp, fn):
        for pth in (osp.join(osp.dirname(img_prefix), pp[0], 'images',
                             osp.basename(img_prefix), fn),
                    osp.join(img_prefix, pp[1], fn)):
            if osp.exists(pth):
                return pth

    names = ([ds.data_infos[i]['filename'][:-4] for i in range(len(ds))]
             if args.all else args.images)
    dw0 = m.fuse_layer.idwt_layers[0]        # level 0 DWTC (LLR/gate 저장됨)
    for name in (mmcv.track_iter_progress(names) if args.all else names):
        fn = name + '.jpg'
        rgb = cv2.imread(find(('visible', 'visible'), fn))
        tp = find(('infrared', 'thermal'), fn)
        th = cv2.resize(cv2.imread(tp), (rgb.shape[1], rgb.shape[0])) if tp else rgb
        v = cv2.resize(rgb, (640, 512))[:, :, ::-1].astype(np.float32)
        t = cv2.resize(th, (640, 512))[:, :, ::-1].astype(np.float32)
        v = torch.from_numpy(((v - mean) / std).transpose(2, 0, 1)[None]).float().cuda()
        t = torch.from_numpy(((t - mean_t) / std_t).transpose(2, 0, 1)[None]).float().cuda()
        with torch.no_grad():
            m.extract_feat([v, t])
        llr = dw0._last_llr[0, 0].cpu().numpy()               # (512,640) [0,1]
        gate = dw0._last_gate[0].mean(0).cpu().numpy()        # (h,w) 3밴드 평균
        canvas = cv2.convertScaleAbs(rgb, alpha=1.0)
        panel = np.hstack([
            lab(rgb, name),
            lab(overlay(canvas, llr), 'LLR (thermal 선호=red)'),
            lab(overlay(canvas, gate), 'gate L0 (avg LH/HL/HH)'),
        ])
        cv2.imwrite(osp.join(args.output_dir, fn), panel,
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
        if not args.all:
            print(f'{name}: LLR mean={llr.mean():.3f} max={llr.max():.3f} | '
                  f'gate mean={gate.mean():.3f}')
    print('저장:', args.output_dir)


if __name__ == '__main__':
    main()
