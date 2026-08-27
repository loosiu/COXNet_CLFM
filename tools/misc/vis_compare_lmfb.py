"""baseline ↔ LMFB tiny1 비교 — 어떤 이미지에서 tiny1(한 변 1~8px)을 더 잘 잡나.

두 모델을 각 val 이미지에 추론하고, tiny1 GT마다 각 모델이 잡았는지(IoU>=0.5) 판정.
2분할 패널(thermal 위):
  [baseline: tiny1 초록=잡음/빨강=놓침]  [LMFB: 초록/빨강, + 노랑=LMFB만 잡음(win), 자홍=LMFB만 놓침(lose)]
tiny1은 작아서 중심에 원을 그려 가시화.

_ranking.txt: (win - lose) 큰 순 = LMFB가 tiny1을 더 잘 잡은 이미지 순.

usage:
  python tools/misc/vis_compare_lmfb.py \
    --base-config configs/coxnet/coxnet_r50_fpn_1x_rgbtdroneperson.py \
    --base-ckpt   work_dir/coxmamba/rgbtdroneperson/coxnet_r50_fpn_1x/epoch_12.pth \
    --lmfb-config configs/coxnet/coxnet_lmfb_r50_fpn_1x_rgbtdroneperson.py \
    --lmfb-ckpt   work_dir/coxmamba/rgbtdroneperson/coxnet_lmfb_r50_fpn_1x/epoch_12.pth \
    --all --output-dir work_dir/cmp_lmfb_tiny1
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

G, R, Y, M = (0, 255, 0), (0, 0, 255), (0, 255, 255), (255, 0, 255)
TINY1_MAX = 8 ** 2   # area <= 64 → 한 변 <=8px


def iou(a, b):
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None); i = wh[..., 0] * wh[..., 1]
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return i / (aa[:, None] + ab[None, :] - i + 1e-9)


def build(cfg_path, ckpt):
    cfg = Config.fromfile(cfg_path); cfg.model.backbone.init_cfg = None
    m = build_detector(cfg.model).cuda()
    load_checkpoint(m, ckpt, map_location='cpu'); m.eval()
    return m, cfg


def lab(img, t, color=(255, 255, 255)):
    bar = np.full((28, img.shape[1], 3), 40, np.uint8)
    cv2.putText(bar, t, (5, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    return np.vstack([bar, img])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base-config', required=True); p.add_argument('--base-ckpt', required=True)
    p.add_argument('--lmfb-config', required=True); p.add_argument('--lmfb-ckpt', required=True)
    p.add_argument('--images', nargs='+'); p.add_argument('--all', action='store_true')
    p.add_argument('--score', type=float, default=0.3)
    p.add_argument('--iou', type=float, default=0.5)
    p.add_argument('--output-dir', required=True)
    args = p.parse_args()

    mb, cfg = build(args.base_config, args.base_ckpt)
    ml, _ = build(args.lmfb_config, args.lmfb_ckpt)
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

    def dets_of(m, v, t, metas):
        with torch.no_grad():
            r = m.simple_test([v, t], metas)[0]
        d = np.vstack([c for c in r if len(c)]) if any(len(c) for c in r) else np.zeros((0, 5))
        return d[d[:, 4] >= args.score][:, :4] if len(d) else np.zeros((0, 4))

    def draw(base_img, gt_t1, ok, extra=None):
        img = base_img.copy()
        for k, (x1, y1, x2, y2) in enumerate(gt_t1.astype(int)):
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            col = extra[k] if extra is not None else (G if ok[k] else R)
            th = 2 if (extra is not None and col in (Y, M)) else 1
            cv2.circle(img, (cx, cy), 7, col, th)
            cv2.rectangle(img, (x1, y1), (x2, y2), col, 1)
        return img

    names = ([ds.data_infos[i]['filename'][:-4] for i in range(len(ds))]
             if args.all else args.images)
    rank = []
    for name in (mmcv.track_iter_progress(names) if args.all else names):
        fn = name + '.jpg'
        rgb = cv2.imread(find(('visible', 'visible'), fn))
        tp = find(('infrared', 'thermal'), fn)
        th_img = cv2.resize(cv2.imread(tp), (rgb.shape[1], rgb.shape[0])) if tp else rgb
        gt = ds.get_ann_info(idx_of[fn])['bboxes']
        if not len(gt):
            continue
        area = (gt[:, 2] - gt[:, 0]) * (gt[:, 3] - gt[:, 1])
        t1 = gt[(area >= 1) & (area <= TINY1_MAX)]
        if not len(t1):
            continue
        v = cv2.resize(rgb, (640, 512))[:, :, ::-1].astype(np.float32)
        t = cv2.resize(th_img, (640, 512))[:, :, ::-1].astype(np.float32)
        v = torch.from_numpy(((v - mean) / std).transpose(2, 0, 1)[None]).float().cuda()
        t = torch.from_numpy(((t - mean_t) / std_t).transpose(2, 0, 1)[None]).float().cuda()
        metas = [dict(img_shape=(512, 640, 3), pad_shape=(512, 640, 3),
                      ori_shape=(rgb.shape[0], rgb.shape[1], 3),
                      scale_factor=np.array([640 / rgb.shape[1], 512 / rgb.shape[0]] * 2,
                                            dtype=np.float32), batch_input_shape=(512, 640))]
        db = dets_of(mb, v, t, metas)
        dl = dets_of(ml, v, t, metas)
        base_ok = (iou(t1, db).max(1) >= args.iou) if len(db) else np.zeros(len(t1), bool)
        lmfb_ok = (iou(t1, dl).max(1) >= args.iou) if len(dl) else np.zeros(len(t1), bool)
        win = lmfb_ok & ~base_ok
        lose = base_ok & ~lmfb_ok
        nwin, nlose = int(win.sum()), int(lose.sum())

        col_l = [Y if win[k] else (M if lose[k] else (G if lmfb_ok[k] else R))
                 for k in range(len(t1))]
        left = draw(th_img, t1, base_ok)
        right = draw(th_img, t1, lmfb_ok, extra=col_l)
        panel = np.hstack([
            lab(left, f'baseline  tiny1 {int(base_ok.sum())}/{len(t1)}'),
            lab(right, f'LMFB  tiny1 {int(lmfb_ok.sum())}/{len(t1)}  '
                       f'(win={nwin} lose={nlose})',
                color=Y if nwin > nlose else (255, 255, 255))])
        cv2.imwrite(osp.join(args.output_dir, fn), panel, [cv2.IMWRITE_JPEG_QUALITY, 92])
        rank.append((name, len(t1), int(base_ok.sum()), int(lmfb_ok.sum()), nwin, nlose))
        if not args.all:
            print(f'{name}: tiny1 base {int(base_ok.sum())} vs LMFB {int(lmfb_ok.sum())} '
                  f'(win {nwin}, lose {nlose})')

    rank.sort(key=lambda r: (-(r[4] - r[5]), -r[4]))
    with open(osp.join(args.output_dir, '_ranking.txt'), 'w') as f:
        f.write('name  n_tiny1  base_ok  lmfb_ok  win  lose   (win-lose 큰 순)\n')
        for r in rank:
            f.write(f'{r[0]}  {r[1]}  {r[2]}  {r[3]}  {r[4]}  {r[5]}\n')
    tt = sum(r[1] for r in rank); bb = sum(r[2] for r in rank); ll = sum(r[3] for r in rank)
    tw = sum(r[4] for r in rank); tl = sum(r[5] for r in rank)
    print(f'\ntiny1 GT {tt} | baseline 잡음 {bb} ({bb/max(tt,1):.3f}) | '
          f'LMFB 잡음 {ll} ({ll/max(tt,1):.3f})')
    print(f'LMFB만 잡음(win) {tw} | LMFB만 놓침(lose) {tl} | 순이득 {tw - tl}')
    print('저장:', args.output_dir, '| win 순위: _ranking.txt')


if __name__ == '__main__':
    main()
