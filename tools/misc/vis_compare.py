"""검출 비교 — baseline vs HF-gate를 같은 이미지에 돌려 나란히 (thermal 위).
초록=정답검출(TP, IoU>=0.5), 빨강=놓친 GT(FN). HF-gate가 빨강→초록이면 개선.

usage:
  python tools/misc/vis_compare.py \
     --base-config <cfg> --base-ckpt <pth> \
     --new-config <hfgate_cfg> --new-ckpt <pth> \
     --images 06007 06025 ... --output-dir work_dir/vis_compare
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

G, R, Y = (0, 255, 0), (0, 0, 255), (0, 255, 255)


def draw_gt(base_img, gt, th=2):
    """GT 박스만 초록으로."""
    img = base_img.copy()
    for k in range(len(gt)):
        x1, y1, x2, y2 = gt[k].astype(int)
        cv2.rectangle(img, (x1, y1), (x2, y2), G, th)
    return img


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
    cfg = Config.fromfile(cfg_path)
    cfg.model.backbone.init_cfg = None
    m = build_detector(cfg.model).cuda()
    load_checkpoint(m, ckpt, map_location='cpu')
    m.eval()
    return m, cfg


def draw(base_img, dets, gt, sc=0.3, iou_thr=0.5, th=2):
    img = base_img.copy()
    kept = dets[dets[:, 4] >= sc][:, :4] if len(dets) else np.zeros((0, 4))
    if len(gt):
        m = iou(gt, kept).max(1) if len(kept) else np.zeros(len(gt))
        for k in range(len(gt)):
            x1, y1, x2, y2 = gt[k].astype(int)
            cv2.rectangle(img, (x1, y1), (x2, y2), R if m[k] < iou_thr else G, th)
    n_miss = int((iou(gt, kept).max(1) < iou_thr).sum()) if len(gt) and len(kept) else len(gt)
    return img, n_miss


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base-config'); p.add_argument('--base-ckpt')
    p.add_argument('--new-config'); p.add_argument('--new-ckpt')
    p.add_argument('--images', nargs='*')
    p.add_argument('--all', action='store_true', help='val 전체')
    p.add_argument('--gt', action='store_true', help='GT 패널 추가(3분할)')
    p.add_argument('--new-name', default='GLRF')
    p.add_argument('--output-dir', required=True)
    args = p.parse_args()

    mb, cfg = build(args.base_config, args.base_ckpt)
    mn, _ = build(args.new_config, args.new_ckpt)
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

    def lab(img, t):
        bar = np.full((26, img.shape[1], 3), 40, np.uint8)
        cv2.putText(bar, t, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        return np.vstack([bar, img])

    names = ([ds.data_infos[i]['filename'][:-4] for i in range(len(ds))]
             if args.all else args.images)
    rank = []
    for name in (mmcv.track_iter_progress(names) if args.all else names):
        fn = name + '.jpg'
        rgb = cv2.imread(find(('visible', 'visible'), fn))
        tp = find(('infrared', 'thermal'), fn)
        th = cv2.resize(cv2.imread(tp), (rgb.shape[1], rgb.shape[0])) if tp else rgb
        canvas = cv2.convertScaleAbs(th, alpha=1.6)  # thermal 위에 그림(야간 가시)
        # GT (원본 해상도)
        ann = ds.get_ann_info(idx_of[fn])
        gt = ann['bboxes']
        # 전처리 쌍
        v = cv2.resize(rgb, (640, 512))[:, :, ::-1].astype(np.float32)
        t = cv2.resize(th, (640, 512))[:, :, ::-1].astype(np.float32)
        v = torch.from_numpy(((v - mean) / std).transpose(2, 0, 1)[None]).float().cuda()
        t = torch.from_numpy(((t - mean_t) / std_t).transpose(2, 0, 1)[None]).float().cuda()
        metas = [dict(img_shape=(512, 640, 3), pad_shape=(512, 640, 3),
                      ori_shape=(rgb.shape[0], rgb.shape[1], 3),
                      scale_factor=np.array([640 / rgb.shape[1], 512 / rgb.shape[0]] * 2,
                                            dtype=np.float32),
                      batch_input_shape=(512, 640))]
        outs = {}
        for tag, m in [('base', mb), ('new', mn)]:
            with torch.no_grad():
                r = m.simple_test([v, t], metas)[0]
            outs[tag] = np.vstack([c for c in r if len(c)]) if any(len(c) for c in r) else np.zeros((0, 5))
        ib, nb = draw(canvas, outs['base'], gt)
        inw, nn = draw(canvas, outs['new'], gt)
        panels = []
        if args.gt:
            panels.append(lab(draw_gt(canvas, gt), f'GT ({len(gt)})'))
        panels += [lab(ib, f'baseline (missed {nb})'),
                   lab(inw, f'{args.new_name} (missed {nn})')]
        panel = np.hstack(panels)
        cv2.imwrite(osp.join(args.output_dir, fn), panel, [cv2.IMWRITE_JPEG_QUALITY, 92])
        rank.append((name, len(gt), nb, nn, nb - nn))
        if not args.all:
            print(f'{name}: baseline missed {nb} | HF-gate missed {nn} '
                  f'(GT {len(gt)})  {"개선" if nn < nb else "동일/악화"}')
    # 개선(놓침 감소) 순위 저장
    rank.sort(key=lambda r: -r[4])
    with open(osp.join(args.output_dir, '_ranking.txt'), 'w') as f:
        f.write('name  GT  base_missed  hfgate_missed  improve(base-hf)\n')
        for r in rank:
            f.write(f'{r[0]}  {r[1]}  {r[2]}  {r[3]}  {r[4]:+d}\n')
    imp = sum(1 for r in rank if r[4] > 0)
    wor = sum(1 for r in rank if r[4] < 0)
    print(f'\n총 {len(rank)}장 | 개선 {imp} · 악화 {wor} · 동일 {len(rank)-imp-wor}')
    print('개선 top 12:')
    for r in rank[:12]:
        print(f'  {r[0]}: {r[2]}→{r[3]} 놓침 ({r[4]:+d}, GT {r[1]})')
    print('저장:', args.output_dir, '| 순위:', osp.join(args.output_dir, '_ranking.txt'))


if __name__ == '__main__':
    main()
