"""RGB-thermal 미정렬을 객체별 확대로 보기 (모델 불필요, 이미지+GT만).

GT는 thermal에 annotation됨(thermal 객체 위치=초록박스). 같은 픽셀좌표의 RGB에서
객체가 얼마나 어긋나 있는지(parallax)를 객체별 확대로 확인.
overlay: thermal=빨강, RGB=초록 → 정렬되면 노랑, 어긋나면 빨강/초록 유령 분리.

객체당 한 행: [thermal 확대+GT | RGB 확대+GT | overlay(빨강=thermal 초록=RGB)]

usage:
  python tools/misc/vis_misalign.py --config ... \
     --images 05540 05661 --output-dir work_dir/misalign [--all] [--max-obj 6]
"""
import argparse
import os.path as osp

import cv2
import mmcv
import numpy as np
from mmcv import Config
from mmdet.datasets import build_dataset

Z = 200


def lab(img, t):
    bar = np.full((24, img.shape[1], 3), 40, np.uint8)
    cv2.putText(bar, t, (5, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return np.vstack([bar, img])


def crop(gray, cx, cy, half, box, col_box=(0, 255, 0)):
    H, W = gray.shape[:2]
    x0, y0 = max(0, cx - half), max(0, cy - half)
    x1, y1 = min(W, cx + half), min(H, cy + half)
    c = gray[y0:y1, x0:x1]
    if c.size == 0:
        c = np.zeros((8, 8), np.uint8)
    s = Z / max(c.shape[0], c.shape[1])
    c = cv2.resize(c, (int(c.shape[1] * s), int(c.shape[0] * s)), interpolation=cv2.INTER_NEAREST)
    c = cv2.copyMakeBorder(c, 0, max(0, Z - c.shape[0]), 0, max(0, Z - c.shape[1]),
                           cv2.BORDER_CONSTANT, value=30)[:Z, :Z]
    return c, (x0, y0, s)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--images', nargs='+'); p.add_argument('--all', action='store_true')
    p.add_argument('--max-obj', type=int, default=6)
    p.add_argument('--output-dir', required=True)
    args = p.parse_args()

    cfg = Config.fromfile(args.config)
    ds = build_dataset(cfg.data.test, dict(test_mode=True))
    idx_of = {ds.data_infos[i]['filename']: i for i in range(len(ds))}
    img_prefix = cfg.data.test.img_prefix
    mmcv.mkdir_or_exist(args.output_dir)

    def find(pp, fn):
        for pth in (osp.join(osp.dirname(img_prefix), pp[0], 'images',
                             osp.basename(img_prefix), fn), osp.join(img_prefix, pp[1], fn)):
            if osp.exists(pth):
                return pth

    def draw_box(c, box, off, col):
        x0, y0, s = off
        cv2.rectangle(c, (int((box[0] - x0) * s), int((box[1] - y0) * s)),
                      (int((box[2] - x0) * s), int((box[3] - y0) * s)), col, 1)

    names = ([ds.data_infos[i]['filename'][:-4] for i in range(len(ds))]
             if args.all else args.images)
    for name in (mmcv.track_iter_progress(names) if args.all else names):
        fn = name + '.jpg'
        rgb = cv2.imread(find(('visible', 'visible'), fn))
        tp = find(('infrared', 'thermal'), fn)
        th = cv2.resize(cv2.imread(tp), (rgb.shape[1], rgb.shape[0])) if tp else rgb
        gt = ds.get_ann_info(idx_of[fn])['bboxes']
        if not len(gt):
            continue
        tg = cv2.cvtColor(cv2.convertScaleAbs(th, alpha=1.3), cv2.COLOR_BGR2GRAY)
        vg = cv2.cvtColor(cv2.convertScaleAbs(rgb, alpha=2.0), cv2.COLOR_BGR2GRAY)

        rows = []
        for k, (x1, y1, x2, y2) in enumerate(gt[:args.max_obj]):
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            half = int(max(24, (x2 - x1) * 3, (y2 - y1) * 3))
            tc, off = crop(tg, cx, cy, half, (x1, y1, x2, y2))
            vc, _ = crop(vg, cx, cy, half, (x1, y1, x2, y2))
            tc3 = cv2.cvtColor(tc, cv2.COLOR_GRAY2BGR); draw_box(tc3, (x1, y1, x2, y2), off, (0, 255, 0))
            vc3 = cv2.cvtColor(vc, cv2.COLOR_GRAY2BGR); draw_box(vc3, (x1, y1, x2, y2), off, (0, 255, 0))
            ov = np.zeros((Z, Z, 3), np.uint8)
            ov[..., 2] = tc      # R = thermal
            ov[..., 1] = vc      # G = RGB(visible)
            draw_box(ov, (x1, y1, x2, y2), off, (255, 255, 255))
            row = np.hstack([lab(tc3, f'obj{k+1} thermal(+GT)'),
                             lab(vc3, 'RGB(+GT, same coord)'),
                             lab(ov, 'overlay R=thermal G=RGB')])
            rows.append(row)
        cv2.imwrite(osp.join(args.output_dir, fn), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 92])
        if not args.all:
            print(f'{name}: {min(len(gt), args.max_obj)} objects')
    print('저장:', args.output_dir)


if __name__ == '__main__':
    main()
