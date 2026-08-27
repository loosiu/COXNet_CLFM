"""Ground-truth 미정렬 시각화 (논문 Fig 4식) — visible GT vs thermal GT.

RGBTDronePerson은 visible/thermal을 각각 annotation(sub_train_visible/thermal.json).
같은 객체의 visible GT(초록)와 thermal GT(빨강) 박스 차이 = 진짜 RGB-thermal 미정렬.

패널: [visible + vGT(초록) | thermal + tGT(초록) | visible overlay: vGT초록+tGT빨강+이동선]

usage:
  python tools/misc/vis_gt_misalign.py --root data/RGBTDronePerson \
     --output-dir work_dir/gt_misalign [--all] [--min-off 8]
"""
import argparse
import json
import os.path as osp

import cv2
import mmcv
import numpy as np


def lab(img, t):
    bar = np.full((26, img.shape[1], 3), 40, np.uint8)
    cv2.putText(bar, t, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return np.vstack([bar, img])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', default='data/RGBTDronePerson')
    p.add_argument('--all', action='store_true')
    p.add_argument('--min-off', type=float, default=0, help='평균 offset 이 이상만 저장')
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--output-dir', required=True)
    args = p.parse_args()

    v = json.load(open(osp.join(args.root, 'sub_train_visible.json')))
    t = json.load(open(osp.join(args.root, 'sub_train_thermal.json')))
    vimg = {i['id']: i['file_name'] for i in v['images']}
    timg = {i['id']: i['file_name'] for i in t['images']}

    def by_file(js, img):
        d = {}
        for a in js['annotations']:
            d.setdefault(img[a['image_id']], []).append(a['bbox'])
        return d
    vb, tb = by_file(v, vimg), by_file(t, timg)
    common = sorted(set(vb) & set(tb))
    mmcv.mkdir_or_exist(args.output_dir)

    def box(b):
        return (int(b[0]), int(b[1]), int(b[0] + b[2]), int(b[1] + b[3]))

    def cen(b):
        return (int(b[0] + b[2] / 2), int(b[1] + b[3] / 2))

    saved = 0
    for fn in (mmcv.track_iter_progress(common) if args.all else common):
        V, T = vb[fn], tb[fn]
        rgb = cv2.imread(osp.join(args.root, 'train', 'visible', fn))
        th = cv2.imread(osp.join(args.root, 'train', 'thermal', fn))
        if rgb is None or th is None:
            continue
        th = cv2.resize(th, (rgb.shape[1], rgb.shape[0]))
        offs = []
        vpan = cv2.convertScaleAbs(rgb, alpha=1.3).copy()
        tpan = cv2.convertScaleAbs(th, alpha=1.3).copy()
        ov = cv2.convertScaleAbs(rgb, alpha=1.3).copy()
        for bv in V:
            x1, y1, x2, y2 = box(bv)
            cv2.rectangle(vpan, (x1, y1), (x2, y2), (0, 255, 0), 1)
            cv2.rectangle(ov, (x1, y1), (x2, y2), (0, 255, 0), 1)   # visible GT 초록
        for bt in T:
            x1, y1, x2, y2 = box(bt)
            cv2.rectangle(tpan, (x1, y1), (x2, y2), (0, 255, 0), 1)
            cv2.rectangle(ov, (x1, y1), (x2, y2), (0, 0, 255), 1)   # thermal GT 빨강
        if len(V) == len(T):
            for bv, bt in zip(V, T):
                cvp, ctp = cen(bv), cen(bt)
                cv2.line(ov, cvp, ctp, (255, 255, 0), 1)
                offs.append(((cvp[0] - ctp[0]) ** 2 + (cvp[1] - ctp[1]) ** 2) ** 0.5)
        mo = float(np.mean(offs)) if offs else 0
        if mo < args.min_off:
            continue
        panel = np.hstack([lab(vpan, f'{fn[:-4]} visible + Visible-GT(green)'),
                           lab(tpan, 'thermal + Thermal-GT(green)'),
                           lab(ov, f'overlay  green=Vis-GT red=Ther-GT  (mean offset {mo:.1f}px)')])
        cv2.imwrite(osp.join(args.output_dir, fn), panel, [cv2.IMWRITE_JPEG_QUALITY, 92])
        saved += 1
        if not args.all:
            print(f'{fn[:-4]}: {len(V)}v/{len(T)}t obj, mean offset {mo:.1f}px')
        if args.limit and saved >= args.limit:
            break
    print(f'저장: {args.output_dir} ({saved}장)')


if __name__ == '__main__':
    main()
