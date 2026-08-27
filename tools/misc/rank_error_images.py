"""baseline 오류가 많은 val 이미지 랭킹 — 논문 정성비교(개선 전후) 후보 선정용

이미지별로 미탐 FN(IoU 0.1=논문 판정 / 0.5=엄격), near-miss(0.5에서만 미탐
= localization 개선이 보일 케이스), 오탐 FP, 밝기(저조도 행 후보 태그)를 계산해
FN 기준 상위를 출력한다. vis_missed.py와 동일한 매칭 규칙(ignore 중립 처리 포함).

usage:
  python tools/misc/rank_error_images.py --config <cfg> --pkl <results.pkl> [--top 15]
"""
import argparse
import os.path as osp

import cv2
import mmcv
import numpy as np
from mmcv import Config
from mmdet.datasets import build_dataset


def bbox_iou_mat(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--pkl', required=True)
    p.add_argument('--score-thr', type=float, default=0.3)
    p.add_argument('--top', type=int, default=15)
    args = p.parse_args()

    cfg = Config.fromfile(args.config)
    dataset = build_dataset(cfg.data.test, dict(test_mode=True))
    results = mmcv.load(args.pkl)
    assert len(results) == len(dataset)
    img_prefix = cfg.data.test.img_prefix

    def vis_path(fname):
        loader_style = osp.join(osp.dirname(img_prefix), 'visible',
                                'images', osp.basename(img_prefix), fname)
        legacy = osp.join(img_prefix, 'visible', fname)
        for path in (loader_style, legacy):
            if osp.exists(path):
                return path
        return None

    rows = []
    for idx in mmcv.track_iter_progress(list(range(len(dataset)))):
        fname = dataset.data_infos[idx]['filename']
        ann = dataset.get_ann_info(idx)
        gt_boxes, gt_labels = ann['bboxes'], ann['labels']
        gt_ignore = ann.get('bboxes_ignore', np.zeros((0, 4)))
        fn01 = fn05 = fp01 = 0
        for cls_id, cls_dets in enumerate(results[idx]):
            kept = cls_dets[cls_dets[:, 4] >= args.score_thr] \
                if len(cls_dets) else cls_dets
            gt_c = gt_boxes[gt_labels == cls_id]
            if len(gt_c):
                ious = bbox_iou_mat(gt_c, kept[:, :4] if len(kept)
                                    else np.zeros((0, 4)))
                mx = ious.max(axis=1) if ious.shape[1] else np.zeros(len(gt_c))
                fn01 += int((mx < 0.1).sum())
                fn05 += int((mx < 0.5).sum())
            if len(kept):
                if len(gt_c):
                    mx_d = bbox_iou_mat(kept[:, :4], gt_c).max(axis=1)
                else:
                    mx_d = np.zeros(len(kept))
                fp_cand = kept[mx_d < 0.1]
                if len(gt_ignore) and len(fp_cand):
                    lt = np.maximum(fp_cand[:, None, :2], gt_ignore[None, :, :2])
                    rb = np.minimum(fp_cand[:, None, 2:4], gt_ignore[None, :, 2:4])
                    wh = np.clip(rb - lt, 0, None)
                    iof = (wh[..., 0] * wh[..., 1]).max(axis=1) / np.maximum(
                        (fp_cand[:, 2] - fp_cand[:, 0]) *
                        (fp_cand[:, 3] - fp_cand[:, 1]), 1e-9)
                    fp_cand = fp_cand[iof < 0.5]
                fp01 += len(fp_cand)
        bright = -1.0
        path = vis_path(fname)
        if path:
            im = cv2.imread(path, cv2.IMREAD_REDUCED_GRAYSCALE_8)
            if im is not None:
                bright = float(im.mean())
        rows.append((fname, len(gt_boxes), fn01, fn05 - fn01, fp01, bright))

    hdr = f"{'filename':<40}{'nGT':>5}{'FN@.1':>7}{'near':>6}{'FP@.1':>7}{'bright':>8}"
    print('\n== 완전 미탐(FN@0.1) 많은 순 — 논문식(IoU 0.1) 그림에서 빨강이 많은 이미지 ==')
    print(hdr)
    for r in sorted(rows, key=lambda r: (-r[2], -r[4]))[:args.top]:
        print(f'{r[0]:<40}{r[1]:>5}{r[2]:>7}{r[3]:>6}{r[4]:>7}{r[5]:>8.1f}')
    print('\n== near-miss(0.5에서만 미탐) 많은 순 — localization 개선이 보일 이미지(IoU 0.5 그림용) ==')
    print(hdr)
    for r in sorted(rows, key=lambda r: (-r[3], -r[2]))[:args.top]:
        print(f'{r[0]:<40}{r[1]:>5}{r[2]:>7}{r[3]:>6}{r[4]:>7}{r[5]:>8.1f}')
    print('\n== 오탐(FP@0.1) 많은 순 ==')
    print(hdr)
    for r in sorted(rows, key=lambda r: (-r[4], -r[2]))[:args.top]:
        print(f'{r[0]:<40}{r[1]:>5}{r[2]:>7}{r[3]:>6}{r[4]:>7}{r[5]:>8.1f}')


if __name__ == '__main__':
    main()
