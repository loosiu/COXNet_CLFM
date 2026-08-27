"""검출(초록) + 놓친 GT(빨강) 시각화 — 논문 Fig. 스타일 재현

릴리즈 코드의 show_result에는 GT 비교 기능이 없어서(내부 버전에서 제거된 흔적만 있음)
저장된 결과 pkl과 GT를 IoU 매칭해 독립적으로 그린다.

usage:
  # 1) 먼저 검출 결과 pkl 저장 (1회만)
  python tools/test.py --config <cfg> --checkpoint <ckpt> --out work_dir/results.pkl
  # 2) 시각화 (GPU 불필요)
  python tools/misc/vis_missed.py --config <cfg> --pkl work_dir/results.pkl \
      --output-dir work_dir/vis_missed [--score-thr 0.3] [--iou-thr 0.5]

  # GT만 그리기 (pkl 불필요, 원본 val 이미지 위에 초록 박스)
  python tools/misc/vis_missed.py --config <cfg> --gt-only --output-dir work_dir/vis_gt_only
"""
import argparse
import os
import os.path as osp

import cv2
import mmcv
import numpy as np
from mmcv import Config
from mmdet.datasets import build_dataset

GREEN = (0, 255, 0)      # BGR: prediction 전체(오탐 포함) / GT(--gt-only 모드)
RED = (0, 0, 255)        # BGR: 놓친 GT(FN, 미탐)
# 논문 Fig.5/6 규약(초록=detected, 빨강=missed): 오탐은 별도 색 없이 초록에
# 포함. GT-only 그림과 나란히 보면 오탐은 GT에 없는 초록 박스로 식별 가능.
# ignore 영역(iscrowd GT)은 그리지 않음(개수는 통계로만 출력).


def bbox_iou_mat(a, b):
    """a: (N,4) xyxy, b: (M,4) xyxy -> (N,M) IoU"""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def draw(img, dets, missed_gt, thickness=2):
    """초록=prediction 전체(오탐 포함), 빨강=미탐(FN) — 논문 Fig.5/6 규약"""
    for x1, y1, x2, y2 in dets[:, :4].astype(int):
        cv2.rectangle(img, (x1, y1), (x2, y2), GREEN, thickness)
    for x1, y1, x2, y2 in missed_gt.astype(int):
        cv2.rectangle(img, (x1, y1), (x2, y2), RED, thickness)
    return img


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--pkl', default=None, help='검출 결과 pkl (--gt-only면 불필요)')
    p.add_argument('--gt-only', action='store_true',
                   help='GT만 초록 박스로 그리기 (pkl 불필요)')
    p.add_argument('--output-dir', required=True)
    p.add_argument('--score-thr', type=float, default=0.3)
    p.add_argument('--iou-thr', type=float, default=0.5)
    p.add_argument('--thickness', type=int, default=2)  # 저자 시각화(show_result)와 동일 굵기
    args = p.parse_args()

    if not args.gt_only and args.pkl is None:
        p.error("--pkl 또는 --gt-only 중 하나는 필요합니다")

    cfg = Config.fromfile(args.config)
    # test_mode=True: tools/test.py와 동일하게 GT 없는 이미지도 포함 (개수 일치)
    dataset = build_dataset(cfg.data.test, dict(test_mode=True))
    results = None
    if not args.gt_only:
        results = mmcv.load(args.pkl)
        assert len(results) == len(dataset), \
            f"pkl {len(results)}개 != dataset {len(dataset)}개 — config/pkl 짝이 맞는지 확인"

    img_prefix = cfg.data.test.img_prefix
    for sub in ['vis', 'thermal']:
        mmcv.mkdir_or_exist(osp.join(args.output_dir, sub))

    def find_img(spectral_pair, fname):
        """LoadYOLOImagePairFromFile과 동일한 경로 우선, 구 레이아웃 fallback."""
        loader_style = osp.join(osp.dirname(img_prefix), spectral_pair[0],
                                'images', osp.basename(img_prefix), fname)
        legacy = osp.join(img_prefix, spectral_pair[1], fname)
        for path in (loader_style, legacy):
            if osp.exists(path):
                return path
        return None

    n_gt_total, n_missed_total, n_fp_total, n_neutral_total, n_tp_total = \
        0, 0, 0, 0, 0
    for idx in mmcv.track_iter_progress(list(range(len(dataset)))):
        fname = dataset.data_infos[idx]['filename']
        ann = dataset.get_ann_info(idx)
        gt_boxes, gt_labels = ann['bboxes'], ann['labels']

        if args.gt_only:
            n_gt_total += len(gt_boxes)
            for sub, spectrals in [('vis', ('visible', 'visible')),
                                   ('thermal', ('infrared', 'thermal'))]:
                img_path = find_img(spectrals, fname)
                if img_path is None:
                    continue
                img = mmcv.imread(img_path)
                if sub == 'thermal':
                    img = mmcv.adjust_contrast(img, factor=1.5)
                for x1, y1, x2, y2 in gt_boxes.astype(int):
                    cv2.rectangle(img, (x1, y1), (x2, y2), GREEN, args.thickness)
                mmcv.imwrite(img, osp.join(args.output_dir, sub, fname))
            continue

        per_cls = results[idx]
        gt_ignore = ann.get('bboxes_ignore', np.zeros((0, 4)))
        missed, tp_all, fp_all, neutral_all = [], [], [], []
        for cls_id, cls_dets in enumerate(per_cls):
            kept = cls_dets[cls_dets[:, 4] >= args.score_thr] if len(cls_dets) else cls_dets
            gt_c = gt_boxes[gt_labels == cls_id]
            # GT 쪽: 같은 클래스 검출이 IoU 기준으로 덮으면 found, 아니면 missed(FN)
            if len(gt_c):
                ious = bbox_iou_mat(gt_c, kept[:, :4] if len(kept) else np.zeros((0, 4)))
                found = ious.max(axis=1) >= args.iou_thr if ious.shape[1] else np.zeros(len(gt_c), bool)
                missed.append(gt_c[~found])
            # 검출 쪽 분류(TP/FP/중립)는 통계 출력용 — 그림에서는 전부 초록.
            # ignore 영역(iscrowd GT)에 IoF>=0.5로 걸치는 검출은 COCO 평가와
            # 동일하게 중립으로 따로 센다.
            if len(kept):
                if len(gt_c):
                    ious_d = bbox_iou_mat(kept[:, :4], gt_c)
                    is_tp = ious_d.max(axis=1) >= args.iou_thr
                else:
                    is_tp = np.zeros(len(kept), bool)
                tp_all.append(kept[is_tp])
                fp_cand = kept[~is_tp]
                if len(gt_ignore) and len(fp_cand):
                    lt = np.maximum(fp_cand[:, None, :2], gt_ignore[None, :, :2])
                    rb = np.minimum(fp_cand[:, None, 2:4], gt_ignore[None, :, 2:4])
                    wh = np.clip(rb - lt, 0, None)
                    inter = wh[..., 0] * wh[..., 1]
                    area_d = ((fp_cand[:, 2] - fp_cand[:, 0]) *
                              (fp_cand[:, 3] - fp_cand[:, 1]))
                    iof = inter.max(axis=1) / np.maximum(area_d, 1e-9)
                    neutral = iof >= 0.5
                    neutral_all.append(fp_cand[neutral])
                    fp_all.append(fp_cand[~neutral])
                else:
                    fp_all.append(fp_cand)
        missed = np.vstack(missed) if missed else np.zeros((0, 4))
        tp_all = np.vstack(tp_all) if tp_all else np.zeros((0, 5))
        fp_all = np.vstack(fp_all) if fp_all else np.zeros((0, 5))
        neutral_all = np.vstack(neutral_all) if neutral_all else np.zeros((0, 5))
        n_gt_total += len(gt_boxes)
        n_missed_total += len(missed)
        n_fp_total += len(fp_all)
        n_neutral_total += len(neutral_all)
        n_tp_total += len(tp_all)

        for sub, spectrals in [('vis', ('visible', 'visible')),
                               ('thermal', ('infrared', 'thermal'))]:
            img_path = find_img(spectrals, fname)
            if img_path is None:
                continue
            img = mmcv.imread(img_path)
            if sub == 'thermal':
                img = mmcv.adjust_contrast(img, factor=1.5)  # 저자 시각화와 동일 보정
            img = draw(img, np.vstack([tp_all, fp_all, neutral_all]),
                       missed, args.thickness)
            mmcv.imwrite(img, osp.join(args.output_dir, sub, fname))

    if args.gt_only:
        print(f"\nGT {n_gt_total}개를 초록 박스로 저장 (원본 val 이미지, augmentation 없음)")
    else:
        print(f"\nGT {n_gt_total}개 중 미탐(FN, 빨강) {n_missed_total}개 "
              f"({n_missed_total / max(n_gt_total, 1) * 100:.1f}%) | "
              f"prediction(초록) {n_tp_total + n_fp_total + n_neutral_total}개 "
              f"(TP {n_tp_total}, 오탐 {n_fp_total}, ignore 위 {n_neutral_total}) | "
              f"score_thr={args.score_thr}, iou_thr={args.iou_thr}")
    print(f"저장: {args.output_dir}/vis/, {args.output_dir}/thermal/")


if __name__ == '__main__':
    main()
