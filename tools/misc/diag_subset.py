"""1단계 진단 — baseline COXNet이 저조도/thermal-clutter 조건에서 실제로 약한가?

각 val 이미지에 대해:
  - brightness = RGB Y(휘도) 평균  (낮음 = 저조도, 가설 A)
  - clutter    = thermal 평균/표준편차 (높음 = 넓게 뜨거움·복잡, 가설 B)
  - recall@0.5 = GT 중 IoU>=0.5로 검출된 비율
를 구해, 조건별(밝기·clutter 사분위)로 recall을 비교한다.

recall이 어두운/clutter 이미지에서 낮으면 → 가설 A/B가 데이터로 확정 → prior-router의
문제가 실재. 그리고 후보 리스트(dark+miss, hot+miss)를 저장해 2단계 CAM에 쓴다.

usage:
  python tools/misc/diag_subset.py \
     --config configs/coxnet/coxnet_r50_fpn_1x_rgbtdroneperson.py \
     --checkpoint work_dir/coxmamba/rgbtdroneperson/coxnet_r50_fpn_1x/epoch_12.pth \
     --output-dir work_dir/diag_subset
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


def iou(a, b):
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None); i = wh[..., 0] * wh[..., 1]
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return i / (aa[:, None] + ab[None, :] - i + 1e-9)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--score', type=float, default=0.3)
    p.add_argument('--output-dir', required=True)
    args = p.parse_args()

    cfg = Config.fromfile(args.config)
    cfg.model.backbone.init_cfg = None
    m = build_detector(cfg.model).cuda()
    load_checkpoint(m, args.checkpoint, map_location='cpu')
    m.eval()
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

    rows = []  # (name, bright, t_mean, t_std, n_gt, recall)
    names = [ds.data_infos[i]['filename'][:-4] for i in range(len(ds))]
    for name in mmcv.track_iter_progress(names):
        fn = name + '.jpg'
        rgb = cv2.imread(find(('visible', 'visible'), fn))
        tp = find(('infrared', 'thermal'), fn)
        th = cv2.resize(cv2.imread(tp), (rgb.shape[1], rgb.shape[0])) if tp else rgb
        # 밝기(Y) / thermal 통계
        yv = cv2.cvtColor(rgb[:, :, ::-1], cv2.COLOR_RGB2YUV)[:, :, 0]
        bright = float(yv.mean()) / 255.0
        tg = cv2.cvtColor(th, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        t_mean, t_std = float(tg.mean()), float(tg.std())
        # 추론
        v = cv2.resize(rgb, (640, 512))[:, :, ::-1].astype(np.float32)
        t = cv2.resize(th, (640, 512))[:, :, ::-1].astype(np.float32)
        v = torch.from_numpy(((v - mean) / std).transpose(2, 0, 1)[None]).float().cuda()
        t = torch.from_numpy(((t - mean_t) / std_t).transpose(2, 0, 1)[None]).float().cuda()
        metas = [dict(img_shape=(512, 640, 3), pad_shape=(512, 640, 3),
                      ori_shape=(rgb.shape[0], rgb.shape[1], 3),
                      scale_factor=np.array([640 / rgb.shape[1], 512 / rgb.shape[0]] * 2,
                                            dtype=np.float32),
                      batch_input_shape=(512, 640))]
        with torch.no_grad():
            r = m.simple_test([v, t], metas)[0]
        dets = np.vstack([c for c in r if len(c)]) if any(len(c) for c in r) else np.zeros((0, 5))
        kept = dets[dets[:, 4] >= args.score][:, :4] if len(dets) else np.zeros((0, 4))
        gt = ds.get_ann_info(idx_of[fn])['bboxes']
        if len(gt):
            matched = (iou(gt, kept).max(1) >= 0.5).sum() if len(kept) else 0
            recall = matched / len(gt)
        else:
            recall = np.nan
        rows.append((name, bright, t_mean, t_std, len(gt), recall))

    arr = [r for r in rows if not np.isnan(r[5])]
    br = np.array([r[1] for r in arr]); tm = np.array([r[2] for r in arr])
    rc = np.array([r[5] for r in arr])

    def bucket(metric, label):
        q = np.quantile(metric, [0.25, 0.5, 0.75])
        print(f'\n=== recall@0.5 by {label} 사분위 ===')
        edges = [-np.inf, q[0], q[1], q[2], np.inf]
        tags = [f'Q1(낮음 <{q[0]:.3f})', f'Q2', f'Q3', f'Q4(높음 >{q[2]:.3f})']
        for k in range(4):
            mask = (metric > edges[k]) & (metric <= edges[k + 1])
            print(f'  {tags[k]:<18} n={mask.sum():4d}  recall={rc[mask].mean():.3f}')

    print(f'\n총 {len(arr)}장 | 전체 recall@0.5 = {rc.mean():.3f}')
    bucket(br, 'RGB 밝기(Y)  [가설 A: 어두우면 recall↓ 기대]')
    bucket(tm, 'thermal 평균 [가설 B: 뜨거우면 recall↓ 기대]')

    # 후보 저장: 어두운데 recall 낮음 / thermal 높은데 recall 낮음
    arr.sort(key=lambda r: (r[1], r[5]))       # 어둡고 recall 낮은 순
    with open(osp.join(args.output_dir, '_candidates_lowlight.txt'), 'w') as f:
        f.write('name  bright  t_mean  n_gt  recall  (어둡고 recall낮은 순)\n')
        for r in arr[:40]:
            f.write(f'{r[0]}  {r[1]:.3f}  {r[2]:.3f}  {r[4]}  {r[5]:.3f}\n')
    arr.sort(key=lambda r: (-r[2], r[5]))      # thermal 높고 recall 낮은 순
    with open(osp.join(args.output_dir, '_candidates_clutter.txt'), 'w') as f:
        f.write('name  bright  t_mean  n_gt  recall  (thermal높고 recall낮은 순)\n')
        for r in arr[:40]:
            f.write(f'{r[0]}  {r[1]:.3f}  {r[2]:.3f}  {r[4]}  {r[5]:.3f}\n')
    print('\n후보 저장:', args.output_dir,
          '(_candidates_lowlight.txt, _candidates_clutter.txt)')


if __name__ == '__main__':
    main()
