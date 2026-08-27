"""조건 분석 — COXNet이 어떤 조건(밀집도·crowding·크기)에서 실패가 두드러지나?

이미 생성된 _ranking.txt(이미지별 name,n_gt,n_miss)의 recall과, 데이터셋 주석에서
계산한 기하(밀집도·crowding·크기)를 결합해, 조건별 GT-가중 recall을 낸다. 새 추론 없음.

  - density   = 이미지당 GT 수 (많을수록 밀집)
  - crowding  = 각 GT의 최근접 이웃거리 / 자기크기 의 중앙값 (작을수록 빽빽)
  - size      = sqrt(면적) 중앙값 (작을수록 tiny)

recall이 밀집↑·crowding빽빽·size작음에서 급락하면 → 그게 '두드러지는 실패 조건'.

usage:
  python tools/misc/analyze_conditions.py \
     --config configs/coxnet/coxnet_star_r50_fpn_1x_rgbtdroneperson.py \
     --ranking work_dir/diag_star_all/_ranking.txt
"""
import argparse
import os.path as osp

import numpy as np
from mmcv import Config
from mmdet.datasets import build_dataset


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--ranking', required=True, help='_ranking.txt (name n_gt n_miss bright)')
    args = p.parse_args()

    cfg = Config.fromfile(args.config)
    ds = build_dataset(cfg.data.test, dict(test_mode=True))
    idx_of = {ds.data_infos[i]['filename'][:-4]: i for i in range(len(ds))}

    # recall from ranking
    rec = {}
    with open(args.ranking) as f:
        next(f)
        for line in f:
            t = line.split()
            if len(t) < 3:
                continue
            name, n_gt, n_miss = t[0], int(t[1]), int(t[2])
            if n_gt > 0:
                rec[name] = (n_gt, n_miss)

    rows = []  # (name, n_gt, n_miss, density, crowding, size)
    for name, (n_gt, n_miss) in rec.items():
        if name not in idx_of:
            continue
        b = ds.get_ann_info(idx_of[name])['bboxes']
        if len(b) == 0:
            continue
        cx = (b[:, 0] + b[:, 2]) / 2
        cy = (b[:, 1] + b[:, 3]) / 2
        size = np.sqrt((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]))
        # 최근접 이웃 거리 / 자기 크기
        if len(b) > 1:
            d = np.sqrt((cx[:, None] - cx[None]) ** 2 + (cy[:, None] - cy[None]) ** 2)
            np.fill_diagonal(d, np.inf)
            nn = d.min(1)
            crowd = np.median(nn / np.maximum(size, 1e-6))
        else:
            crowd = np.inf
        rows.append((name, n_gt, n_miss, float(n_gt), float(crowd),
                     float(np.median(size))))

    arr = rows
    tot_gt = sum(r[1] for r in arr)
    tot_miss = sum(r[2] for r in arr)
    print(f'\n총 {len(arr)}장 | GT {tot_gt}, miss {tot_miss} '
          f'| 전체 recall = {1 - tot_miss / tot_gt:.3f}')

    def bucket(key_idx, label, reverse=False):
        # GT-가중 recall을 조건 사분위별로
        vals = np.array([r[key_idx] for r in arr if np.isfinite(r[key_idx])])
        q = np.quantile(vals, [0.25, 0.5, 0.75])
        edges = [-np.inf, q[0], q[1], q[2], np.inf]
        order = range(3, -1, -1) if reverse else range(4)
        tags = {0: f'Q1(낮음<{q[0]:.2f})', 1: 'Q2', 2: 'Q3',
                3: f'Q4(높음>{q[2]:.2f})'}
        print(f'\n=== recall by {label} ===')
        for k in order:
            sel = [r for r in arr if np.isfinite(r[key_idx])
                   and edges[k] < r[key_idx] <= edges[k + 1]]
            g = sum(r[1] for r in sel); m = sum(r[2] for r in sel)
            print(f'  {tags[k]:<16} 이미지 {len(sel):4d}  GT {g:5d}  '
                  f'recall={1 - m / max(g, 1):.3f}')

    bucket(3, 'density(이미지당 GT수) [많을수록 밀집]')
    bucket(4, 'crowding(최근접/크기) [작을수록 빽빽]')
    bucket(5, 'size(sqrt면적 중앙값) [작을수록 tiny]')


if __name__ == '__main__':
    main()
