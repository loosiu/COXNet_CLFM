"""COXNet 정성 진단 — AAM(적응 정렬) 오프셋이 실제로 정렬을 하는가?

HOFM의 각 레벨 conv_offset이 예측하는 오프셋을 후킹해, 실제 학습된 shift 크기를
측정한다. 설계상 한계: shift ∈ ±0.5 feature-pixel × om_range_factor (line 388).

  - shift(feature-px) 평균/최대 : AAM이 이 캡을 얼마나 쓰는가
  - shift(input-px)   = shift(feature-px) × stride
  - |shift|/cap(%)    : 100%면 포화, ~0%면 사실상 정렬 안 함(= Mamba γ처럼 inert)

만약 평균 shift가 캡의 몇 % 수준이고 input-px로 1px 미만이면 →
"Adaptive Alignment"이 정성적으로 거의 항등(identity)에 가깝다는 증거.

usage:
  python tools/misc/probe_aam_offset.py \
     --config configs/coxnet/coxnet_r50_fpn_1x_rgbtdroneperson.py \
     --checkpoint work_dir/coxmamba/rgbtdroneperson/coxnet_r50_fpn_1x/epoch_12.pth \
     --output-dir work_dir/probe_aam [--vis 06064 05540] [--limit 300]
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

STRIDES = [8, 16, 32, 64]   # P3~P6 (HOFM 레벨 순)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--limit', type=int, default=0, help='0=전체')
    p.add_argument('--vis', nargs='+', default=[], help='offset field 그릴 이미지')
    args = p.parse_args()

    cfg = Config.fromfile(args.config); cfg.model.backbone.init_cfg = None
    m = build_detector(cfg.model).cuda()
    load_checkpoint(m, args.checkpoint, map_location='cpu'); m.eval()
    ds = build_dataset(cfg.data.test, dict(test_mode=True))
    idx_of = {ds.data_infos[i]['filename']: i for i in range(len(ds))}
    img_prefix = cfg.data.test.img_prefix
    tr = cfg.data.test.pipeline[1]['transforms'][2]
    mean = np.array(tr['mean_list'][0]); std = np.array(tr['std_list'][0])
    mean_t = np.array(tr['mean_list'][1]); std_t = np.array(tr['std_list'][1])
    mmcv.mkdir_or_exist(args.output_dir)

    layers = m.fuse_layer.hofm_layers
    nL = len(layers)
    omf = [float(getattr(layers[i], 'om_range_factor', 1.0)) for i in range(nL)]

    # conv_offset 출력(raw) 후킹 → 스케일링해서 shift(feature-px) 저장
    cap_px = [0.5 * omf[i] for i in range(nL)]     # 레벨별 최대 |shift| 한 축 (feature-px)
    store = {}

    def mk_hook(i):
        def hook(mod, inp, out):
            store[i] = out.detach()     # (B,2,Hk,Wk) raw
        return hook
    handles = [layers[i].conv_offset.register_forward_hook(mk_hook(i))
               for i in range(nL)]

    def load_pair(fn):
        rgb = cv2.imread(find(('visible', 'visible'), fn))
        tp = find(('infrared', 'thermal'), fn)
        th = cv2.resize(cv2.imread(tp), (rgb.shape[1], rgb.shape[0])) if tp else rgb
        v = cv2.resize(rgb, (640, 512))[:, :, ::-1].astype(np.float32)
        t = cv2.resize(th, (640, 512))[:, :, ::-1].astype(np.float32)
        v = torch.from_numpy(((v - mean) / std).transpose(2, 0, 1)[None]).float().cuda()
        t = torch.from_numpy(((t - mean_t) / std_t).transpose(2, 0, 1)[None]).float().cuda()
        return rgb, th, v, t

    def find(pp, fn):
        for pth in (osp.join(osp.dirname(img_prefix), pp[0], 'images',
                             osp.basename(img_prefix), fn),
                    osp.join(img_prefix, pp[1], fn)):
            if osp.exists(pth):
                return pth

    def shift_px(i):
        """store[i] raw → per-axis shift(feature-px): t.tanh()*omf*0.5"""
        raw = store[i]
        t = raw.tanh()                       # (B,2,Hk,Wk) in [-1,1]
        sy = t[:, 0] * omf[i] * 0.5          # feature-px (H축)
        sx = t[:, 1] * omf[i] * 0.5
        return sy[0].cpu().numpy(), sx[0].cpu().numpy()

    # ---- 전체(또는 limit) 통계 ----
    names = [ds.data_infos[i]['filename'][:-4] for i in range(len(ds))]
    if args.limit:
        names = names[:args.limit]
    acc = [dict(sum=0.0, sq=0.0, mx=0.0, n=0) for _ in range(nL)]
    for name in mmcv.track_iter_progress(names):
        fn = name + '.jpg'
        _, _, v, t = load_pair(fn)
        with torch.no_grad():
            m.extract_feat([v, t])
        for i in range(nL):
            sy, sx = shift_px(i)
            mag = np.sqrt(sy ** 2 + sx ** 2)     # feature-px
            acc[i]['sum'] += mag.sum(); acc[i]['sq'] += (mag ** 2).sum()
            acc[i]['mx'] = max(acc[i]['mx'], float(mag.max()))
            acc[i]['n'] += mag.size

    print('\n=== AAM 오프셋 실측 (평균 |shift|) ===')
    print(f'{"level":<6}{"stride":>7}{"cap(fpx)":>10}{"mean(fpx)":>11}'
          f'{"mean(inpx)":>12}{"max(fpx)":>10}{"cap사용%":>9}')
    lines = []
    for i in range(nL):
        n = acc[i]['n']; mean_mag = acc[i]['sum'] / n
        cap_mag = cap_px[i] * np.sqrt(2)                 # 두 축 동시 최대
        row = (f'P{i+3:<5}{STRIDES[i]:>7}{cap_mag:>10.3f}{mean_mag:>11.4f}'
               f'{mean_mag*STRIDES[i]:>12.3f}{acc[i]["mx"]:>10.4f}'
               f'{100*mean_mag/cap_mag:>8.1f}%')
        print(row); lines.append(row)
    with open(osp.join(args.output_dir, '_aam_offset_stats.txt'), 'w') as f:
        f.write('level stride cap(fpx) mean(fpx) mean(inpx) max(fpx) cap사용%\n')
        f.write('\n'.join(lines) + '\n')

    # ---- offset field 시각화 (P3) ----
    for name in args.vis:
        fn = name + '.jpg'
        rgb, th, v, t = load_pair(fn)
        with torch.no_grad():
            m.extract_feat([v, t])
        sy, sx = shift_px(0)                  # P3
        Hk, Wk = sy.shape
        base = cv2.resize(cv2.convertScaleAbs(th, alpha=1.4), (Wk * 8, Hk * 8))
        vis = base.copy()
        step = 1
        AMP = 40                              # 화살표 증폭(가시성)
        for yy in range(0, Hk, step):
            for xx in range(0, Wk, step):
                cy, cx = int((yy + 0.5) * 8), int((xx + 0.5) * 8)
                dy, dx = sy[yy, xx] * 8 * AMP, sx[yy, xx] * 8 * AMP
                cv2.arrowedLine(vis, (cx, cy), (int(cx + dx), int(cy + dy)),
                                (0, 255, 255), 1, tipLength=0.3)
        mag = np.sqrt(sy ** 2 + sx ** 2)
        cv2.putText(vis, f'{name} P3 offset x{AMP} | mean|shift|={mag.mean():.4f}fpx '
                    f'({mag.mean()*8:.3f}inpx) max={mag.max():.3f}fpx',
                    (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        cv2.imwrite(osp.join(args.output_dir, f'offset_{fn}'), vis)
        print(f'offset field 저장: offset_{fn}  (mean|shift|={mag.mean():.4f} fpx)')

    for h in handles:
        h.remove()
    print('\n저장:', args.output_dir)


if __name__ == '__main__':
    main()
