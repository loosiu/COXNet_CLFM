"""HF gate 정성 시각화 — 학습된 모델에서 M_L(LPM)과 gate g를 추출.
저조도(M_L↓) 영역에서 gate g가 높아져 thermal edge를 선택하는지 눈으로 확인.

usage:
  python tools/misc/vis_gate.py --config <hfgate_cfg> --checkpoint <pth> \
      --images 06007 06064 05661 --output-dir work_dir/vis_gate
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


def heat(m):
    m = (m - m.min()) / (m.max() - m.min() + 1e-9)
    return cv2.applyColorMap(np.uint8(255 * m), cv2.COLORMAP_JET)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--images', nargs='+', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--level', type=int, default=0, help='시각화할 FPN 레벨')
    args = p.parse_args()

    cfg = Config.fromfile(args.config)
    cfg.model.backbone.init_cfg = None
    model = build_detector(cfg.model).cuda()
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    model.eval()
    fl = model.fuse_layer
    img_prefix = cfg.data.test.img_prefix
    tr = cfg.data.test.pipeline[1]['transforms'][2]
    mean = np.array(tr['mean_list'][0]); std = np.array(tr['std_list'][0])
    mean_t = np.array(tr['mean_list'][1]); std_t = np.array(tr['std_list'][1])
    mmcv.mkdir_or_exist(args.output_dir)

    # 훅: IPM 출력(M_L=weight_rgb) + gate_conv 출력(gate logit) 캡처
    cap = {'ml': [], 'g': []}
    L = args.level
    fl.crw_ipm[L].register_forward_hook(
        lambda m, i, o: cap['ml'].append(o[1].detach()))
    fl.idwt_layers[L].gate_conv.register_forward_hook(
        lambda m, i, o: cap['g'].append(torch.sigmoid(o).detach()))

    def find(prefix_pair, fname):
        loader = osp.join(osp.dirname(img_prefix), prefix_pair[0], 'images',
                          osp.basename(img_prefix), fname)
        legacy = osp.join(img_prefix, prefix_pair[1], fname)
        for pth in (loader, legacy):
            if osp.exists(pth):
                return pth
        return None

    def lab(img, t):
        bar = np.full((26, img.shape[1], 3), 40, np.uint8)
        cv2.putText(bar, t, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1)
        return np.vstack([bar, img])

    for name in args.images:
        fn = name + '.jpg'
        rgb = cv2.imread(find(('visible', 'visible'), fn))
        tp = find(('infrared', 'thermal'), fn)
        th = cv2.imread(tp) if tp else np.zeros_like(rgb)
        H, W = rgb.shape[:2]; th = cv2.resize(th, (W, H))
        v = cv2.resize(rgb, (640, 512))[:, :, ::-1].astype(np.float32)
        t = cv2.resize(th, (640, 512))[:, :, ::-1].astype(np.float32)
        v = torch.from_numpy(((v - mean) / std).transpose(2, 0, 1)[None]).float().cuda()
        t = torch.from_numpy(((t - mean_t) / std_t).transpose(2, 0, 1)[None]).float().cuda()
        metas = [dict(img_shape=(512, 640, 3), pad_shape=(512, 640, 3),
                      ori_shape=(512, 640, 3), scale_factor=1.0,
                      batch_input_shape=(512, 640))]
        cap['ml'].clear(); cap['g'].clear()
        with torch.no_grad():
            model.simple_test([v, t], metas)
        ml = cap['ml'][0][0, 0].cpu().numpy()
        # gate_conv은 밴드당 1회(3회) → 평균
        g = torch.stack(cap['g'][:3]).mean(0)[0, 0].cpu().numpy()
        ml_r = cv2.resize(ml, (W, H)); g_r = cv2.resize(g, (W, H))
        panel = np.hstack([
            lab(rgb, 'RGB'),
            lab(heat(ml_r), f'M_L (RGB취약, L{L}) mean={ml.mean():.2f}'),
            lab(heat(g_r), f'gate g (thermal선택) mean={g.mean():.2f}'),
            lab(cv2.convertScaleAbs(th, alpha=1.5), 'thermal')])
        cv2.imwrite(osp.join(args.output_dir, fn), panel,
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
        print(f'{name}: M_L={ml.mean():.3f}, gate_g={g.mean():.3f} '
              f'(어두울수록 M_L↓·g↑ 기대)')
    print('저장:', args.output_dir)


if __name__ == '__main__':
    main()
