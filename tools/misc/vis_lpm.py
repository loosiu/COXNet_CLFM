"""LPM(=CCRDet IPM) 신뢰지도 시각화 — RGB가 필요한/믿을만한 영역이 구분되는지 정성 확인.

두 가지를 그린다:
  1) 밝기 지도(brightness prior) — LPM의 입력, 결정론적(학습 불필요). RGB 밝은=신뢰,
     어두운=비신뢰(thermal 필요). compute_brightness_map과 동일 로직.
  2) --checkpoint 주면 학습된 w_rgb(RGB 신뢰 지도) — backbone+neck+IPM만 돌려 추출
     (CLFM/DWT는 안 거치므로 CPU 가능). 체크포인트 없으면 IPM이 랜덤이라 생략.

패널: [RGB | 밝기지도 | RGB위 오버레이 | thermal]  (+ ckpt면 레벨별 w_rgb 추가)

usage:
  # 밝기 지도만 (지금, 학습 불필요)
  python tools/misc/vis_lpm.py --config configs/coxnet/coxnet_lpm_r50_fpn_1x_rgbtdroneperson.py \
      --images 01698 01076 04146 05099 05661 --output-dir work_dir/vis_lpm
  # 학습 후 실제 w_rgb까지
  python tools/misc/vis_lpm.py --config <lpm_cfg> --checkpoint <lpm.pth> --images ... --output-dir ...
"""
import argparse
import os.path as osp

import cv2
import numpy as np
import torch
from mmcv import Config


def heatmap(m):
    """HxW float -> BGR JET heatmap (per-map min-max 정규화)."""
    m = (m - m.min()) / (m.max() - m.min() + 1e-9)
    return cv2.applyColorMap(np.uint8(255 * m), cv2.COLORMAP_JET)


def brightness_prior(bgr):
    """compute_brightness_map과 동일: YUV-Y, min-max [0,1]. (raw 이미지용)"""
    y = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV)[:, :, 0]
    y = cv2.normalize(y, None, 0, 255, cv2.NORM_MINMAX)
    return y / 255.0


def find_img(img_prefix, spectral_pair, fname):
    loader = osp.join(osp.dirname(img_prefix), spectral_pair[0], 'images',
                      osp.basename(img_prefix), fname)
    legacy = osp.join(img_prefix, spectral_pair[1], fname)
    for p in (loader, legacy):
        if osp.exists(p):
            return p
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--checkpoint', default=None,
                   help='주면 학습된 w_rgb까지 추출(backbone+neck+IPM)')
    p.add_argument('--images', nargs='+', required=True, help='확장자 없는 파일명')
    p.add_argument('--output-dir', required=True)
    p.add_argument('--label-bar', type=int, default=26)
    args = p.parse_args()

    cfg = Config.fromfile(args.config)
    import mmcv
    mmcv.mkdir_or_exist(args.output_dir)
    img_prefix = cfg.data.test.img_prefix

    model = None
    if args.checkpoint:
        from mmdet.models import build_detector
        from mmcv.runner import load_checkpoint
        cfg.model.backbone.init_cfg = None
        model = build_detector(cfg.model)
        load_checkpoint(model, args.checkpoint, map_location='cpu')
        model.eval()
        mean = np.array(cfg.data.test.pipeline[1]['transforms'][2]['mean_list'][0])
        std = np.array(cfg.data.test.pipeline[1]['transforms'][2]['std_list'][0])
        mean_t = np.array(cfg.data.test.pipeline[1]['transforms'][2]['mean_list'][1])
        std_t = np.array(cfg.data.test.pipeline[1]['transforms'][2]['std_list'][1])

    def labeled(img, text):
        bar = np.full((args.label_bar, img.shape[1], 3), 40, np.uint8)
        cv2.putText(bar, text, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1)
        return np.vstack([bar, img])

    for name in args.images:
        fname = name + '.jpg'
        rgb_p = find_img(img_prefix, ('visible', 'visible'), fname)
        th_p = find_img(img_prefix, ('infrared', 'thermal'), fname)
        if rgb_p is None:
            print(f'skip {name}: RGB 경로 없음')
            continue
        rgb = cv2.imread(rgb_p)
        th = cv2.imread(th_p) if th_p else np.zeros_like(rgb)
        H, W = rgb.shape[:2]
        th = cv2.resize(th, (W, H))

        bp = brightness_prior(rgb)
        bp_h = heatmap(bp)
        overlay = cv2.addWeighted(rgb, 0.5, bp_h, 0.5, 0)
        panel = [labeled(rgb, 'RGB'),
                 labeled(bp_h, f'brightness prior (mean={bp.mean():.2f})'),
                 labeled(overlay, 'overlay'),
                 labeled(cv2.convertScaleAbs(th, alpha=1.5), 'thermal')]

        if model is not None:
            v = cv2.resize(rgb, (640, 512))[:, :, ::-1].astype(np.float32)
            t = cv2.resize(th, (640, 512))[:, :, ::-1].astype(np.float32)
            v = ((v - mean) / std).transpose(2, 0, 1)[None]
            t = ((t - mean_t) / std_t).transpose(2, 0, 1)[None]
            v = torch.from_numpy(v).float()
            t = torch.from_numpy(t).float()
            from mmdet.models.utils.region_weight import compute_brightness_map
            with torch.no_grad():
                vf, tf = model.backbone(v, t)
                vf = model.neck(vf)
                bm = compute_brightness_map(v)
                for i in range(len(model.fuse_layer.crw_ipm)):
                    _, w_rgb, _ = model.fuse_layer.crw_ipm[i](vf[i], bm)
                    wm = w_rgb[0, 0].cpu().numpy()
                    wm = cv2.resize(wm, (W, H))
                    wh = heatmap(wm)
                    panel.append(labeled(
                        wh, f'LPM w_rgb L{i} (mean={wm.mean():.2f})'))

        out = np.hstack(panel)
        dst = osp.join(args.output_dir, f'{name}.jpg')
        cv2.imwrite(dst, out, [cv2.IMWRITE_JPEG_QUALITY, 92])
        print(f'{name}: brightness mean={bp.mean():.3f} -> {dst}')
    print(f'저장: {args.output_dir}/')


if __name__ == '__main__':
    main()
