"""head 직전 융합 feature 응답을 RGB 위에 오버레이 — baseline vs HF-gate.
논문식: 객체 위에서 응답이 튀는지(배경과 구분) 확인. GT는 흰 박스.

usage:
  python tools/misc/vis_feat_compare.py --base-config <cfg> --base-ckpt <pth> \
     --new-config <hfgate> --new-ckpt <pth> --images ... --output-dir ...
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


def score_map(model, v, t, metas, L):
    """검출 head의 분류 점수맵(=객체 위치) 추출. (h,w) numpy [0,1]."""
    with torch.no_grad():
        feats = model.extract_feat([v, t])
        out = model.bbox_head(feats)
        cls = out[0][L]                       # (B, num_classes, h, w)
        s = cls.sigmoid().max(1)[0][0]        # 클래스 max = objectness
    return s.cpu().numpy()


def overlay(base_img, r, gt=None):
    r = (r - r.min()) / (r.max() - r.min() + 1e-9)
    r = cv2.resize(r, (base_img.shape[1], base_img.shape[0]))
    hm = cv2.applyColorMap(np.uint8(255 * r), cv2.COLORMAP_JET)
    out = cv2.addWeighted(base_img, 0.5, hm, 0.5, 0)
    if gt is not None:
        for x1, y1, x2, y2 in gt.astype(int):
            cv2.rectangle(out, (x1, y1), (x2, y2), (255, 255, 255), 1)
    return out


def build(cfg_path, ckpt):
    cfg = Config.fromfile(cfg_path)
    cfg.model.backbone.init_cfg = None
    m = build_detector(cfg.model).cuda()
    load_checkpoint(m, ckpt, map_location='cpu')
    m.eval()
    return m, cfg


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base-config'); p.add_argument('--base-ckpt')
    p.add_argument('--new-config'); p.add_argument('--new-ckpt')
    p.add_argument('--images', nargs='+', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--level', type=int, default=0)
    args = p.parse_args()

    mb, cfg = build(args.base_config, args.base_ckpt)
    mn, _ = build(args.new_config, args.new_ckpt)
    L = args.level

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

    for name in args.images:
        fn = name + '.jpg'
        rgb = cv2.imread(find(('visible', 'visible'), fn))
        tp = find(('infrared', 'thermal'), fn)
        th = cv2.resize(cv2.imread(tp), (rgb.shape[1], rgb.shape[0])) if tp else rgb
        H, W = rgb.shape[:2]
        gt = ds.get_ann_info(idx_of[fn])['bboxes']
        v = cv2.resize(rgb, (640, 512))[:, :, ::-1].astype(np.float32)
        t = cv2.resize(th, (640, 512))[:, :, ::-1].astype(np.float32)
        vt = torch.from_numpy(((v - mean) / std).transpose(2, 0, 1)[None]).float().cuda()
        tt = torch.from_numpy(((t - mean_t) / std_t).transpose(2, 0, 1)[None]).float().cuda()
        metas = [dict(img_shape=(512, 640, 3), pad_shape=(512, 640, 3),
                      ori_shape=(512, 640, 3), scale_factor=1.0,
                      batch_input_shape=(512, 640))]
        sb = score_map(mb, vt, tt, metas, L)
        sn = score_map(mn, vt, tt, metas, L)
        panel = np.hstack([
            lab(rgb, 'RGB'),
            lab(overlay(rgb, sb, gt), 'baseline score'),
            lab(overlay(rgb, sn, gt), 'HF-gate score'),
            lab(cv2.convertScaleAbs(th, alpha=1.5), 'thermal')])
        cv2.imwrite(osp.join(args.output_dir, fn), panel, [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(f'{name}: 저장')
    print('저장:', args.output_dir)


if __name__ == '__main__':
    main()
