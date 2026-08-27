"""AAM 정렬을 박스 단위로 검증 (warping 없음, quilting 없음).

thermal은 AAM에서 고정(identity). GT(초록)=thermal 객체 위치=정렬 목표.
AAM은 GT에서 offset만큼 떨어진 곳에서 RGB를 샘플링 → 노란박스(GT+AAM offset)=
"AAM이 본 RGB 객체 위치". 노란박스가 실제 RGB 객체 위에 놓이면 = 정렬 성공.
초록↔노랑 이동 화살표. IoU로 겹침 정도(=보정 크기) 표기.

패널: [RGB + 초록(GT)+노랑(AAM추정)+화살표 | thermal + 초록(GT)]

usage:
  python tools/misc/vis_aam_box.py --config ... --checkpoint ... \
     --images 05540 05661 --level 0 --output-dir work_dir/aam_box [--all] [--amp 1]
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


def lab(img, t):
    bar = np.full((26, img.shape[1], 3), 40, np.uint8)
    cv2.putText(bar, t, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    return np.vstack([bar, img])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True); p.add_argument('--checkpoint', required=True)
    p.add_argument('--images', nargs='+'); p.add_argument('--all', action='store_true')
    p.add_argument('--level', type=int, default=0)
    p.add_argument('--amp', type=float, default=1.0, help='offset 증폭(기본1=실제)')
    p.add_argument('--output-dir', required=True)
    args = p.parse_args()

    cfg = Config.fromfile(args.config); cfg.model.backbone.init_cfg = None
    m = build_detector(cfg.model).cuda(); load_checkpoint(m, args.checkpoint, map_location='cpu'); m.eval()
    ds = build_dataset(cfg.data.test, dict(test_mode=True))
    idx_of = {ds.data_infos[i]['filename']: i for i in range(len(ds))}
    img_prefix = cfg.data.test.img_prefix
    tr = cfg.data.test.pipeline[1]['transforms'][2]
    mean = np.array(tr['mean_list'][0]); std = np.array(tr['std_list'][0])
    mean_t = np.array(tr['mean_list'][1]); std_t = np.array(tr['std_list'][1])
    mmcv.mkdir_or_exist(args.output_dir)
    L = args.level
    layer = m.fuse_layer.hofm_layers[L]
    omf = float(getattr(layer, 'om_range_factor', 1.0))

    def find(pp, fn):
        for pth in (osp.join(osp.dirname(img_prefix), pp[0], 'images',
                             osp.basename(img_prefix), fn), osp.join(img_prefix, pp[1], fn)):
            if osp.exists(pth):
                return pth

    names = ([ds.data_infos[i]['filename'][:-4] for i in range(len(ds))]
             if args.all else args.images)
    for name in (mmcv.track_iter_progress(names) if args.all else names):
        fn = name + '.jpg'
        rgb = cv2.imread(find(('visible', 'visible'), fn))
        tp = find(('infrared', 'thermal'), fn)
        th = cv2.resize(cv2.imread(tp), (rgb.shape[1], rgb.shape[0])) if tp else rgb
        H0, W0 = rgb.shape[:2]
        gt = ds.get_ann_info(idx_of[fn])['bboxes']
        if not len(gt):
            continue
        v = cv2.resize(rgb, (640, 512))[:, :, ::-1].astype(np.float32)
        t = cv2.resize(th, (640, 512))[:, :, ::-1].astype(np.float32)
        v = torch.from_numpy(((v - mean) / std).transpose(2, 0, 1)[None]).float().cuda()
        t = torch.from_numpy(((t - mean_t) / std_t).transpose(2, 0, 1)[None]).float().cuda()
        raw = {}
        ho = layer.conv_offset.register_forward_hook(lambda md, i, o: raw.__setitem__('o', o.detach()))
        with torch.no_grad():
            m.extract_feat([v, t])
        ho.remove()
        ro = raw['o']; Hk, Wk = ro.shape[2], ro.shape[3]; tt = ro.tanh()
        sy = (tt[:, 0] * omf * 0.5)[0].cpu().numpy()   # feature-px
        sx = (tt[:, 1] * omf * 0.5)[0].cpu().numpy()

        rgb_o = cv2.convertScaleAbs(rgb, alpha=1.0).copy()
        th_o = cv2.convertScaleAbs(th, alpha=1.3).copy()
        n_ov = 0
        for x1, y1, x2, y2 in gt:
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            fx = min(Wk - 1, max(0, int(cx / W0 * Wk)))
            fy = min(Hk - 1, max(0, int(cy / H0 * Hk)))
            dpx = sx[fy, fx] * (W0 / Wk) * args.amp
            dpy = sy[fy, fx] * (H0 / Hk) * args.amp
            # 초록 = GT(thermal 객체 위치)
            g = (int(x1), int(y1), int(x2), int(y2))
            # 노랑 = GT + AAM offset (AAM이 본 RGB 객체 위치)
            yb = (int(x1 + dpx), int(y1 + dpy), int(x2 + dpx), int(y2 + dpy))
            # IoU(초록,노랑) = 보정 후 겹침
            ix1, iy1 = max(g[0], yb[0]), max(g[1], yb[1])
            ix2, iy2 = min(g[2], yb[2]), min(g[3], yb[3])
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            ua = (g[2] - g[0]) * (g[3] - g[1]) + (yb[2] - yb[0]) * (yb[3] - yb[1]) - inter
            iou = inter / (ua + 1e-9)
            n_ov += (iou > 0.5)
            for img in (rgb_o, th_o):
                cv2.rectangle(img, (g[0], g[1]), (g[2], g[3]), (0, 255, 0), 1)   # 초록 GT
                cv2.rectangle(img, (yb[0], yb[1]), (yb[2], yb[3]), (0, 255, 255), 1)  # 노랑
                cv2.arrowedLine(img, (int(cx + dpx), int(cy + dpy)), (int(cx), int(cy)),
                                (255, 255, 0), 1, tipLength=0.3)  # 노랑→초록(정렬방향)
        panel = np.hstack([
            lab(rgb_o, f'{name} RGB  green=GT(thermal) yellow=AAM-est(x{args.amp:.0f})'),
            lab(th_o, f'thermal  green=GT   (obj IoU>0.5: {n_ov}/{len(gt)})')])
        cv2.imwrite(osp.join(args.output_dir, fn), panel, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if not args.all:
            print(f'{name}: {len(gt)} obj, green↔yellow IoU>0.5 = {n_ov}')
    print('저장:', args.output_dir)


if __name__ == '__main__':
    main()
