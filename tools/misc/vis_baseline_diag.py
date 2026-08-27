"""baseline COXNet 진단 뷰 — 저조도/clutter 후보에서 잡나 못 잡나 + 어디를 보나.

한 이미지당 4분할(RGB·thermal은 원본 그대로):
  [RGB 원본] [Thermal 원본] [검출: 초록=TP,빨강=놓친 GT] [Objectness score]

가설 확인:
  A(저조도): RGB 새까만데 → 검출/CAM이 객체(thermal 위치)를 잡으면 = thermal이 구제 = 가설 A 반박
             CAM이 객체에서 벗어나고 miss면 = 저조도가 진짜 문제 = 가설 A 지지
  miss인데 CAM은 객체를 봄 = "찾긴 했는데 국소화 실패(tiny)" = 병목은 크기(generic)

usage:
  python tools/misc/vis_baseline_diag.py \
     --config configs/coxnet/coxnet_r50_fpn_1x_rgbtdroneperson.py \
     --checkpoint work_dir/coxmamba/rgbtdroneperson/coxnet_r50_fpn_1x/epoch_12.pth \
     --images 06064 06061 06056 --output-dir work_dir/diag_cam
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

G, R = (0, 255, 0), (0, 0, 255)


def iou(a, b):
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None); i = wh[..., 0] * wh[..., 1]
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return i / (aa[:, None] + ab[None, :] - i + 1e-9)


def grad_cam(model, v, t, metas, L=0):
    act = {}

    def hook(m, i, o):
        o.retain_grad(); act['A'] = o
    h = model.fuse_layer.hofm_layers[L].register_forward_hook(hook)
    feats = model.extract_feat([v, t])
    cls = model.bbox_head(feats)[0][L]
    score = cls.sigmoid().max(1)[0].sum()
    model.zero_grad(); score.backward()
    A = act['A']; w = A.grad.mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((w * A).sum(1))[0]
    h.remove()
    return cam.detach().cpu().numpy()


def score_map(model, v, t, metas, L=0):
    """분류 head의 위치별 objectness 예측맵 (h,w). 검출된 객체가 높게 뜸."""
    with torch.no_grad():
        feats = model.extract_feat([v, t])
        cls = model.bbox_head(feats)[0][L]          # (B, num_cls, h, w)
        s = cls.sigmoid().max(1)[0][0]              # (h, w) 최대 클래스 확신도
    return s.detach().cpu().numpy()


def cam_overlay(base_img, cam):
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-9)
    cam = cv2.resize(cam, (base_img.shape[1], base_img.shape[0]))
    hm = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    out = cv2.addWeighted(base_img, 0.5, hm, 0.5, 0)
    return out


def lab(img, t):
    bar = np.full((26, img.shape[1], 3), 40, np.uint8)
    cv2.putText(bar, t, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    return np.vstack([bar, img])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--images', nargs='+')
    p.add_argument('--all', action='store_true', help='val 전체')
    p.add_argument('--skip-existing', action='store_true', help='이미 만든 건 건너뜀(이어서)')
    p.add_argument('--score', type=float, default=0.3)
    p.add_argument('--iou', type=float, default=0.5, help='TP 판정 IoU')
    p.add_argument('--output-dir', required=True)
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

    def find(pp, fn):
        for pth in (osp.join(osp.dirname(img_prefix), pp[0], 'images',
                             osp.basename(img_prefix), fn),
                    osp.join(img_prefix, pp[1], fn)):
            if osp.exists(pth):
                return pth

    names = ([ds.data_infos[i]['filename'][:-4] for i in range(len(ds))]
             if args.all else args.images)
    rank = []
    for name in (mmcv.track_iter_progress(names) if args.all else names):
        fn = name + '.jpg'
        if args.skip_existing and osp.exists(osp.join(args.output_dir, fn)):
            continue
        rgb = cv2.imread(find(('visible', 'visible'), fn))
        tp = find(('infrared', 'thermal'), fn)
        th = cv2.resize(cv2.imread(tp), (rgb.shape[1], rgb.shape[0])) if tp else rgb
        gt = ds.get_ann_info(idx_of[fn])['bboxes']
        v = cv2.resize(rgb, (640, 512))[:, :, ::-1].astype(np.float32)
        t = cv2.resize(th, (640, 512))[:, :, ::-1].astype(np.float32)
        v = torch.from_numpy(((v - mean) / std).transpose(2, 0, 1)[None]).float().cuda()
        t = torch.from_numpy(((t - mean_t) / std_t).transpose(2, 0, 1)[None]).float().cuda()
        metas = [dict(img_shape=(512, 640, 3), pad_shape=(512, 640, 3),
                      ori_shape=(rgb.shape[0], rgb.shape[1], 3),
                      scale_factor=np.array([640 / rgb.shape[1], 512 / rgb.shape[0]] * 2,
                                            dtype=np.float32), batch_input_shape=(512, 640))]
        with torch.no_grad():
            r = m.simple_test([v, t], metas)[0]
        dets = np.vstack([c for c in r if len(c)]) if any(len(c) for c in r) else np.zeros((0, 5))
        kept = dets[dets[:, 4] >= args.score][:, :4] if len(dets) else np.zeros((0, 4))
        cam = score_map(m, v, t, metas, L=0)   # objectness 예측맵 (검출객체가 높게)

        bright = float(cv2.cvtColor(rgb[:, :, ::-1],
                                    cv2.COLOR_RGB2YUV)[:, :, 0].mean()) / 255
        det = th.copy()                                      # 원본 thermal 위 검출표시
        mm = iou(gt, kept).max(1) if (len(gt) and len(kept)) else np.zeros(len(gt))
        n_miss = 0
        for k in range(len(gt)):
            x1, y1, x2, y2 = gt[k].astype(int)
            ok = len(kept) and mm[k] >= args.iou
            cv2.rectangle(det, (x1, y1), (x2, y2), G if ok else R, 2)
            n_miss += (0 if ok else 1)
        camv = cam_overlay(th, cam)                          # 원본 thermal 위 heatmap
        panel = np.hstack([lab(rgb, f'RGB (Y~{bright:.3f})'),      # 원본 RGB
                           lab(th, 'Thermal'),                     # 원본 thermal
                           lab(det, f'Detection (miss {n_miss}/{len(gt)})'),
                           lab(camv, 'Objectness score')])
        cv2.imwrite(osp.join(args.output_dir, fn), panel, [cv2.IMWRITE_JPEG_QUALITY, 92])
        rank.append((name, len(gt), n_miss, bright))
        if not args.all:
            print(f'{name}: GT {len(gt)}, miss {n_miss}')
    rank.sort(key=lambda r: (-r[2], -r[1]))     # miss 많은 순
    with open(osp.join(args.output_dir, '_ranking.txt'), 'w') as f:
        f.write('name  n_gt  n_miss  bright   (miss 많은 순 = 볼 가치 큰 순)\n')
        for r in rank:
            f.write(f'{r[0]}  {r[1]}  {r[2]}  {r[3]:.3f}\n')
    tot_gt = sum(r[1] for r in rank); tot_miss = sum(r[2] for r in rank)
    print(f'\n총 {len(rank)}장 | GT {tot_gt}, miss {tot_miss} (recall~{1-tot_miss/max(tot_gt,1):.3f})')
    print('저장:', args.output_dir, '| miss 순위: _ranking.txt')


if __name__ == '__main__':
    main()
