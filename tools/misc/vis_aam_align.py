"""AAM 정렬 시각화 — 정렬 직전 visible feature vs 정렬 후(warp) vs thermal.

HOFM.forward 안에서 feat_v는 grid_sample(vis_pos)로 warp됨. F.grid_sample을 임시
패치해 (입력=정렬前 feat_v, 출력=정렬後 feat_v)를 캡처하고, feat_t와 offset도 잡는다.

패널: [thermal | feat_t 활성 | feat_v(정렬前) | feat_v(정렬後) | offset 크기(px)]
라벨에 교차모달 코사인유사도(정렬前→後)와 평균 offset(px). 유사도가 오르면 정렬이 유효.

usage:
  python tools/misc/vis_aam_align.py --config ... --checkpoint ... \
     --images 05661 06064 --level 0 --output-dir work_dir/aam_align
"""
import argparse
import os.path as osp

import cv2
import mmcv
import numpy as np
import torch
import torch.nn.functional as F
from mmcv import Config
from mmcv.runner import load_checkpoint
from mmdet.models import build_detector
from mmdet.datasets import build_dataset


def heat(feat, size):
    """feature 채널평균 |활성| → 정규화 → JET heatmap (size=(W,H))."""
    m = feat.abs().mean(1)[0].detach().cpu().numpy()
    m = (m - m.min()) / (m.max() - m.min() + 1e-9)
    m = cv2.resize(m, size)
    return cv2.applyColorMap(np.uint8(255 * m), cv2.COLORMAP_JET)


def cos_sim(a, b):
    a = a.flatten(2); b = b.flatten(2)
    a = a / (a.norm(dim=1, keepdim=True) + 1e-6)
    b = b / (b.norm(dim=1, keepdim=True) + 1e-6)
    return float((a * b).sum(1).mean())


def lab(img, t):
    bar = np.full((26, img.shape[1], 3), 40, np.uint8)
    cv2.putText(bar, t, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return np.vstack([bar, img])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True); p.add_argument('--checkpoint', required=True)
    p.add_argument('--images', nargs='+', required=True)
    p.add_argument('--level', type=int, default=0)
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

    for name in args.images:
        fn = name + '.jpg'
        rgb = cv2.imread(find(('visible', 'visible'), fn))
        tp = find(('infrared', 'thermal'), fn)
        th = cv2.resize(cv2.imread(tp), (rgb.shape[1], rgb.shape[0])) if tp else rgb
        v = cv2.resize(rgb, (640, 512))[:, :, ::-1].astype(np.float32)
        t = cv2.resize(th, (640, 512))[:, :, ::-1].astype(np.float32)
        v = torch.from_numpy(((v - mean) / std).transpose(2, 0, 1)[None]).float().cuda()
        t = torch.from_numpy(((t - mean_t) / std_t).transpose(2, 0, 1)[None]).float().cuda()

        # offset(raw) 캡처
        raw = {}
        ho = layer.conv_offset.register_forward_hook(lambda md, i, o: raw.__setitem__('o', o.detach()))
        # grid_sample 캡처 (레벨 L의 첫 호출 = feat_v 정렬)
        gs = []
        orig = F.grid_sample
        def patched(input, grid, **kw):
            out = orig(input, grid, **kw); gs.append((input.detach(), out.detach())); return out
        F.grid_sample = patched
        with torch.no_grad():
            m.extract_feat([v, t])
        F.grid_sample = orig; ho.remove()

        # 레벨 L: gs[2L]=feat_v(pre,aligned), gs[2L+1]=feat_t(pre, ~동일)
        fv_pre, fv_al = gs[2 * L]
        ft = gs[2 * L + 1][0]
        # offset → px 크기
        ro = raw['o']; tt = ro.tanh()
        sy = (tt[:, 0] * omf * 0.5)[0].cpu().numpy(); sx = (tt[:, 1] * omf * 0.5)[0].cpu().numpy()
        offmag = np.sqrt(sy ** 2 + sx ** 2)   # feature-px

        before = cos_sim(fv_pre, ft); after = cos_sim(fv_al, ft)
        Hk, Wk = offmag.shape
        base = cv2.resize(cv2.convertScaleAbs(th, alpha=1.4), (Wk * 8, Hk * 8))
        size = (Wk * 8, Hk * 8)
        omh = cv2.applyColorMap(np.uint8(255 * offmag / (offmag.max() + 1e-9)),
                                cv2.COLORMAP_HOT)
        omh = cv2.resize(omh, size, interpolation=cv2.INTER_NEAREST)
        panel = np.hstack([
            lab(base, f'{name} thermal'),
            lab(heat(ft, size), 'feat_t (thermal)'),
            lab(heat(fv_pre, size), 'feat_v BEFORE align'),
            lab(heat(fv_al, size), 'feat_v AFTER align'),
            lab(omh, f'offset |{offmag.mean():.2f}fpx|(~{offmag.mean()*8:.1f}inpx)')])
        # 유사도 요약 바
        s = np.full((30, panel.shape[1], 3), 30, np.uint8)
        col = (0, 255, 0) if after > before else (0, 0, 255)
        cv2.putText(s, f'P{L+3}  cross-modal cos-sim  BEFORE {before:.4f} -> AFTER {after:.4f}'
                    f'  (delta {after-before:+.4f})   mean offset {offmag.mean()*8:.2f} input-px',
                    (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 1)
        panel = np.vstack([s, panel])
        cv2.imwrite(osp.join(args.output_dir, fn), panel, [cv2.IMWRITE_JPEG_QUALITY, 92])
        print(f'{name} P{L+3}: cos-sim {before:.4f} -> {after:.4f} '
              f'({after-before:+.4f}) | offset {offmag.mean()*8:.2f} inpx | gs calls {len(gs)}')
    print('저장:', args.output_dir)


if __name__ == '__main__':
    main()
