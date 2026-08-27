"""AAM 효과 정성 검증 — AAM 예측 offset vs GT 미정렬 (ground-truth 대비).

train 서브셋은 visible/thermal GT 둘 다 있음(sub_train_*.json).
객체마다: GT 미정렬 벡터(thermal중심→visible중심) = AAM이 고쳐야 할 것
        AAM 예측 offset 벡터(conv_offset, thermal위치에서) = AAM이 실제 하는 것
둘이 일치하면 AAM 효과적. AAM벡터가 짧거나(포화) 방향 틀리면 비효과적.

thermal 이미지 위: 초록박스=thermal GT, 파랑박스=visible GT(RGB객체 실제위치),
  초록 화살표=GT 미정렬(고쳐야 할 것), 자홍 화살표=AAM 예측 offset(실제 하는 것). 둘 다 ×amp.
라벨에 |AAM|/|GT| 비율(1이면 완벽), 방향 cos(1이면 정방향).

usage:
  python tools/misc/vis_aam_verify.py --config ... --checkpoint ... \
     --root data/RGBTDronePerson --amp 5 --output-dir work_dir/aam_verify [--all]
"""
import argparse
import json
import os.path as osp

import cv2
import mmcv
import numpy as np
import torch
from mmcv import Config
from mmcv.runner import load_checkpoint
from mmdet.models import build_detector


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True); p.add_argument('--checkpoint', required=True)
    p.add_argument('--root', default='data/RGBTDronePerson')
    p.add_argument('--images', nargs='+'); p.add_argument('--all', action='store_true')
    p.add_argument('--level', type=int, default=0)
    p.add_argument('--amp', type=float, default=5.0)
    p.add_argument('--single-only', action='store_true', help='단일객체 이미지만(무모호 매칭)')
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--output-dir', required=True)
    args = p.parse_args()

    cfg = Config.fromfile(args.config); cfg.model.backbone.init_cfg = None
    m = build_detector(cfg.model).cuda(); load_checkpoint(m, args.checkpoint, map_location='cpu'); m.eval()
    tr = cfg.data.test.pipeline[1]['transforms'][2]
    mean = np.array(tr['mean_list'][0]); std = np.array(tr['std_list'][0])
    mean_t = np.array(tr['mean_list'][1]); std_t = np.array(tr['std_list'][1])
    mmcv.mkdir_or_exist(args.output_dir)
    L = args.level
    layer = m.fuse_layer.hofm_layers[L]
    omf = float(getattr(layer, 'om_range_factor', 1.0))

    V = json.load(open(osp.join(args.root, 'sub_train_visible.json')))
    T = json.load(open(osp.join(args.root, 'sub_train_thermal.json')))
    vimg = {i['id']: i['file_name'] for i in V['images']}
    timg = {i['id']: i['file_name'] for i in T['images']}

    def by_file(js, img):
        d = {}
        for a in js['annotations']:
            d.setdefault(img[a['image_id']], []).append(a['bbox'])
        return d
    vb, tb = by_file(V, vimg), by_file(T, timg)
    common = sorted(set(vb) & set(tb))
    names = ([n for n in (args.images or [])] if not args.all else common)

    all_ratio, all_cos = [], []
    saved = 0
    for name in (mmcv.track_iter_progress(common) if args.all else names):
        fn = name if name.endswith('.jpg') else name + '.jpg'
        if fn not in vb or fn not in tb:
            continue
        Vb, Tb = vb[fn], tb[fn]
        if len(Vb) != len(Tb) or not len(Vb):
            continue
        if args.single_only and len(Vb) != 1:
            continue
        rgb = cv2.imread(osp.join(args.root, 'train', 'visible', fn))
        th = cv2.imread(osp.join(args.root, 'train', 'thermal', fn))
        if rgb is None or th is None:
            continue
        th = cv2.resize(th, (rgb.shape[1], rgb.shape[0]))
        H0, W0 = rgb.shape[:2]
        v = cv2.resize(rgb, (640, 512))[:, :, ::-1].astype(np.float32)
        t = cv2.resize(th, (640, 512))[:, :, ::-1].astype(np.float32)
        vt = torch.from_numpy(((v - mean) / std).transpose(2, 0, 1)[None]).float().cuda()
        tt_ = torch.from_numpy(((t - mean_t) / std_t).transpose(2, 0, 1)[None]).float().cuda()
        raw = {}
        ho = layer.conv_offset.register_forward_hook(lambda md, i, o: raw.__setitem__('o', o.detach()))
        with torch.no_grad():
            m.extract_feat([vt, tt_])
        ho.remove()
        ro = raw['o']; Hk, Wk = ro.shape[2], ro.shape[3]; tt = ro.tanh()
        sy = (tt[:, 0] * omf * 0.5)[0].cpu().numpy()   # feature-px
        sx = (tt[:, 1] * omf * 0.5)[0].cpu().numpy()
        # 입력(640x512) 기준. 원본이 다르면 스케일
        rx, ry = 640.0 / W0, 512.0 / H0

        img = cv2.convertScaleAbs(th, alpha=1.3)
        ratios, coss = [], []
        for bv, bt in zip(Vb, Tb):
            vcx, vcy = bv[0] + bv[2] / 2, bv[1] + bv[3] / 2      # visible 객체 중심(원본px)
            tcx, tcy = bt[0] + bt[2] / 2, bt[1] + bt[3] / 2      # thermal 객체 중심
            gdx, gdy = vcx - tcx, vcy - tcy                      # GT 미정렬 벡터(원본px)
            # AAM offset: thermal 위치의 feature cell에서 (입력640기준 feature-px→입력px→원본px)
            fx = min(Wk - 1, max(0, int(tcx * rx / 8))); fy = min(Hk - 1, max(0, int(tcy * ry / 8)))
            adx = sx[fy, fx] * 8 / rx; ady = sy[fy, fx] * 8 / ry  # AAM offset(원본px)
            gmag = (gdx ** 2 + gdy ** 2) ** 0.5; amag = (adx ** 2 + ady ** 2) ** 0.5
            if gmag > 1:
                ratios.append(amag / gmag)
                coss.append((gdx * adx + gdy * ady) / (gmag * amag + 1e-9))
            # 박스
            cv2.rectangle(img, (int(bt[0]), int(bt[1])), (int(bt[0] + bt[2]), int(bt[1] + bt[3])), (255, 255, 255), 1)
            cv2.rectangle(img, (int(bv[0]), int(bv[1])), (int(bv[0] + bv[2]), int(bv[1] + bv[3])), (255, 120, 0), 1)
            A = args.amp
            cv2.arrowedLine(img, (int(tcx), int(tcy)), (int(tcx + gdx * A), int(tcy + gdy * A)), (0, 220, 0), 2, tipLength=.25)   # GT
            cv2.arrowedLine(img, (int(tcx), int(tcy)), (int(tcx + adx * A), int(tcy + ady * A)), (255, 0, 255), 2, tipLength=.25)  # AAM
        mr = float(np.mean(ratios)) if ratios else 0
        mc = float(np.mean(coss)) if coss else 0
        all_ratio += ratios; all_cos += coss
        bar = np.full((30, img.shape[1], 3), 30, np.uint8)
        cv2.putText(bar, f'{name}  white=Ther-GT blue=Vis-GT | green=GT-misalign magenta=AAM-offset (x{A:.0f})'
                    f'  |AAM|/|GT|={mr:.2f} dir-cos={mc:.2f}', (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imwrite(osp.join(args.output_dir, fn), np.vstack([bar, img]), [cv2.IMWRITE_JPEG_QUALITY, 92])
        saved += 1
        if not args.all:
            print(f'{name}: |AAM|/|GT|={mr:.2f} dir-cos={mc:.2f} ({len(ratios)}obj)')
        if args.limit and saved >= args.limit:
            break
    if all_ratio:
        print(f'\n=== 전체 {len(all_ratio)}객체 ===')
        print(f'  |AAM offset|/|GT 미정렬| 평균 {np.mean(all_ratio):.2f} 중앙 {np.median(all_ratio):.2f}  (1=완벽보정, <1=과소보정)')
        print(f'  방향 cos 평균 {np.mean(all_cos):.2f}  (1=정방향, 0=무관, <0=역방향)')
    print('저장:', args.output_dir)


if __name__ == '__main__':
    main()
