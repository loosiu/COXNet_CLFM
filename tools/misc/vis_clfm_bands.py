"""Render CLFM's DWT sub-bands using COXNet's own feature-map visualisation.

The recipe is copied from this repository's `mmdet/models/utils/wavelet_process.py`
(`featuremap_2_heatmap` + `draw_feature_map`, present in the authors' initial
commit), so the output is directly comparable to Fig. 4 and Fig. 8 of the paper:

    sum over channels -> mean over batch -> ReLU -> divide by max
    -> uint8 -> resize to image -> COLORMAP_JET -> heatmap*0.5 + image*0.3

Two deviations, both deliberate and both reported in the output:

* The authors read the source jpg from a hard-coded absolute path; this reads the
  de-normalised tensor coming out of the data pipeline instead.
* The channel SUM is signed.  That is fine for a feature map, whose channels are
  largely one-signed, but a DWT detail band is a difference operator: its
  channels are roughly zero-mean, so summing them cancels the very structure we
  want to see.  Each band is therefore rendered twice -- `sum` (the authors'
  operator, verbatim) and `abs` (channel-mean of |activation|) -- and the two are
  shown side by side so the reader can see what the cancellation does rather than
  having to take it on trust.

Usage:
    PYTHONPATH=. python tools/misc/vis_clfm_bands.py [--level 0] [--n 3]
        [--stride 53] [--out work_dir/abf_evidence]
"""
import argparse
import copy
import os

import cv2
import numpy as np
import torch
from mmcv import Config
from mmcv.runner import load_checkpoint

from mmdet.datasets import build_dataset
from mmdet.models import build_detector

CFG = 'configs/coxnet/coxnet_r50_fpn_1x_rgbtdroneperson.py'
CKPT = ('work_dir/coxnet/rgbtdroneperson/coxnet_r50_fpn_1x/epoch_12.pth')
BANDS = ['LL', 'LH', 'HL', 'HH']
MV, SV = np.array([115.37, 121.82, 122.63]), np.array([85.13, 89.01, 88.27])
MT, ST = np.array([93.10] * 3), np.array([50.24] * 3)


def featuremap_2_heatmap(feature_map, mode='sum'):
    """COXNet's operator verbatim when mode='sum'; |.|-mean when mode='abs'."""
    feature_map = feature_map.detach()
    if mode == 'sum':
        heatmap = feature_map[:, 0, :, :] * 0
        for c in range(feature_map.shape[1]):
            heatmap += feature_map[:, c, :, :]
    else:
        heatmap = feature_map.abs().mean(dim=1)
    heatmap = np.mean(heatmap.cpu().numpy(), axis=0)
    heatmap = np.maximum(heatmap, 0)
    m = heatmap.max()
    return heatmap / m if m > 0 else heatmap


def overlay(heat, img):
    """COXNet's compositing verbatim: JET on the heatmap, then h*0.5 + img*0.3."""
    h = cv2.applyColorMap(np.uint8(255 * heat), cv2.COLORMAP_JET)
    h = cv2.resize(h, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    return np.uint8(np.clip(h * 0.5 + img[:, :, ::-1] * 0.3, 0, 255))   # img is RGB


def denorm(x, m, s):
    return np.clip(x.permute(1, 2, 0).numpy() * s + m, 0, 255).astype(np.uint8)


def label(canvas, text, y, color=(255, 255, 255)):
    cv2.putText(canvas, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 4)
    cv2.putText(canvas, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 1)


def dump_all(model, ds, level, out):
    """Extract the eight sub-bands for EVERY val image and write them to disk.

    The full tensors are 256-channel, which is ~13 GB for one level, so what is
    stored is the channel-mean |activation| of each band -- the same reduction
    the figures display, and the quantity all the reported statistics are
    computed from.  Stored as float16, which is ~25 MB for the whole val set.

    The bands are taken at exactly the point CLFM computes them: straight out of
    `DWT`, before the `cat`/`fusion_conv`.  Thermal is decomposed at its own
    resolution, visible after `deconv`, matching the module.  The model is in
    eval mode, so these are the values that exist at inference.
    """
    import numpy as np
    dw = model.fuse_layer.idwt_layers[level]
    names = [f'{m}_{b}' for m in ('T', 'V') for b in BANDS]
    maps = {n: [] for n in names}
    rows = []
    for i in range(len(ds)):
        s = ds[i]
        v_img, t_img = s['img'][0].data, s['img'][1].data
        boxes = s['gt_bboxes'].data.numpy()
        with torch.no_grad():
            vf, tf = model.backbone(v_img[None].cuda(), t_img[None].cuda())
            vf, tf = model.neck(vf), model.neck_t(tf)
            packs = (dw.DWT(tf[level], mode='full'),
                     dw.DWT(dw.deconv(vf[level]), mode='full'))
        m = {}
        for mi, pack in enumerate(packs):
            for bi, b in enumerate(BANDS):
                m[names[mi * 4 + bi]] = pack[bi][0].abs().mean(0).cpu().numpy()
        h, w = m['T_LL'].shape
        H, W = v_img.shape[-2:]
        occ = np.zeros((h, w), bool)
        for b in boxes:
            x0, y0 = int(b[0] * w / W), int(b[1] * h / H)
            x1 = max(int(np.ceil(b[2] * w / W)), x0 + 1)
            y1 = max(int(np.ceil(b[3] * h / H)), y0 + 1)
            occ[max(y0, 0):min(y1, h), max(x0, 0):min(x1, w)] = True
        rgb = np.clip(v_img.permute(1, 2, 0).numpy() * SV + MV, 0, 255)
        lum = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).mean()
        row = dict(idx=i, n_obj=len(boxes), luminance=round(float(lum), 2),
                   gt_cells=int(occ.sum()))
        for n in names:
            maps[n].append(m[n].astype(np.float16))
            if occ.any() and not occ.all():
                g, bg = m[n][occ].mean(), m[n][~occ].mean()
                row[f'{n}_gt'] = round(float(g), 5)
                row[f'{n}_bg'] = round(float(bg), 5)
                row[f'{n}_ratio'] = round(float(g / (bg + 1e-9)), 4)
        rows.append(row)
        if (i + 1) % 200 == 0:
            print(f'  {i + 1}/{len(ds)}')

    npz = f'{out}/clfm_bands_level{level}.npz'
    np.savez_compressed(npz, **{n: np.stack(maps[n]) for n in names})
    import csv
    csvp = f'{out}/clfm_bands_level{level}_stats.csv'
    keys = list(rows[0].keys())
    for r in rows:
        for k in keys:
            r.setdefault(k, '')
    with open(csvp, 'w', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        wr.writerows(rows)
    sz = os.path.getsize(npz) / 1e6
    print(f'\nsaved {npz}  ({sz:.1f} MB, 8 bands x {len(ds)} images x {h}x{w}, float16)')
    print(f'saved {csvp}  ({len(rows)} rows, per-image per-band GT/background stats)')
    print(f'\nload with:  d = np.load("{npz}");  d["T_LH"].shape -> ({len(ds)}, {h}, {w})')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--level', type=int, default=0)
    ap.add_argument('--n', type=int, default=3)
    ap.add_argument('--stride', type=int, default=53)
    ap.add_argument('--out', default='work_dir/abf_evidence')
    ap.add_argument('--dump', action='store_true',
                    help='extract the bands for every val image instead of '
                         'rendering figures')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    cfg = Config.fromfile(CFG)
    val = copy.deepcopy(cfg.data.val)
    val.pipeline = copy.deepcopy(cfg.data.train.pipeline)     # keeps gt_bboxes
    for t in val.pipeline:
        if t['type'] == 'RandomFlip':
            t['flip_ratio'] = 0.0
    ds = build_dataset(val)
    model = build_detector(cfg.model)
    load_checkpoint(model, CKPT, map_location='cpu')
    model = model.cuda().eval()          # eval == what happens at inference
    dw = model.fuse_layer.idwt_layers[args.level]

    if args.dump:
        dump_all(model, ds, args.level, args.out)
        return

    done = 0
    for i in range(0, len(ds), args.stride):
        s = ds[i]
        boxes = s['gt_bboxes'].data.numpy()
        if len(boxes) < 6:
            continue
        v_img, t_img = s['img'][0].data, s['img'][1].data
        with torch.no_grad():
            vf, tf = model.backbone(v_img[None].cuda(), t_img[None].cuda())
            vf, tf = model.neck(vf), model.neck_t(tf)
            T = dw.DWT(tf[args.level], mode='full')
            V = dw.DWT(dw.deconv(vf[args.level]), mode='full')

        vis, the = denorm(v_img, MV, SV), denorm(t_img, MT, ST)
        H, W = vis.shape[:2]

        def boxed(img_bgr, tag, sub=None, warn=False):
            c = img_bgr.copy()
            for b in boxes:
                cv2.rectangle(c, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])),
                              (0, 255, 0), 1)
            label(c, tag, 24)
            if sub:
                label(c, sub, 48, (80, 80, 255) if warn else (255, 255, 255))
            return c

        def band_panel(t_or_v, pack, bi, mode):
            nm = BANDS[bi]
            kept = not (t_or_v == 'T' and nm != 'LL')
            tag = f'{"THERMAL" if t_or_v == "T" else "VISIBLE"} {nm}  [{mode}]'
            return boxed(overlay(featuremap_2_heatmap(pack[bi], mode), vis), tag,
                         'USED by CLFM' if kept else 'DISCARDED', warn=not kept)

        # 4 rows x 5 panels: (input, LL, LH, HL, HH) for each modality x each operator
        rows_spec = [('T', T, 'sum', vis), ('V', V, 'sum', the),
                     ('T', T, 'abs', vis), ('V', V, 'abs', the)]
        pad_w = np.full((H, 6, 3), 255, np.uint8)
        rows = []
        for pre, pack, mode, src in rows_spec:
            first = boxed(src[:, :, ::-1].copy(),
                          f'{"VISIBLE" if src is vis else "THERMAL"} input')
            panels = [first] + [band_panel(pre, pack, bi, mode) for bi in range(4)]
            rows.append(np.hstack(sum([[p, pad_w] for p in panels], [])[:-1]))
        pad_h = np.full((6, rows[0].shape[1], 3), 255, np.uint8)
        canvas = np.vstack(sum([[r, pad_h] for r in rows], [])[:-1])
        p = f'{args.out}/coxnet_style_bands_{done}.png'
        cv2.imwrite(p, canvas)
        print(f'saved {p}   (val #{i}, {len(boxes)} objects, level {args.level})')
        done += 1
        if done >= args.n:
            break


if __name__ == '__main__':
    main()
