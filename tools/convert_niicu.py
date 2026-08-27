"""Convert NII-CU MAPD (rgb-t) to COXNet YOLO-style layout + COCO json.

- RGB (3840x2160) is downscaled to thermal-native 995x560; thermal is symlinked.
- Labels (xyxy in RGB coords, cols: x1 y1 x2 y2 type occluded bad) are scaled to 995x560.
- type semantics (nii-cu-multispectral.org): 0 = visible in both RGB+thermal,
  1 = thermal(-only) box, 2 = RGB(-only) box.
- Mapping (asymmetric by split, decided 2026-07-15 from 2x2 experiments):
  * train: type 0/1 -> person, type 2 -> ignore. Training WITH type1 positives
    (+27% instances, +520 images) beats type0-only supervision by ~2 AP50 on the
    same eval GT (95.3 vs 93.1).
  * val: type 0 -> person, type 1/2 -> ignore. Evaluating vs type0-only GT is the
    only configuration aligning with COXNet Table III (95.3 vs paper 98.2; with
    type1 included AP50 drops to 86.6 since those persons are RGB-occluded).
    KAIST-style instance-level ignore; same philosophy as Speth's aligned-only eval.
  (type0+type1 = 18,736 equals the paper's "18,736 instances" annotation count.)
"""
import json
import os
import os.path as osp

import cv2

SRC = '/data/siwoo/COXNet-release/data/NII-CU/NII_CU_MAPD_dataset/rgb-t'
DST = '/data/siwoo/COXNet-release/data/NII-CU/coxnet'
W, H = 995, 560
SX, SY = W / 3840.0, H / 2160.0


def main():
    for split in ('train', 'val'):
        os.makedirs(osp.join(DST, 'visible', 'images', split), exist_ok=True)
        os.makedirs(osp.join(DST, 'infrared', 'images'), exist_ok=True)
        link = osp.join(DST, 'infrared', 'images', split)
        if not osp.islink(link):
            os.symlink(osp.relpath(osp.join(SRC, 'images', 'thermal', split),
                                   osp.dirname(link)), link)

        images, annotations = [], []
        ann_id = 0
        names = sorted(os.listdir(osp.join(SRC, 'images', 'rgb', split)))
        for img_id, name in enumerate(names):
            dst_img = osp.join(DST, 'visible', 'images', split, name)
            if not osp.isfile(dst_img):
                im = cv2.imread(osp.join(SRC, 'images', 'rgb', split, name))
                cv2.imwrite(dst_img, cv2.resize(im, (W, H)),
                            [cv2.IMWRITE_JPEG_QUALITY, 95])
            images.append(dict(file_name=name, width=W, height=H, id=img_id))

            label = osp.join(SRC, 'labels', split, name.replace('.jpg', '.txt'))
            if osp.isfile(label):
                for line in open(label):
                    p = line.split()
                    if not p:
                        continue
                    x1, y1, x2, y2 = (float(p[0]) * SX, float(p[1]) * SY,
                                      float(p[2]) * SX, float(p[3]) * SY)
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(W, x2), min(H, y2)
                    w, h = x2 - x1, y2 - y1
                    if w <= 1 or h <= 1:
                        continue
                    annotations.append(dict(
                        id=ann_id, image_id=img_id, category_id=0,
                        bbox=[round(x1, 2), round(y1, 2), round(w, 2), round(h, 2)],
                        area=round(w * h, 2),
                        iscrowd=0 if int(p[4]) in
                        ((0, 1) if split == 'train' else (0,)) else 1))
                    ann_id += 1

        coco = dict(images=images, annotations=annotations,
                    categories=[dict(supercategory='none', id=0, name='person')])
        out = osp.join(DST, f'{split}_thermal.json')
        json.dump(coco, open(out, 'w'))
        n_crowd = sum(a['iscrowd'] for a in annotations)
        print(f'{split}: {len(images)} imgs, {len(annotations)} anns '
              f'({n_crowd} ignore) -> {out}')


if __name__ == '__main__':
    main()
