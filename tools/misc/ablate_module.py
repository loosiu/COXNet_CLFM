"""모듈 ablation — 각 모듈을 identity로 강제하고 mAP 재측정(기여도).

추론 시 hook으로 모듈을 무력화 → mAP 변화 = 그 모듈의 실제 기여.
mAP 거의 불변이면 그 모듈은 inert(제대로 작동 안 함) = 약점.

--ablate:
  none        : baseline (검증용)
  aam_offset  : AAM offset→0 (grid_sample 정렬 끔, vis_pos=reference)
  msf         : MSF(멀티스케일 attention) 끔 (feat_c += a 스킵)

usage:
  python tools/misc/ablate_module.py --config ... --checkpoint ... --ablate aam_offset
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True); p.add_argument('--checkpoint', required=True)
    p.add_argument('--ablate', default='none', choices=['none', 'aam_offset', 'msf'])
    p.add_argument('--score', type=float, default=0.05)
    args = p.parse_args()

    cfg = Config.fromfile(args.config); cfg.model.backbone.init_cfg = None
    m = build_detector(cfg.model).cuda()
    load_checkpoint(m, args.checkpoint, map_location='cpu'); m.eval()

    handles = []
    if args.ablate == 'aam_offset':
        # conv_offset 출력을 0으로 → tanh(0)=0 → vis_pos=reference (정렬 없음)
        def zero_hook(mod, inp, out):
            return torch.zeros_like(out)
        for lyr in m.fuse_layer.hofm_layers:
            handles.append(lyr.conv_offset.register_forward_hook(zero_hook))
    elif args.ablate == 'msf':
        # msf 경로 proj_msf 출력을 0으로 → feat_c += 0 (멀티스케일 attention 무효)
        def zero_hook(mod, inp, out):
            return torch.zeros_like(out)
        for lyr in m.fuse_layer.hofm_layers:
            if hasattr(lyr, 'proj_msf'):
                handles.append(lyr.proj_msf.register_forward_hook(zero_hook))

    ds = build_dataset(cfg.data.test, dict(test_mode=True))
    idx_of = {ds.data_infos[i]['filename']: i for i in range(len(ds))}
    img_prefix = cfg.data.test.img_prefix
    tr = cfg.data.test.pipeline[1]['transforms'][2]
    mean = np.array(tr['mean_list'][0]); std = np.array(tr['std_list'][0])
    mean_t = np.array(tr['mean_list'][1]); std_t = np.array(tr['std_list'][1])

    def find(pp, fn):
        for pth in (osp.join(osp.dirname(img_prefix), pp[0], 'images',
                             osp.basename(img_prefix), fn), osp.join(img_prefix, pp[1], fn)):
            if osp.exists(pth):
                return pth

    results = []
    for i in mmcv.track_iter_progress(list(range(len(ds)))):
        fn = ds.data_infos[i]['filename']
        rgb = cv2.imread(find(('visible', 'visible'), fn))
        tp = find(('infrared', 'thermal'), fn)
        th = cv2.resize(cv2.imread(tp), (rgb.shape[1], rgb.shape[0])) if tp else rgb
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
        results.append(r)
    for h in handles:
        h.remove()

    print(f"\n===== ablate = {args.ablate} =====")
    metrics = ds.evaluate(results, metric='bbox')
    print({k: round(float(v), 4) for k, v in metrics.items()
           if k in ('bbox_mAP_25', 'bbox_mAP_50', 'bbox_mAP_75',
                    'bbox_mAP_50_tiny', 'bbox_mAP_50_tiny1', 'bbox_mAP_50_small')})


if __name__ == '__main__':
    main()
