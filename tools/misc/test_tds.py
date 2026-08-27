"""Pre-flight for TDS (Thermal Detail Selection) — run before any training.

Three things must hold or a 12-epoch run answers nothing:
  1. identity at init      s=1, k=0 and a uniform R make the module a no-op,
                           so step 0 is exactly the champion
  2. RGB path untouched    v_out must not depend on the tdr branch at all
  3. everything trains     the heatmap loss reaches f_g AND f_d; the detection
                           path reaches s and k; nothing is silently dead
"""
import sys

import torch

sys.path.insert(0, '.')
from mmcv import Config  # noqa: E402
from mmdet.models import build_detector  # noqa: E402

CFG = 'configs/coxnet/coxnet_tds_r50_fpn_1x_rgbtdroneperson.py'

cfg = Config.fromfile(CFG)
model = build_detector(cfg.model).cuda()
dw = model.fuse_layer.idwt_layers[0]
tdr = dw.tdr

g = torch.Generator().manual_seed(0)
t = torch.randn(2, 256, 64, 80, generator=g).cuda()
v = torch.randn(2, 256, 32, 40, generator=g).cuda()

# 1. identity at init
with torch.no_grad():
    out = tdr(dw.DWT(t, mode='full')[0],
              tuple(dw.DWT(t, mode='full')[1:]))
    bands = dw.DWT(t, mode='full')[1:]
dev = max(float((o - b).abs().max()) for o, b in zip(out, bands))
lo, hi = float(tdr.target.min()), float(tdr.target.max())
print(f'[{"PASS" if dev < 1e-5 else "FAIL"}] 초기 항등          편차 {dev:.1e}   R∈[{lo:.3f},{hi:.3f}]')

# 2. RGB path independent of the thermal branch
with torch.no_grad():
    v1, t1 = dw(t, v)
    for p in tdr.parameters():
        p.add_(torch.randn_like(p))
    v2, t2 = dw(t, v)
dv = float((v1 - v2).abs().max())
dt = float((t1 - t2).abs().max())
print(f'[{"PASS" if dv == 0 and dt > 0 else "FAIL"}] RGB 경로 독립      v 변화 {dv:.1e}   t 변화 {dt:.1e}')

# 3. gradients reach everything
model = build_detector(cfg.model).cuda()
tdr = model.fuse_layer.idwt_layers[0].tdr
_, t_out = model.fuse_layer.idwt_layers[0](t, v)
(t_out.pow(2).sum() + tdr.target.pow(2).sum()).backward()
dead = [k for k, p in tdr.named_parameters()
        if p.grad is None or p.grad.abs().max() == 0]
print(f'[{"PASS" if not dead else "FAIL"}] 죽은 파라미터        {dead if dead else 0}')
for k in ('s', 'k'):
    print(f'        {k}.grad {float(getattr(tdr, k).grad.abs().mean()):.2e}')
