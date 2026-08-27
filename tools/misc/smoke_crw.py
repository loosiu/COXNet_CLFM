"""CRW/LPM(CCRDet 이식) 전체 모델 스모크 — GPU 필요 (CLFM v3의 DWT가 CUDA 전용).

CPU 단위검증(모듈·보간·grad·하위호환)은 통과 상태이므로, 이 스크립트는
CLFM v3 + wf_loss + GFLQHead까지 실제 챔피언 경로 관통만 확인한다.

usage (GPU 여유 ~2GB면 충분):
  python tools/misc/smoke_crw.py [config_path]
  기본 config = LPM 단독(coxnet_lpm_r50_fpn_1x_rgbtdroneperson.py)
"""
import sys
import torch
from mmcv import Config
from mmdet.models import build_detector

cfg_path = sys.argv[1] if len(sys.argv) > 1 else \
    'configs/coxnet/coxnet_lpm_r50_fpn_1x_rgbtdroneperson.py'
cfg = Config.fromfile(cfg_path)
cfg.model.backbone.init_cfg = None
model = build_detector(cfg.model).cuda()
model.CLASSES = ('person', 'rider', 'crowd')

n_crw = sum(p.numel() for n, p in model.named_parameters() if 'crw_' in n)
n_tot = sum(p.numel() for p in model.parameters())
print(f'{cfg_path}\nCRW params: {n_crw:,} / total {n_tot:,} (+{n_crw / n_tot * 100:.3f}%)')

B, H, W = 1, 512, 640  # 실제 학습 해상도 (CLFM DWT가 작은 맵에서 깨져서 축소 불가)
v = torch.randn(B, 3, H, W).cuda()
t = torch.randn(B, 3, H, W).cuda()
metas = [dict(img_shape=(H, W, 3), pad_shape=(H, W, 3), ori_shape=(H, W, 3),
              scale_factor=1.0, batch_input_shape=(H, W)) for _ in range(B)]
gt_b = [torch.tensor([[10., 10., 30., 40.], [50., 60., 70., 90.]]).cuda()
        for _ in range(B)]
gt_l = [torch.tensor([0, 0]).cuda() for _ in range(B)]

model.train()
losses = model.forward_train([v, t], metas, gt_b, gt_l)
print('train losses:', {k: f'{(sum(x.item() for x in v_) if isinstance(v_, list) else v_.item()):.4f}'
                        for k, v_ in losses.items()})
loss = sum(sum(x for x in v_) if isinstance(v_, list) else v_
           for v_ in losses.values())
loss.backward()
missing = [n for n, p in model.named_parameters()
           if 'crw_' in n and p.grad is None]
assert not missing, f'CRW no-grad: {missing}'
print('CRW grads ok')

model.eval()
with torch.no_grad():
    res = model.simple_test([v, t], metas)
print(f'eval ok: {len(res)} imgs x {len(res[0])} classes')
print('FULL SMOKE PASS')
