# TDS — Thermal Detail Selection.  최종 방법 config.
#
# CLFM 은 visible 만 웨이블릿 정제하고 thermal 은 LL 을 재료로 내줄 뿐 자신은
# 정제받지 않는다.  버려지는 thermal 세부 밴드의 객체 판별 AUC 는 0.90~0.92
# (재구성에 쓰이는 visible 밴드는 0.36), GT 게이팅 오라클은 객체 자리 +1.93 /
# 배경 자리 -1.99.  TDS 는 그 빈 자리를 채운다:
#
#   D  = f_d([LH,HL,HH])            세부 반응의 서술
#   R  = sigmoid(f_g([LL, D]))      Detail Relevance Map (GT heatmap 지도)
#   g  = s(1 + k(R - R̄))            밴드별·위치별 가중 (전부 학습, 항등 시작)
#   T' = IDWT(LL, g·detail)         LL 원본 보존 → 구조·위치 불변
#
# RGB 경로는 챔피언 up_new 와 비트 단위 동일.  근거·측정은 maclfm.py docstring.
#
# 결과 (seed 0, best ep10):  mAP50 45.98 → 46.77,  tiny1 18.01 → 24.12.
# knockout(추론에서 s=1,k=0): 45.71 로 baseline 회귀, tiny1 반토막 — 이득이
# 모듈 순전파에 실려 있음이 결정적으로 확인됨.
#
# ablation 은 config 를 늘리지 말고 아래로:
#   --cfg-options model.tdr_loss_weight=0.0    지도 제거 (R 사망 → g=s 균일 재수용)
#                 model.tdr_hm_min_sigma=0.5   heatmap 을 좁게
#                 model.tdr_levels="(0,1,2,3)" 전 레벨 (죽은 레벨 확인용)
_base_ = ['./coxnet_r50_fpn_1x_rgbtdroneperson.py']

model = dict(
    clfm_mode='up_tdr',
    tdr_levels=(0, ),          # 레벨 0 만: 레벨 1~3 은 CLFM 기울기가 1e-4 이하
    tdr_hm_min_sigma=1.0,      # 타깃 Gaussian 최소 폭 (셀 단위)
    tdr_loss_weight=0.01,      # heatmap focal 의 가중 (손실 크기 비율로 설정)
)

work_dir = 'work_dir/coxmamba/rgbtdroneperson/coxnet_tds'
