"""TDS: Thermal Detail Selection — the thermal half of COXNet's fusion stage.

COXNet's CLFM enhances only the visible feature: it takes thermal's LL band as
an ingredient ("x_ll, _, _, _" in the champion path) and hands DASR the raw
thermal feature untouched.  So the fusion stage is asymmetric — visible gets a
wavelet-domain refinement, thermal gets none — and the bands it computes and
then discards are the ones aligned with the objects:

    discarded   t_LH 0.903   t_HL 0.910   t_HH 0.916   (object-vs-bg AUC, lvl 0)
    used        v_LH 0.361   v_HL 0.368                (below 0.5: avoids objects)

An oracle on the trained champion, gating thermal's detail bands by the
ground-truth boxes and reading mAP:

    every detail band scaled by 1.7                46.42   +0.44
    scaled only inside the boxes                   47.91   +1.93
    scaled only outside the boxes                  43.99   -1.99
    detail removed inside, kept outside            31.43  -14.55

Thermal detail is a tiny-object cue at object positions and clutter over
background; dropping all of it regardless of position throws away the first to
be rid of the second.  TDS gives the thermal branch its own refinement, sharing
the DWT that CLFM already computes:

    D  = f_d([LH, HL, HH])            what detail response is present
    R  = sigmoid(f_g([LL, D]))        is that response object-related
    g_b = s_b (1 + k_b (R - Rbar))    per band, per position
    T' = IDWT(LL, g_LH LH, g_HL HL, g_HH HH)

R is a Detail Relevance Map: LL is insensitive to fine variation but preserves
object extent and spatial context stably, so structure in LL corroborating a
response in D marks the detail as object-related, and a response with no
supporting structure marks it as clutter.  This is causal, not narrative — on
the trained model, spatially permuting LL alone collapses R (AUC 0.970 ->
0.879, object response halved) while permuting D barely moves it.

R is supervised by a Gaussian heatmap on the GT box centres with CenterNet's
penalty-reduced focal loss (see FusionLayer.compute_tdr_loss).  Supervision is
not optional: the detection loss reaches this gate with a gradient two to four
orders of magnitude below the neighbouring layers, and an unsupervised run
converged to a constant with standard deviation 0.0002.

Three properties are deliberate.  LL enters the IDWT untouched, and Haar's
orthonormality means nothing done to the detail bands can leak into it — the
thermal feature's structure and position survive exactly while only its local
contrast changes (band scaling by g is identically "block mean + g * (x - block
mean)" on each 2x2 block).  s = 1, k = 0 is the identity, so training starts at
the champion and the worst case is the champion.  And nothing is initialised to
zero except through a small-std normal: a zero anywhere in a multiplicative
chain silences every other factor's gradient.

What training and a deterministic knockout then established (RGBTDronePerson,
seed 0, best epoch):

    learned weights   s = [1.21, 1.32, 1.29]  k = [0.073, 0.101, 0.113]
                      s > 1 and k > 0 reproduced across four gate variants;
                      k ordered LH < HL < HH, matching the bands' measured
                      discriminability
    mAP50             baseline 45.98  ->  46.77
    knockout          resetting s = 1, k = 0 at inference on the SAME trained
                      checkpoint returns 45.71 (baseline level) and halves
                      tiny1 (24.12 -> 11.30) — the gain rides in this module's
                      forward computation, and the detector routes its
                      smallest-object evidence through the re-admitted detail
    where it lands    the change the module makes to the feature has box AUC
                      0.9808; object/background energy ratio 1.703 -> 1.902

Known limit: scenes whose LL contrast vanishes (objects on ground as warm as
they are) blind the guide — R goes quiet and the module degrades to identity.
"""
import torch
import torch.nn as nn


class ThermalDetailSelection(nn.Module):
    """Judge, per position, whether thermal detail is object-related, and
    weight the bands accordingly.  See the module docstring for the evidence
    behind each decision."""

    def __init__(self, channels, reduction=2, hidden=16):
        super().__init__()
        c = max(channels // reduction, 32)
        # One description of the detail response, merged across the three
        # orientations: a person's outline is split between LH and HL depending
        # on pose, so the orientations are combined before being judged.  The
        # BatchNorm removes the bands' magnitude imbalance (HH arrives ~28%
        # smaller than LH at level 0) before they meet LL.  A single-concat
        # gate (LL and the raw bands into one convolution) learns the same map
        # (cell-wise correlation 0.95) but scored 0.78 lower — the extra
        # nonlinear stage on the detail side is what the parameters buy.
        self.f_d = nn.Sequential(
            nn.Conv2d(channels * 3, c, 3, padding=1),
            nn.BatchNorm2d(c),
            nn.SiLU(inplace=True),
        )
        # [LL context ; detail description] -> one scalar per position.  3x3
        # kernels give the judgement a 48px neighbourhood — whether a response
        # is compact (a person) or extended (a building edge) needs neighbours.
        self.f_g = nn.Sequential(
            nn.Conv2d(channels + c, hidden, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, 1, 3, padding=1),
        )
        # Small-std, not zero: zeroing the last layer of a stack starves every
        # layer before it.  R starts uniform at 0.5, so with k = 0 the module
        # is exactly the identity.
        nn.init.normal_(self.f_g[-1].weight, std=1e-3)
        nn.init.zeros_(self.f_g[-1].bias)
        # s: how much of band b to use overall.  k: how much that use follows
        # the map.  Per band because the orientations differ in measured
        # selectivity (0.930 / 0.937 / 0.951).  Neither sits in the other's
        # gradient path, so both move from the first step.
        self.s = nn.Parameter(torch.ones(3))
        self.k = nn.Parameter(torch.zeros(3))
        self.target = None          # R, supervised per cell by the heatmap loss
        self._last_gate = None      # detached copy for diagnostics

    def forward(self, t_ll, t_bands):
        d = self.f_d(torch.cat(t_bands, dim=1))
        m = torch.sigmoid(self.f_g(torch.cat([t_ll, d], dim=1)))
        self.target = m
        self._last_gate = m.detach()
        # Centre per image: focal supervision under a ~30:1 imbalance
        # calibrates R around 0.33 on objects and 0.04 elsewhere, so without
        # the shift both ends collapse and k has no leverage.  (Dividing by the
        # spatial std as well raises the object/background weighting from 1.03
        # to 1.62 — and moved mAP 46.77 -> 46.53.  Contrast is not what
        # limits; the uniform re-admission s > 1 carries the effect.)
        mn = m - m.mean(dim=(2, 3), keepdim=True)
        return tuple(self.s[i] * (1.0 + self.k[i] * mn) * b
                     for i, b in enumerate(t_bands))
