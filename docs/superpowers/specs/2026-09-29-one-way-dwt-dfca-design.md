# One-Way DWT-DFCA for COXNet Design

## Objective

Replace only COXNet's CLFM computation with a one-way, DWT-adapted version of DyFCLT's Dynamic Frequency-Band Decoupled Cross-Modal Attention (DFCA). Preserve the official COXNet cross-stage FPN pairing, RGB deconvolution, Thermal stream, HOFM/AAM, `wf_loss`, GFL head, QLS assignment, and evaluation settings so that the experiment isolates the CLFM replacement.

The first experiment enhances RGB only. A bidirectional version is explicitly deferred until the one-way result and diagnostics justify expanding the method.

## Research Boundary

This is a DWT-domain adaptation of DFCA, not a reproduction of the complete DyFCLT architecture.

- Included: full Q/K/V decomposition, band-wise cross-modal correlation, content-adaptive band weighting, RGB residual calibration.
- Excluded: FFT, learnable radial FFT boundaries, DyFCLT SSE/IBS/FREF, bidirectional enhancement, new foreground-mask loss, new detector head, and same-stage FPN changes.
- Baseline invariants: RGB FPN `start_level=2`, Thermal FPN `start_level=1`, four cross-stage pairs, RGB x2 DeConv, and existing `wf_loss=True` with `kl_v2`.

## Data Flow

For each of the four cross-stage pyramid pairs, let the higher-resolution Thermal feature be `T` and the lower-resolution RGB feature be `R`.

1. Upsample RGB using the original learned x2 transposed convolution:

   `R_up = DeConv(R)`

2. Construct DFCA projections using a 1x1 point-wise convolution followed by a 3x3 depthwise convolution:

   `Q = proj_q(T)`

   `K = proj_k(R_up)`

   `V = proj_v(R_up)`

   This follows the target-branch convention in DyFCLT: the opposite modality supplies the query, while key and value come from the modality being enhanced. Therefore Thermal conditions the selection of RGB information without replacing RGB with Thermal values.

3. Apply one-level Haar DWT to all Q, K, and V features:

   `DWT(X) = {X_LL, X_LH, X_HL, X_HH}`

   The DWT groups are:

   - low: `LL`
   - mid: `LH`, `HL`, preserving both orientations separately
   - high: `HH`

4. For each sub-band, compute channel-normalized local cross-modal correlation:

   `C_b = sum_c(norm(Q_b) * norm(K_b))`

   Convert it to a spatial gate with a 3x3 convolution and sigmoid:

   `G_b = sigmoid(conv_b(C_b))`

   Low and high groups have independent gate convolutions. `LH` and `HL` share the mid-frequency gate convolution but are evaluated separately, preserving directional coefficients.

5. Build per-image band statistics from the mean absolute correlation of low, mid, and high groups. A learnable 3-to-3 router followed by softmax produces:

   `A = softmax(router([s_low, s_mid, s_high]))`

   The router is initialized to equal weights. This content-adaptive weighting is the DWT-compatible replacement for DyFCLT's learnable radial FFT boundaries; it must not be described as learning DWT boundaries.

6. Modulate RGB value coefficients:

   `D_b = A_group(b) * G_b * V_b`

   Reconstruct a spatial residual with IDWT and a small output projection:

   `Delta = tanh(out_proj(IDWT(D_LL, D_LH, D_HL, D_HH)))`

7. Preserve the RGB identity path:

   `R_out = R_up + 0.1 * Delta`

   The output projection uses a small non-zero initialization so detection gradients reach Q, K, V, gates, and the band router from the first update. Thermal is returned unchanged.

8. Pass `(R_out, T)` to the existing HOFM/AAM path.

## Components and Interfaces

### `DWTDFCA`

Location: `mmdet/models/utils/dwt_dfca.py`

Public interface:

```python
class DWTDFCA(nn.Module):
    def __init__(self, channels: int, residual_scale: float = 0.1,
                 eps: float = 1e-6): ...

    def forward(self, thermal: torch.Tensor,
                visible: torch.Tensor) -> torch.Tensor: ...

    def get_diagnostics(self) -> dict[str, torch.Tensor]: ...
```

`visible` is the lower-resolution cross-stage feature. The module owns the baseline-compatible x2 DeConv and returns enhanced RGB at Thermal resolution. It rejects channel or post-DeConv spatial mismatches instead of silently interpolating them.

### `FusionLayer` integration

- Preserve the existing `use_clfm=['v3']` path unchanged.
- Add a separate `use_clfm=['dwt_dfca']` path.
- Construct one `DWTDFCA` instance per pyramid level.
- Call `visible = dwt_dfca_layers[i](thermal, visible)` before the existing HOFM layer.
- Expose averaged, detached diagnostics from all four levels.

### Detector diagnostics

During training, `FusionNetXO.forward_train` adds detached diagnostic tensors to the returned dictionary using names that do not contain `loss`, so MMDetection logs them but excludes them from the optimized loss sum:

- `dfca_w_low`
- `dfca_w_mid`
- `dfca_w_high`
- `dfca_gate_mean`
- `dfca_delta_ratio`

`delta_ratio` is `mean(abs(0.1 * Delta)) / (mean(abs(R_up)) + eps)`.

### Configuration

Create `configs/coxnet/dwt_dfca/DWT_DFCA.py` by inheriting the official RGBTDronePerson COXNet configuration and overriding only:

- `model.use_clfm=['dwt_dfca']`
- `work_dir`

The official baseline config remains unchanged.

## Initialization and Numerical Safety

- Q/K/V point-wise and depthwise convolutions use standard Kaiming initialization.
- Band-router weights and bias initialize to zero, producing exactly one-third weights before learning.
- Gate convolution weights use a small non-zero normal initialization (`std=1e-3`) with zero bias, producing a gate close to 0.5 while allowing Q/K gradients from the first update.
- Output projection uses normal initialization with standard deviation `1e-3`, not zero initialization.
- Correlations use channel L2 normalization with `eps=1e-6`.
- The module requires finite floating-point inputs and validates tensor rank, batch, channel, and spatial compatibility.
- Official FPN feature sizes are even at the DWT input. Tests also pin the intended failure for unsupported mismatched sizes rather than adding hidden interpolation.

## Verification Contract

Focused CPU tests must prove:

1. Cross-stage inputs produce an RGB output at Thermal resolution.
2. Thermal input is not modified in place.
3. Initial band weights are finite, positive, and sum to one.
4. Q, K, V, gate, router, output projection, DeConv, and both input tensors receive finite non-zero gradients.
5. A mismatched post-DeConv shape raises a clear error.
6. `FusionLayer` uses four DWT-DFCA instances and produces the four expected pyramid shapes.
7. The new config builds the official detector while retaining FPN start levels, `wf_loss`, HOFM/AAM, GFL, and QLS settings.
8. The original `v3` baseline configuration remains byte-for-byte unchanged.

The upstream full pytest collection is not a release gate for this fork because the clean official baseline already fails collection due to absent `pytorch_wavelets` in the test environment and unrelated removed upstream modules/dependencies. Those baseline failures must remain documented. Focused tests run in an environment containing the project dependencies, followed by `compileall`, config/model construction, and a real GPU forward/backward smoke test in `coxmamba`.

## Training and Comparison

After verification and GitHub push:

1. Stop the existing GPU 1 TPSC job only after the DWT-DFCA GPU smoke test passes.
2. Run DWT-DFCA seeds 0, 1, and 2 sequentially on physical GPU 1.
3. Use the same dataset root, optimizer, schedule, augmentation, batch size, deterministic flag, and evaluation settings as the official baseline.
4. Preserve separate work directories and console logs for every seed.
5. Compare each seed and the three-seed mean with the corresponding official COXNet baseline values; do not infer improvement from training loss or a partial run.
6. Monitor `dfca_w_*`, `dfca_gate_mean`, and `dfca_delta_ratio`. Treat saturated band weights, constant gates, or near-zero residuals as module-activation failures even if training itself completes.

## Git Integration

- Development branch: `dwt-dfca-oneway`
- Official upstream remote remains read-only and is never used as the push target.
- Planned research push target: `loosiu/COXNet_CLFM`, matching the existing user-owned COXNet CLFM remote in the local workspace.
- Push the feature branch, not the upstream `master` branch.

## Success Criteria

Implementation is ready to train only when all focused tests pass, the detector config builds, the GPU smoke test completes forward and backward, diagnostics are finite and active, and the git diff contains no unrelated model or dataset changes.

Research success is determined only after all three seeds finish. The first decision compares DWT-DFCA against the official COXNet baseline under identical evaluation. A bidirectional extension is justified only if the one-way module is active and either improves the target metrics or reveals a specific one-way limitation worth testing.
