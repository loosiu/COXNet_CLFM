# One-Way DWT-DFCA Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the official COXNet CLFM computation with a one-way, DWT-domain DFCA module while preserving cross-stage FPN pairing, RGB DeConv, Thermal features, HOFM/AAM, `wf_loss`, and detector settings.

**Architecture:** Each pyramid level upsamples RGB with the original x2 learned DeConv, uses Thermal as Query and RGB as Key/Value, decomposes all Q/K/V tensors with one-level Haar DWT, and applies local cross-modal gates plus dynamic low/mid/high band weights to RGB Value coefficients. IDWT reconstructs a bounded residual that enhances RGB only before the unchanged HOFM/AAM path.

**Tech Stack:** Python 3.9, PyTorch 1.13.1, MMCV 1.7.1, MMDetection 2.28.2 fork, device-agnostic orthonormal Haar DWT/IDWT, stdlib `unittest`, Bash/tmux/flock.

**Spec:** `docs/superpowers/specs/2026-09-29-one-way-dwt-dfca-design.md`

## Global Constraints

- Base all production changes on official commit `991ee55f8419a56d3c15fc1ee4a838f4fcb56c2c` plus the approved design commit.
- Keep RGB FPN `start_level=2`, Thermal FPN `start_level=1`, four cross-stage pairs, and learned RGB x2 DeConv.
- Keep Thermal unchanged and enhance RGB only.
- Keep HOFM/AAM, `wf_loss=True` with `kl_v2`, GFL, QLS assignment, NMS, dataset, optimizer, schedule, augmentation, and batch size unchanged.
- Do not add FFT, SSE/IBS/FREF, bidirectional enhancement, same-stage FPN, or an auxiliary supervision loss.
- Keep the official `configs/coxnet/coxnet_r50_fpn_1x_rgbtdroneperson.py` file unchanged.
- Use physical GPU 0 for the preemption-safe smoke test; use physical GPU 1 only for sequential seed 0, 1, and 2 training.
- Do not stop the existing GPU 1 TPSC job until all focused tests and the GPU 0 smoke test pass.
- Push branch `dwt-dfca-oneway` to the user-owned `loosiu/COXNet_CLFM` remote; never push to Troy-peng-0327 upstream.

## Review Focus

- Post-DeConv size differs from Thermal size: Task 1 must raise a clear `ValueError`; it must not silently interpolate.
- All-zero RGB/Thermal tensors: Task 2 must produce finite output and diagnostics without NaN/Inf.
- Small non-zero initialization: Task 2 must prove non-zero finite gradients reach Q, K, V, gates, router, output projection, DeConv, and both inputs on the first backward pass.
- Non-loss diagnostics in detector output: Task 4 must prove `_parse_losses` logs them but excludes them from the optimized loss sum.
- Baseline drift: Task 4 must prove inherited FPN, `wf_loss`, head, assignment, and test settings remain equal to the official config, and `git diff` must show no edit to the official baseline config.

---

### Task 1: Cross-Stage DWT-DFCA Tensor Contract

**Files:**
- Create: `mmdet/models/utils/dwt_dfca.py`
- Modify: `mmdet/models/utils/__init__.py`
- Create: `tests/test_models/test_utils/test_dwt_dfca.py`

**Interfaces:**
- Consumes: `torch.Tensor` Thermal `[N,C,2H,2W]` and RGB `[N,C,H,W]` inputs.
- Produces: `DWTDFCA.forward(thermal, visible) -> torch.Tensor` at `[N,C,2H,2W]` and `get_diagnostics() -> dict[str, Tensor]`.

- [ ] **Step 1: Write failing shape and validation tests**

Add `unittest.TestCase` cases that independently assert:

```python
def test_cross_stage_rgb_is_upsampled_to_thermal_shape(self):
    module = DWTDFCA(channels=8)
    thermal = torch.randn(2, 8, 16, 20)
    visible = torch.randn(2, 8, 8, 10)
    self.assertEqual(module(thermal, visible).shape, thermal.shape)

def test_post_deconv_shape_mismatch_is_rejected(self):
    module = DWTDFCA(channels=8)
    with self.assertRaisesRegex(ValueError, "post-DeConv"):
        module(torch.randn(1, 8, 15, 20), torch.randn(1, 8, 8, 10))
```

Also assert rank, batch, and channel mismatches raise descriptive `ValueError`s, Thermal is unchanged after forward, and a hand-sized even tensor survives the public Haar DWT/IDWT round trip within `1e-6` while preserving dtype and device.

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
/home/viplab/anaconda3/envs/coxmamba/bin/python tests/test_models/test_utils/test_dwt_dfca.py -v
```

Expected: import failure because `mmdet.models.utils.dwt_dfca` does not exist.

- [ ] **Step 3: Implement the minimal tensor contract**

Create:

```python
class DWTDFCA(nn.Module):
    def __init__(self, channels: int, residual_scale: float = 0.1,
                 eps: float = 1e-6): ...
    def forward(self, thermal: torch.Tensor,
                visible: torch.Tensor) -> torch.Tensor: ...
    def get_diagnostics(self) -> Dict[str, torch.Tensor]: ...
```

Use the existing `TransBasicConv2d(channels, channels)` behavior for x2 RGB DeConv, validate inputs and post-DeConv shape, and initially return the upsampled RGB. Implement device-agnostic orthonormal one-level Haar DWT/IDWT helpers from 2x2 even/odd samples; do not call the legacy `DWT_2D.forward()` because it unconditionally moves inputs to CUDA. Export `DWTDFCA` from `mmdet.models.utils`.

- [ ] **Step 4: Run the focused tests and verify GREEN**

Run the Step 2 command. Expected: all Task 1 cases pass.

- [ ] **Step 5: Commit**

```bash
git add mmdet/models/utils/dwt_dfca.py mmdet/models/utils/__init__.py tests/test_models/test_utils/test_dwt_dfca.py
git commit -m "feat: add DWT-DFCA cross-stage tensor contract"
```

### Task 2: Band-Wise DWT Attention, Residual, and Diagnostics

**Files:**
- Modify: `mmdet/models/utils/dwt_dfca.py`
- Modify: `tests/test_models/test_utils/test_dwt_dfca.py`

**Interfaces:**
- Consumes: validated cross-stage tensors from Task 1.
- Produces: RGB residual calibration and detached scalar diagnostics `w_low`, `w_mid`, `w_high`, `gate_mean`, and `delta_ratio`.

- [ ] **Step 1: Add failing behavior tests**

Add real-code tests:

```python
def test_initial_band_weights_are_equal_positive_and_sum_to_one(self): ...
def test_zero_features_produce_finite_output_and_diagnostics(self): ...
def test_first_backward_reaches_every_dfca_path_and_both_inputs(self): ...
```

Use literal initial weights `1/3`. For the gradient test, use fixed-seed non-zero tensors with `requires_grad=True`, optimize `output.square().mean()`, and assert finite non-zero gradients for `q_proj`, `k_proj`, `v_proj`, gate convolution, router, output projection, DeConv, Thermal, and RGB. Do not assert implementation-private tensor values beyond the public diagnostics.

- [ ] **Step 2: Run and verify RED**

Run the focused file. Expected: diagnostics are absent and the DFCA parameter names/gradients do not exist.

- [ ] **Step 3: Implement DWT-DFCA**

Implement:

- 1x1 point-wise plus 3x3 depthwise Q/K/V projections.
- The Task 1 device-agnostic one-level Haar DWT/IDWT on every Q/K/V tensor.
- Channel-normalized sub-band correlation with `eps`.
- Independent low/high 3x3 gate convolutions and a shared mid gate for separate `LH`/`HL` coefficients.
- Per-image low/mid/high absolute-correlation statistics and a zero-initialized `nn.Linear(3, 3)` softmax router.
- Band modulation, IDWT, `tanh` output projection with `std=1e-3`, and fixed residual scale `0.1`.
- Small non-zero gate weights (`std=1e-3`) and zero gate bias.
- Detached, finite diagnostics updated on every forward.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run the focused file. Expected: all Task 1 and Task 2 cases pass.

- [ ] **Step 5: Commit**

```bash
git add mmdet/models/utils/dwt_dfca.py tests/test_models/test_utils/test_dwt_dfca.py
git commit -m "feat: implement one-way band-wise DWT-DFCA"
```

### Task 3: FusionLayer Integration

**Files:**
- Modify: `mmdet/models/utils/fusion_strategy.py`
- Create: `tests/test_models/test_utils/test_dwt_dfca_fusion.py`

**Interfaces:**
- Consumes: `use_clfm=['dwt_dfca']` and four RGB/Thermal FPN tuples.
- Produces: four unchanged HOFM outputs and averaged detached diagnostics via `FusionLayer.get_dwt_dfca_diagnostics()`.

- [ ] **Step 1: Write failing integration tests**

Instantiate a real `FusionLayer` with `in_channels=8`, `num_layers=4`, `fs_type='fusionnet-xo'`, `use_clfm=['dwt_dfca']`, lightweight HOFM options, and paired RGB/Thermal shapes `(8,10)->(16,20)`, `(4,5)->(8,10)`, `(2,3)->(4,6)`, `(1,2)->(2,4)`. Assert four outputs at Thermal spatial shapes and finite averaged diagnostics with three weights summing to one.

- [ ] **Step 2: Run and verify RED**

Run:

```bash
/home/viplab/anaconda3/envs/coxmamba/bin/python tests/test_models/test_utils/test_dwt_dfca_fusion.py -v
```

Expected: `FusionLayer` does not recognize or construct DWT-DFCA.

- [ ] **Step 3: Add the isolated integration path**

Import `DWTDFCA`, construct `self.dwt_dfca_layers`, dispatch the exact `dwt_dfca` token before legacy CLFM substring branches, and average diagnostics across the four levels. Do not alter `v3`, `t3`, interpolation, HOFM, or `wf_loss` behavior.

- [ ] **Step 4: Run Task 1-3 tests and verify GREEN**

Run both DWT-DFCA test files. Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add mmdet/models/utils/fusion_strategy.py tests/test_models/test_utils/test_dwt_dfca_fusion.py
git commit -m "feat: integrate DWT-DFCA into COXNet fusion"
```

### Task 4: Detector Diagnostics and Controlled Configuration

**Files:**
- Modify: `mmdet/models/detectors/fusionnet_xo.py`
- Create: `configs/coxnet/dwt_dfca/DWT_DFCA.py`
- Create: `tests/test_models/test_detectors/test_dwt_dfca_config.py`

**Interfaces:**
- Consumes: FusionLayer diagnostics and official baseline config inheritance.
- Produces: buildable detector config and training log metrics excluded from the optimized loss.

- [ ] **Step 1: Write failing config and loss-parsing tests**

Assert with `mmcv.Config` that the new config changes only `model.use_clfm` and `work_dir` relative to the official config, including exact preservation of both FPN start levels, `wf_loss`, head strides, QLS assigner, and test NMS settings. Build the detector and assert four `DWTDFCA` levels exist. Use `BaseDetector._parse_losses` on a dictionary containing a true loss and detached DFCA diagnostics; assert the scalar optimized loss equals only the true loss while all diagnostics appear in log variables.

- [ ] **Step 2: Run and verify RED**

Run:

```bash
/home/viplab/anaconda3/envs/coxmamba/bin/python tests/test_models/test_detectors/test_dwt_dfca_config.py -v
```

Expected: config missing and detector diagnostics not exposed.

- [ ] **Step 3: Add config and training diagnostics**

Create an inherited config overriding only `model.use_clfm=['dwt_dfca']` and its `work_dir`. In `FusionNetXO.forward_train`, append detached FusionLayer diagnostics to the returned loss dictionary after detection losses, using keys without the substring `loss`.

- [ ] **Step 4: Verify Task 1-4 tests and baseline config immutability**

Run all three focused files, then:

```bash
git diff 991ee55 -- configs/coxnet/coxnet_r50_fpn_1x_rgbtdroneperson.py
```

Expected: focused tests pass and the baseline config diff is empty.

- [ ] **Step 5: Commit**

```bash
git add mmdet/models/detectors/fusionnet_xo.py configs/coxnet/dwt_dfca/DWT_DFCA.py tests/test_models/test_detectors/test_dwt_dfca_config.py
git commit -m "feat: add controlled DWT-DFCA experiment config"
```

### Task 5: GPU-Selectable Launcher and Sequential Seed Runner

**Files:**
- Modify: `tools/train.py`
- Create: `tools/run_dwt_dfca_seeds_gpu1.sh`
- Create: `tests/test_tools/test_dwt_dfca_launcher.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: external `CUDA_VISIBLE_DEVICES`, `MMDET_DATASETS`, seed list `0 1 2`, and the DWT-DFCA config.
- Produces: reproducible sequential work directories and logs without overriding the selected physical GPU.

- [ ] **Step 1: Write failing launcher behavior tests**

Run `tools/train.py` as an imported module in a subprocess with `CUDA_VISIBLE_DEVICES=1` and assert the value remains `1`; this catches the current hard-coded override. Run the seed script with `DWT_DFCA_DRY_RUN=1` and assert its output contains exactly one ordered command for each of seeds 0, 1, and 2 with separate work directories.

- [ ] **Step 2: Run and verify RED**

Run:

```bash
/home/viplab/anaconda3/envs/coxmamba/bin/python tests/test_tools/test_dwt_dfca_launcher.py -v
```

Expected: the train launcher changes the GPU to `5` and the seed script does not exist.

- [ ] **Step 3: Implement launcher behavior**

Change the single hard-coded assignment in `tools/train.py` to `os.environ.setdefault('CUDA_VISIBLE_DEVICES', '5')`. Add an executable seed runner that:

- acquires `/tmp/coxnet_dwt_dfca_gpu1.lock` using `flock`;
- exports physical GPU 1, the existing RGBTDronePerson data root, and repository `PYTHONPATH`;
- runs seeds 0, 1, and 2 sequentially with `--deterministic`;
- writes `work_dir/coxmamba/rgbtdroneperson/dwt_dfca/seed{N}/console.log`;
- stops on the first failed seed;
- supports a no-side-effect dry-run mode for the behavior test.

Document the config, diagnostics, smoke command, and seed runner in README.

- [ ] **Step 4: Run launcher and all focused tests**

Expected: launcher tests and all prior focused tests pass.

- [ ] **Step 5: Commit**

```bash
git add tools/train.py tools/run_dwt_dfca_seeds_gpu1.sh tests/test_tools/test_dwt_dfca_launcher.py README.md
git commit -m "chore: add reproducible DWT-DFCA seed launcher"
```

### Task 6: Fresh Verification, GPU Smoke, Push, and Training Handoff

**Files:**
- Verify all modified files.
- Runtime outputs: `work_dir/coxmamba/rgbtdroneperson/dwt_dfca/`

**Interfaces:**
- Consumes: completed branch and existing RGBTDronePerson dataset.
- Produces: verified pushed branch and active sequential GPU 1 training session.

- [ ] **Step 1: Run fresh focused verification**

Run all DWT-DFCA unit/integration/launcher tests, `python -m compileall` on modified Python files, config parsing, model construction, `git diff --check`, and `git status --short`. Record the known full-suite baseline collection failures separately; do not call them DWT-DFCA regressions.

- [ ] **Step 2: Run a real GPU 0 forward/backward smoke test**

Using `coxmamba`, physical GPU 0, the DWT-DFCA config, and one real RGBTDronePerson training batch, run model initialization, forward loss computation, backward, and assert:

- all detection losses and `wf_loss` are finite;
- all DFCA diagnostics are finite;
- low/mid/high weights sum to one;
- `dfca_delta_ratio > 0`;
- DFCA Q/K/V, gate, router, output projection, and DeConv gradients are finite and non-zero.

- [ ] **Step 3: Review the complete diff against the spec**

Confirm no same-stage FPN, bidirectional, SSE, new auxiliary loss, dataset, head, assignment, or evaluation changes entered the branch. Commit any verification-only corrections through their own RED/GREEN cycle.

- [ ] **Step 4: Configure and push the user-owned remote**

Add or update remote `mine` to `git@github.com:loosiu/COXNet_CLFM.git`, verify the outgoing commits, and push `dwt-dfca-oneway`. Do not push upstream `master`.

- [ ] **Step 5: Stop the old GPU 1 job and launch DWT-DFCA seeds**

After the push succeeds, terminate tmux session `tpsc_gpu1`, verify its training PIDs are gone and GPU 1 memory is released, then start `tools/run_dwt_dfca_seeds_gpu1.sh` in tmux session `dwt_dfca_gpu1`.

- [ ] **Step 6: Verify live training rather than only command submission**

Confirm the new process is bound to physical GPU 1, inspect the resolved config in the seed-0 log, and wait for the first logged optimizer iteration with finite losses and active DFCA diagnostics. Report seed 1 and 2 as queued, not running or complete.

- [ ] **Step 7: Final status report**

Report branch/commit/push target, exact tests and smoke evidence, tmux/PID/GPU/log/work directories, known upstream test-suite limitations, and the fact that AP results remain pending until validation completes.
