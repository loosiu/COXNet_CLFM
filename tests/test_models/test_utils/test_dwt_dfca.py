import unittest

import torch

from mmdet.models.utils.dwt_dfca import DWTDFCA


class TestDWTDFCATensorContract(unittest.TestCase):

    def assert_parameter_group_has_finite_nonzero_gradient(self, module,
                                                           prefix):
        gradients = [
            parameter.grad for name, parameter in module.named_parameters()
            if name.startswith(prefix)
        ]
        self.assertTrue(gradients, f'no parameters found for {prefix}')
        self.assertTrue(
            all(gradient is not None for gradient in gradients),
            f'missing gradient in {prefix}')
        self.assertTrue(
            all(torch.isfinite(gradient).all() for gradient in gradients),
            f'non-finite gradient in {prefix}')
        self.assertGreater(
            sum(gradient.abs().sum().item() for gradient in gradients), 0.0,
            f'zero gradient in {prefix}')

    def test_cross_stage_rgb_is_upsampled_to_thermal_shape(self):
        torch.manual_seed(0)
        module = DWTDFCA(channels=8)
        thermal = torch.randn(2, 8, 16, 20)
        visible = torch.randn(2, 8, 8, 10)

        output = module(thermal, visible)

        self.assertEqual(output.shape, thermal.shape)

    def test_forward_does_not_modify_thermal_in_place(self):
        module = DWTDFCA(channels=8)
        thermal = torch.randn(2, 8, 16, 20)
        thermal_before = thermal.clone()
        visible = torch.randn(2, 8, 8, 10)

        module(thermal, visible)

        torch.testing.assert_close(thermal, thermal_before)

    def test_post_deconv_shape_mismatch_is_rejected(self):
        module = DWTDFCA(channels=8)

        with self.assertRaisesRegex(ValueError, 'post-DeConv'):
            module(
                torch.randn(1, 8, 15, 20),
                torch.randn(1, 8, 8, 10))

    def test_rank_mismatch_is_rejected(self):
        module = DWTDFCA(channels=8)

        with self.assertRaisesRegex(ValueError, '4D'):
            module(torch.randn(8, 16, 20), torch.randn(1, 8, 8, 10))

    def test_batch_mismatch_is_rejected(self):
        module = DWTDFCA(channels=8)

        with self.assertRaisesRegex(ValueError, 'batch'):
            module(
                torch.randn(2, 8, 16, 20),
                torch.randn(1, 8, 8, 10))

    def test_channel_mismatch_is_rejected(self):
        module = DWTDFCA(channels=8)

        with self.assertRaisesRegex(ValueError, 'channel'):
            module(
                torch.randn(1, 7, 16, 20),
                torch.randn(1, 8, 8, 10))

    def test_haar_round_trip_preserves_values_dtype_and_device(self):
        source = torch.arange(64, dtype=torch.float64).reshape(1, 1, 8, 8)

        components = DWTDFCA.haar_dwt(source)
        reconstructed = DWTDFCA.haar_idwt(*components)

        torch.testing.assert_close(reconstructed, source, atol=1e-6, rtol=0)
        self.assertEqual(reconstructed.dtype, source.dtype)
        self.assertEqual(reconstructed.device, source.device)

    def test_initial_band_weights_are_equal_positive_and_sum_to_one(self):
        torch.manual_seed(1)
        module = DWTDFCA(channels=8)

        module(
            torch.randn(2, 8, 16, 20),
            torch.randn(2, 8, 8, 10))
        diagnostics = module.get_diagnostics()
        weights = torch.stack([
            diagnostics['w_low'], diagnostics['w_mid'],
            diagnostics['w_high']
        ])

        torch.testing.assert_close(
            weights,
            torch.full_like(weights, 1.0 / 3.0),
            atol=1e-6,
            rtol=0)
        self.assertTrue((weights > 0).all())
        torch.testing.assert_close(
            weights.sum(), weights.new_tensor(1.0), atol=1e-6, rtol=0)
        self.assertFalse(any(value.requires_grad
                             for value in diagnostics.values()))

    def test_zero_features_produce_finite_output_and_diagnostics(self):
        module = DWTDFCA(channels=8)

        output = module(
            torch.zeros(2, 8, 16, 20),
            torch.zeros(2, 8, 8, 10))
        diagnostics = module.get_diagnostics()

        self.assertTrue(torch.isfinite(output).all())
        self.assertEqual(
            set(diagnostics),
            {'w_low', 'w_mid', 'w_high', 'gate_mean', 'delta_ratio'})
        self.assertTrue(
            all(torch.isfinite(value).all()
                for value in diagnostics.values()))

    def test_first_backward_reaches_every_dfca_path_and_both_inputs(self):
        torch.manual_seed(2)
        module = DWTDFCA(channels=8)
        thermal = torch.randn(2, 8, 16, 20, requires_grad=True)
        visible = torch.randn(2, 8, 8, 10, requires_grad=True)

        output = module(thermal, visible)
        output.square().mean().backward()

        for prefix in ('q_proj', 'k_proj', 'v_proj', 'low_gate',
                       'mid_gate', 'high_gate', 'band_router', 'out_proj',
                       'deconv'):
            self.assert_parameter_group_has_finite_nonzero_gradient(
                module, prefix)
        for name, gradient in (('thermal', thermal.grad),
                               ('visible', visible.grad)):
            self.assertIsNotNone(gradient, f'{name} gradient is missing')
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(gradient.abs().sum().item(), 0.0)


if __name__ == '__main__':
    unittest.main()
