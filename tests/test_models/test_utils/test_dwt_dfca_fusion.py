import unittest

import torch

from mmdet.models.utils.fusion_strategy import FusionLayer


class TestDWTDFCAFusionIntegration(unittest.TestCase):

    def _build_layer(self):
        return FusionLayer(
            in_channels=8,
            reduction=4,
            num_layers=4,
            fs_type='fusionnet-xo',
            use_om=False,
            use_grid=False,
            use_msf=False,
            use_clfm=['dwt_dfca'],
            usepoolup=[])

    def _features(self):
        visible_shapes = ((8, 10), (4, 5), (2, 3), (1, 2))
        thermal_shapes = ((16, 20), (8, 10), (4, 6), (2, 4))
        visible = tuple(
            torch.randn(2, 8, height, width)
            for height, width in visible_shapes)
        thermal = tuple(
            torch.randn(2, 8, height, width)
            for height, width in thermal_shapes)
        return visible, thermal, thermal_shapes

    def test_four_cross_stage_levels_produce_thermal_resolution_outputs(self):
        layer = self._build_layer()
        visible, thermal, thermal_shapes = self._features()

        outputs = layer(visible, thermal)

        self.assertEqual(len(outputs), 4)
        self.assertEqual(
            [tuple(output.shape[-2:]) for output in outputs],
            list(thermal_shapes))

    def test_four_level_diagnostics_are_finite_and_normalized(self):
        layer = self._build_layer()
        visible, thermal, _ = self._features()

        layer(visible, thermal)
        diagnostics = layer.get_dwt_dfca_diagnostics()

        self.assertEqual(
            set(diagnostics),
            {'w_low', 'w_mid', 'w_high', 'gate_mean', 'delta_ratio'})
        self.assertTrue(
            all(torch.isfinite(value).all()
                for value in diagnostics.values()))
        weight_sum = (diagnostics['w_low'] + diagnostics['w_mid'] +
                      diagnostics['w_high'])
        torch.testing.assert_close(
            weight_sum, weight_sum.new_tensor(1.0), atol=1e-6, rtol=0)


if __name__ == '__main__':
    unittest.main()
