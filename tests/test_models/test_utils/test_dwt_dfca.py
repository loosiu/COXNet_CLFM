import unittest

import torch

from mmdet.models.utils.dwt_dfca import DWTDFCA


class TestDWTDFCATensorContract(unittest.TestCase):

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


if __name__ == '__main__':
    unittest.main()
