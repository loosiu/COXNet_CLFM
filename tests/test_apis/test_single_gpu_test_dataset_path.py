import unittest
from unittest import mock

import numpy as np

from mmdet.apis.test import single_gpu_test


class _Dataset:
    PALETTE = None

    def __len__(self):
        return 1


class _DataLoader:
    dataset = _Dataset()

    def __iter__(self):
        yield {}


class _Model:

    def eval(self):
        return self

    def __call__(self, return_loss, rescale, **data):
        return [[np.empty((0, 5), dtype=np.float32)]]


class TestSingleGpuTestDatasetPath(unittest.TestCase):

    def test_inference_does_not_open_a_hardcoded_ground_truth_file(self):
        with mock.patch(
                'mmdet.apis.test.load_ground_truth',
                side_effect=AssertionError('hardcoded GT path was accessed')):
            results = single_gpu_test(_Model(), _DataLoader())

        self.assertEqual(len(results), 1)


if __name__ == '__main__':
    unittest.main()
