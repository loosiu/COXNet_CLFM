import copy
import unittest

import torch
from mmcv import Config

from mmdet.models import build_detector
from mmdet.models.detectors.base import BaseDetector
from mmdet.models.utils import DWTDFCA


BASELINE_CONFIG = 'configs/coxnet/coxnet_r50_fpn_1x_rgbtdroneperson.py'
DFCA_CONFIG = 'configs/coxnet/dwt_dfca/DWT_DFCA.py'


class TestDWTDFCAConfig(unittest.TestCase):

    def setUp(self):
        self.baseline = Config.fromfile(BASELINE_CONFIG)
        self.experiment = Config.fromfile(DFCA_CONFIG)

    def test_experiment_changes_only_clfm_selector_and_work_dir(self):
        expected = copy.deepcopy(self.baseline._cfg_dict.to_dict())
        expected['model']['use_clfm'] = ['dwt_dfca']
        expected['work_dir'] = (
            'work_dir/coxnet/rgbtdroneperson/dwt_dfca_oneway')

        self.assertEqual(self.experiment._cfg_dict.to_dict(), expected)
        self.assertEqual(self.experiment.model.neck.start_level, 2)
        self.assertEqual(self.experiment.model.neck_t.start_level, 1)
        self.assertTrue(self.experiment.model.wf_loss)
        self.assertEqual(self.experiment.model.wf_loss_mode, 'kl_v2')
        self.assertEqual(
            self.experiment.model.bbox_head.anchor_generator.strides,
            [8, 16, 32, 64])
        self.assertEqual(self.experiment.model.train_cfg.assigner.type,
                         'QLSAssigner')
        self.assertEqual(self.experiment.model.test_cfg.nms.iou_threshold, 0.3)

    def test_experiment_builds_four_dfca_levels(self):
        model = build_detector(
            self.experiment.model,
            train_cfg=self.experiment.get('train_cfg'),
            test_cfg=self.experiment.get('test_cfg'))

        self.assertEqual(len(model.fuse_layer.dwt_dfca_layers), 4)
        self.assertTrue(
            all(isinstance(layer, DWTDFCA)
                for layer in model.fuse_layer.dwt_dfca_layers))

    def test_forward_train_exposes_finite_dfca_diagnostics(self):
        torch.manual_seed(3)
        model = build_detector(
            self.experiment.model,
            train_cfg=self.experiment.get('train_cfg'),
            test_cfg=self.experiment.get('test_cfg'))
        model.train()
        rgb = torch.randn(2, 3, 128, 256)
        thermal = torch.randn(2, 3, 128, 256)
        img_metas = [
            dict(
                img_shape=(128, 256, 3),
                pad_shape=(128, 256, 3),
                scale_factor=1.0,
                flip=False) for _ in range(2)
        ]
        gt_bboxes = [
            torch.tensor([[40.0, 36.0, 58.0, 76.0]]) for _ in range(2)
        ]
        gt_labels = [torch.tensor([0], dtype=torch.long) for _ in range(2)]

        outputs = model.forward_train(
            [rgb, thermal], img_metas, gt_bboxes, gt_labels)

        diagnostic_names = {
            'dfca_w_low', 'dfca_w_mid', 'dfca_w_high',
            'dfca_gate_mean', 'dfca_delta_ratio'
        }
        self.assertTrue(diagnostic_names.issubset(outputs))
        self.assertTrue(
            all(torch.isfinite(outputs[name]).all()
                for name in diagnostic_names))

    def test_parse_losses_logs_diagnostics_without_optimizing_them(self):
        losses = dict(
            loss_cls=torch.tensor(2.0, requires_grad=True),
            dfca_w_low=torch.tensor(0.2),
            dfca_w_mid=torch.tensor(0.3),
            dfca_w_high=torch.tensor(0.5),
            dfca_gate_mean=torch.tensor(0.6),
            dfca_delta_ratio=torch.tensor(0.01))

        total, log_vars = BaseDetector._parse_losses(object(), losses)

        torch.testing.assert_close(total, torch.tensor(2.0))
        self.assertEqual(log_vars['loss'], 2.0)
        self.assertAlmostEqual(log_vars['dfca_delta_ratio'], 0.01, places=7)


if __name__ == '__main__':
    unittest.main()
