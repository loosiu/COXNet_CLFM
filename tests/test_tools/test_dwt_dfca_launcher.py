import os
from pathlib import Path
import subprocess
import sys
import unittest


REPO_ROOT = Path(__file__).resolve().parents[2]


class TestDWTDFCALauncher(unittest.TestCase):

    def test_train_module_preserves_external_cuda_visible_devices(self):
        probe = """
import importlib.util
import os
spec = importlib.util.spec_from_file_location('coxnet_train', 'tools/train.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
print(os.environ['CUDA_VISIBLE_DEVICES'])
"""
        environment = os.environ.copy()
        environment['CUDA_VISIBLE_DEVICES'] = '1'

        result = subprocess.run(
            [sys.executable, '-c', probe],
            cwd=REPO_ROOT,
            env=environment,
            text=True,
            capture_output=True,
            check=True)

        self.assertEqual(result.stdout.strip().splitlines()[-1], '1')

    def test_seed_runner_dry_run_lists_three_ordered_isolated_runs(self):
        environment = os.environ.copy()
        environment['DWT_DFCA_DRY_RUN'] = '1'

        result = subprocess.run(
            ['bash', 'tools/run_dwt_dfca_seeds_gpu1.sh'],
            cwd=REPO_ROOT,
            env=environment,
            text=True,
            capture_output=True,
            check=True)
        run_lines = [
            line for line in result.stdout.splitlines()
            if line.startswith('RUN seed=')
        ]

        self.assertEqual(len(run_lines), 3)
        self.assertEqual(
            [line.split()[1] for line in run_lines],
            ['seed=0', 'seed=1', 'seed=2'])
        for seed, line in enumerate(run_lines):
            self.assertIn(
                f'work_dir/coxmamba/rgbtdroneperson/dwt_dfca/seed{seed}',
                line)
            self.assertIn('--deterministic', line)


if __name__ == '__main__':
    unittest.main()
