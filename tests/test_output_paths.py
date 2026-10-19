"""Check run isolation, exported values and checkpoint destinations on CPU."""
import ast
import contextlib
import io
import os
from pathlib import Path
import random
import tempfile
import types
import unittest
from unittest.mock import patch

import run_gamed as entry
import output_paths

try:
    import numpy as np
    import pandas as pd
    import openpyxl
    import torch
except ImportError:
    torch = None

ROOT = Path(__file__).resolve().parents[1]


class OutputDirectoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.home = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_all_dataset_pairs_variants_and_modes(self):
        for train in ('yang', 'fakeddit'):
            for val in ('yang', 'fakeddit'):
                for model in ('gamed', 'gamed_without_veto'):
                    for validation in (False, True):
                        with self.subTest(train=train, val=val, model=model, validation=validation):
                            path = output_paths.create_run_directory(self.home, train, val, model, validation)
                            self.assertEqual(path.parent, self.home/f'train_{train}__val_{val}'/model)
                            self.assertTrue(path.name.startswith('validate_' if validation else 'train_'))
                            self.assertTrue(path.is_dir())

    def test_repeated_runs_preserve_existing_files(self):
        first = output_paths.create_run_directory(self.home, 'yang', 'yang', 'gamed')
        (first/'keep.txt').write_text('keep')
        second = output_paths.create_run_directory(self.home, 'yang', 'yang', 'gamed')
        self.assertNotEqual(first, second)
        self.assertEqual((first/'keep.txt').read_text(), 'keep')

    def test_absolute_and_relative_output_roots_with_spaces(self):
        previous = Path.cwd()
        try:
            os.chdir(self.home)
            for root in ('relative output', self.home/'absolute output'):
                path = output_paths.create_run_directory(root, 'yang', 'fakeddit', 'gamed')
                self.assertEqual(path.parents[2], Path(root).resolve())
        finally:
            os.chdir(previous)

    def test_epoch_directories_are_one_based_and_not_reused(self):
        run = output_paths.create_run_directory(self.home, 'yang', 'yang', 'gamed')
        for epoch in (1, 2, 1000):
            path = output_paths.create_epoch_directory(run, epoch)
            self.assertEqual(path.name, f'epoch_{epoch:03d}')
            (path/'keep.txt').write_text('keep')
            with self.assertRaises(FileExistsError):
                output_paths.create_epoch_directory(run, epoch)
            self.assertEqual((path/'keep.txt').read_text(), 'keep')
        for epoch in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                output_paths.create_epoch_directory(run, epoch)

    def test_run_identifier_is_generated_once_and_collision_fails(self):
        with patch.object(output_paths, 'uuid4', return_value=types.SimpleNamespace(hex='fixed')) as uuid, \
             patch.object(output_paths, 'datetime') as clock:
            clock.now.return_value.strftime.return_value = 'fixedtime'
            path = output_paths.create_run_directory(self.home, 'yang', 'yang', 'gamed')
            uuid.assert_called_once_with()
            (path/'keep.txt').write_text('keep')
            with self.assertRaises(FileExistsError):
                output_paths.create_run_directory(self.home, 'yang', 'yang', 'gamed')
            self.assertEqual((path/'keep.txt').read_text(), 'keep')

    def test_invalid_selectors_cannot_escape_output_root(self):
        for train, val, model in (('../x', 'yang', 'gamed'), ('yang', '../x', 'gamed'),
                                  ('yang', 'yang', '../x')):
            with self.assertRaises(ValueError):
                output_paths.create_run_directory(self.home, train, val, model)

    def test_python_random_state_is_unchanged(self):
        before = random.getstate()
        run = output_paths.create_run_directory(self.home, 'yang', 'yang', 'gamed')
        output_paths.create_epoch_directory(run, 1)
        self.assertEqual(random.getstate(), before)

    def test_main_creates_one_run_outside_epoch_loop(self):
        tree = ast.parse((ROOT/'run_gamed.py').read_text(encoding='utf-8'))
        main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
        calls = [n for n in ast.walk(main) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Name) and n.func.id == 'create_run_directory']
        self.assertEqual(len(calls), 1)
        owner = next(n for n in main.body if isinstance(n, ast.Assign) and n.value is calls[0])
        loop = next(n for n in main.body if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == 'epoch')
        self.assertLess(main.body.index(owner), main.body.index(loop))


@unittest.skipIf(torch is None, 'PyTorch, NumPy, pandas and openpyxl are needed for export tests')
class OutputTensorTests(unittest.TestCase):
    setUp = OutputDirectoryTests.setUp
    tearDown = OutputDirectoryTests.tearDown

    def test_numpy_and_torch_random_states_are_unchanged(self):
        numpy_before = np.random.get_state()
        torch_before = torch.get_rng_state().clone()
        run = output_paths.create_run_directory(self.home, 'fakeddit', 'yang', 'gamed_without_veto', True)
        output_paths.create_epoch_directory(run, 1)
        numpy_after = np.random.get_state()
        self.assertEqual(numpy_before[0], numpy_after[0])
        np.testing.assert_array_equal(numpy_before[1], numpy_after[1])
        self.assertEqual(numpy_before[2:], numpy_after[2:])
        self.assertTrue(torch.equal(torch_before, torch.get_rng_state()))

    def model_and_loader(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_parameter('weight', torch.nn.Parameter(torch.tensor([2., 3.])))
                self.register_buffer('counter', torch.tensor(7))

            def forward(self, **kwargs):
                count = len(kwargs['image'])
                base = torch.arange(count).reshape(-1, 1).float()
                logits = [base * .7 - .3 + i * .1 for i in range(5)]
                features = tuple(torch.arange(count*64).reshape(count, 64).float()+i*1000 for i in range(5))
                return (*logits, torch.tensor(.3), features)

        loader = []
        for count in (2, 1):
            texts = tuple(torch.zeros(count, 2, dtype=torch.long) for _ in range(3))
            others = (torch.zeros(count, 3, 2, 2), torch.zeros(count, 3, 2, 2),
                      torch.arange(count) % 2, torch.zeros(count), ['sample '+str(i) for i in range(count)])
            loader.append((texts, others, ['unused']*count, None, None))
        return Model(), loader

    def test_actual_evaluation_files_keep_feature_roles_and_records(self):
        run = output_paths.create_run_directory(self.home, 'yang', 'fakeddit', 'gamed', True)
        directory = output_paths.create_epoch_directory(run, 1)
        model, loader = self.model_and_loader()
        with patch.object(entry, 'torch', torch, create=True), patch.object(entry, 'np', np, create=True), \
             patch.object(entry, 'to_var', lambda x: x), \
             patch.object(torch.Tensor, 'cuda', lambda self, *a, **k: self), \
             torch.no_grad(), contextlib.redirect_stdout(io.StringIO()):
            metrics = entry.evaluate(loader, model, torch.nn.BCEWithLogitsLoss(),
                                     setting={'thresh': .5, 'is_use_bce': True, 'val_dataname': 'fakeddit'},
                                     output_directory=directory)
        files = ('mix_features.txt', 'is_features.txt', 't_features.txt', 'ip_features.txt', 'mm_aux_features.txt')
        for i, name in enumerate(files):
            actual = np.loadtxt(directory/name)
            expected_features = np.concatenate((np.arange(128).reshape(2, 64), np.arange(64).reshape(1, 64)))+i*1000
            np.testing.assert_array_equal(actual[:, 0], [0, 1, 0])
            np.testing.assert_array_equal(actual[:, 1:], expected_features)
            self.assertEqual(actual.shape, (3, 65))
        frame = pd.read_excel(directory/'predictions.xlsx')
        self.assertEqual(len(frame), 2)  # The existing singleton-batch evaluation behaviour.
        self.assertEqual(list(frame['text']), ['sample 0', 'sample 1'])
        self.assertIn('mm_feature', frame.columns)
        self.assertEqual(len(metrics[0]), 100)
        self.assertEqual(set(p.name for p in directory.iterdir()), {*files, 'predictions.xlsx'})

    def test_actual_save_block_preserves_state_dict_and_load_interface(self):
        tree = ast.parse((ROOT/'run_gamed.py').read_text(encoding='utf-8'))
        main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
        block = next(n for n in ast.walk(main) if isinstance(n, ast.If) and isinstance(n.test, ast.Constant)
                     and n.test.value is True)
        code = compile(ast.Module(body=[block], type_ignores=[]), '<save block>', 'exec')
        model, _ = self.model_and_loader()
        before = {k: v.clone() for k, v in model.state_dict().items()}
        for validation in (False, True):
            run = output_paths.create_run_directory(self.home, 'fakeddit', 'yang', 'gamed_without_veto', validation)
            for epoch in (0, 1):
                namespace = dict(os=os, torch=torch, model=model, validate_acc=.75, epoch=epoch, run_directory=run)
                with contextlib.redirect_stdout(io.StringIO()):
                    exec(code, namespace)
                path = run/'checkpoints'/f'epoch_{epoch+1:03d}.pkl'
                loaded = torch.load(path, weights_only=True)
                self.assertEqual(list(loaded), list(before))
                for key in before:
                    self.assertEqual(loaded[key].shape, before[key].shape)
                    self.assertTrue(torch.equal(loaded[key], before[key]))
                target, _ = self.model_and_loader()
                with patch.object(entry, 'torch', torch, create=True), \
                     patch.object(entry, '_read_path', lambda x: x, create=True):
                    result = entry.load_model(target, str(path))
                self.assertEqual((result.missing_keys, result.unexpected_keys), ([], []))


if __name__ == '__main__':
    unittest.main()
