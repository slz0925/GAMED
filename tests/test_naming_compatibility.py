"""Checkpoint, optimiser and output-schema checks without pretrained resources."""
import ast
from collections import OrderedDict
import copy
import io
from pathlib import Path
import types
import unittest
from unittest.mock import patch

from models.checkpoint_compat import MODULE_NAMES, canonical_key, remap_state_dict

try:
    import torch
except ModuleNotFoundError as error:
    if error.name != 'torch':
        raise
    torch = None

ROOT = Path(__file__).resolve().parents[1]


def tree(name):
    return ast.parse((ROOT/name).read_text(encoding='utf-8'))


def execute(nodes, namespace):
    exec(compile(ast.fix_missing_locations(ast.Module(body=copy.deepcopy(nodes),
                                                     type_ignores=[])), '<source>', 'exec'), namespace)


def fixture(variant):
    """Use the model's real load method with small parameter/buffer subtrees."""
    from models.checkpoint_compat import prepare_state_dict
    cls = next(n for n in tree('models/'+variant+'.py').body
               if isinstance(n, ast.ClassDef) and n.name.startswith('GAMED'))
    loader = next(n for n in cls.body if isinstance(n, ast.FunctionDef)
                  and n.name == 'load_state_dict')
    cls.body = [loader]
    namespace = {'nn': torch.nn, 'prepare_state_dict': prepare_state_dict}
    execute([cls], namespace)
    model = namespace[cls.name]()
    for name in MODULE_NAMES.values():
        child = torch.nn.Linear(3, 2)
        child.register_buffer('running_count', torch.tensor(7, dtype=torch.int64))
        model.add_module(name, child)
    model.irrelevant_tensor = torch.nn.Parameter(torch.ones(1, 3))
    model.image_encoder.weight.requires_grad_(False)
    return model


def old_state(model, prefix=''):
    reverse = {value: key for key, value in MODULE_NAMES.items()}
    result = OrderedDict()
    for key, value in model.state_dict().items():
        root, separator, tail = key.partition('.')
        result[prefix+reverse.get(root, root)+separator+tail] = value.clone()
    return result


class NamingSourceTests(unittest.TestCase):
    def test_key_conversion_is_root_scoped_and_one_to_one(self):
        self.assertEqual(len(MODULE_NAMES), len(set(MODULE_NAMES.values())))
        self.assertEqual(canonical_key('module.vgg_net.block.weight'), 'ip_backbone.block.weight')
        self.assertEqual(canonical_key('mm_experts.vgg_net.weight'), 'mm_experts.vgg_net.weight')
        self.assertEqual(canonical_key('vgg_net_extra.weight'), 'vgg_net_extra.weight')
        self.assertEqual(canonical_key('module.ip_backbone.block.weight'), 'ip_backbone.block.weight')

    def test_both_constructors_register_current_roots_once(self):
        for variant in ('gamed', 'gamed_without_veto'):
            cls = next(n for n in tree('models/'+variant+'.py').body
                       if isinstance(n, ast.ClassDef) and n.name.startswith('GAMED'))
            init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '__init__')
            roots = [n.attr for n in ast.walk(init) if isinstance(n, ast.Attribute)
                     and isinstance(n.ctx, ast.Store) and isinstance(n.value, ast.Name) and n.value.id == 'self']
            for name in MODULE_NAMES.values():
                self.assertEqual(roots.count(name), 1, (variant, name))
            self.assertFalse(set(roots) & set(MODULE_NAMES))

    def test_export_fields_refer_to_the_correct_values(self):
        evaluate = next(n for n in tree('run_gamed.py').body if isinstance(n, ast.FunctionDef) and n.name == 'evaluate')
        actual = {}
        for node in ast.walk(evaluate):
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                target = node.targets[0]
                if isinstance(target, ast.Subscript) and ast.unparse(target.value) == 'record':
                    actual[target.slice.value] = ast.unparse(node.value)
        for column, value in [('final_feature', 'mix_feature'), ('is_feature', 'is_feature'),
                              ('t_feature', 't_feature'), ('ip_feature', 'ip_feature'),
                              ('mm_feature', 'mm_aux_feature')]:
            self.assertEqual(actual[column], value+'[idx].cpu().numpy().tolist()')
        row = next(n for n in ast.walk(evaluate) if isinstance(n, ast.Assign)
                   and isinstance(n.targets[0], ast.Tuple)
                   and ast.unparse(n.targets[0]).startswith("(record['y_pred_mm']"))
        self.assertEqual([n.slice.value for n in row.targets[0].elts],
                         ['y_pred_mm', 'y_pred_is', 'y_pred_t', 'y_pred_ip'])
        self.assertEqual([ast.unparse(n) for n in row.value.elts],
                         [n+'[idx].item()' for n in ['mm_aux_output', 'is_output', 't_output', 'ip_output']])

    def test_log_labels_preserve_their_expressions(self):
        expected = {
            'Mix_loss': {'mix_loss.item()', 'val_loss.item()'},
            'IS_loss': {'is_loss.item()', 'val_is_loss.item()'},
            'T_loss': {'t_loss.item()', 'val_t_loss.item()'},
            'IP_loss': {'ip_loss.item()', 'val_ip_loss.item()'},
            'MM_mean_prob': {'torch.mean(torch.sigmoid(mm_aux_output)).item()'},
            'Irr_mean': {'irrelevant_mean.item()'}, 'Train_acc': {'mean_acc'},
            'MM_ambiguity_loss': {'loss_ambiguity.item()'},
        }
        actual = {name: set() for name in expected}
        counts = {name: 0 for name in expected}
        for n in ast.walk(tree('run_gamed.py')):
            if isinstance(n, ast.Call) and ast.unparse(n.func) == 'logs.append':
                label, value = n.args[0].elts
                if isinstance(label, ast.Constant) and label.value in expected:
                    actual[label.value].add(ast.unparse(value)); counts[label.value] += 1
        self.assertEqual(actual, expected)
        self.assertEqual(counts, {k: len(v) for k, v in expected.items()})


@unittest.skipUnless(torch is not None, 'Checkpoint tensor tests require PyTorch')
class CheckpointTensorTests(unittest.TestCase):
    def test_all_old_parameters_and_buffers_load_in_both_variants(self):
        for variant in ('gamed', 'gamed_without_veto'):
            for prefix in ('', 'module.'):
                with self.subTest(variant=variant, prefix=prefix):
                    source = fixture(variant); target = fixture(variant)
                    before_ids = [id(p) for p in target.parameters()]
                    state = old_state(source, prefix)
                    mapped = remap_state_dict(state)
                    self.assertEqual(list(mapped), list(source.state_dict()))
                    for old, new in zip(state.values(), mapped.values()):
                        self.assertIs(old, new)
                    incompatible = target.load_state_dict(state)
                    self.assertEqual((incompatible.missing_keys, incompatible.unexpected_keys), ([], []))
                    self.assertEqual(before_ids, [id(p) for p in target.parameters()])
                    for k, v in source.state_dict().items():
                        self.assertTrue(torch.equal(v, target.state_dict()[k]), k)

    def test_new_checkpoint_serialization_round_trip(self):
        source = fixture('gamed'); target = fixture('gamed')
        stream = io.BytesIO(); torch.save(source.state_dict(), stream); stream.seek(0)
        target.load_state_dict(torch.load(stream, weights_only=True))
        for k, v in source.state_dict().items():
            self.assertTrue(torch.equal(v, target.state_dict()[k]), k)

    def test_missing_unexpected_and_shape_errors_precede_any_copy(self):
        for variant in ('gamed', 'gamed_without_veto'):
            model = fixture(variant)
            before = {k: v.clone() for k, v in model.state_dict().items()}
            for kind in ('missing', 'unexpected', 'shape'):
                state = old_state(model)
                state['vgg_trim.weight'].fill_(99)
                if kind == 'missing': del state['vgg_net.weight']
                if kind == 'unexpected': state['typo.weight'] = torch.zeros(1)
                if kind == 'shape': state['vgg_net.weight'] = torch.zeros(4, 3)
                with self.subTest(variant=variant, kind=kind), self.assertRaises(RuntimeError):
                    model.load_state_dict(state)
                for k, v in model.state_dict().items():
                    self.assertTrue(torch.equal(v, before[k]), k)

    def test_alias_and_module_prefix_collisions_are_rejected_even_if_equal(self):
        for key in ('ip_backbone.weight', 'module.vgg_net.weight'):
            model = fixture('gamed'); state = old_state(model)
            state[key] = state['vgg_net.weight']
            for strict in (True, False):
                with self.subTest(key=key, strict=strict), self.assertRaises(ValueError):
                    model.load_state_dict(state, strict=strict)

    def test_explicit_partial_load_warns_and_returns_diagnostics(self):
        model = fixture('gamed'); state = old_state(model)
        del state['aux_classifier.weight']; state['unknown'] = torch.zeros(1)
        with self.assertWarnsRegex(RuntimeWarning, 'mm_aux_classifier.weight'):
            incompatible = model.load_state_dict(state, strict=False)
        self.assertEqual(incompatible.missing_keys, ['mm_aux_classifier.weight'])
        self.assertEqual(incompatible.unexpected_keys, ['unknown'])
        state['vgg_net.weight'] = torch.zeros(1)
        with self.assertRaises(RuntimeError): model.load_state_dict(state, strict=False)

    def test_metadata_tracks_modules_and_preserves_batchnorm_versions(self):
        state = OrderedDict([('module.vgg_net.bn.running_mean', torch.ones(2))])
        state._metadata = OrderedDict([('', {'version': 1}), ('module', {'version': 8}),
                                       ('module.vgg_net.bn', {'version': 2})])
        mapped = remap_state_dict(state)
        self.assertEqual(mapped._metadata, OrderedDict([('', {'version': 8}), ('ip_backbone.bn', {'version': 2})]))
        state._metadata['ip_backbone.bn'] = {'version': 2}
        with self.assertRaises(ValueError): remap_state_dict(state)

    def test_real_entry_loader_is_strict_and_uses_model_compatibility(self):
        import run_gamed
        model = fixture('gamed'); source = fixture('gamed')
        with patch.object(run_gamed, 'torch', create=True) as mocked, \
             patch.object(run_gamed, '_fnd_read_path', lambda p: p, create=True):
            mocked.load.return_value = old_state(source, 'module.')
            run_gamed.load_model(model, 'not-read.pkl')
            mocked.load.assert_called_once_with('not-read.pkl')
            del mocked.load.return_value['module.vgg_net.weight']
            with self.assertRaises(RuntimeError): run_gamed.load_model(model, 'not-read.pkl')

    def test_each_parameter_keeps_its_optimizer_group_and_settings(self):
        main = next(n for n in tree('run_gamed.py').body if isinstance(n, ast.FunctionDef) and n.name == 'main')
        loop = next(n for n in main.body if isinstance(n, ast.For) and ast.unparse(n.iter) == 'model.named_parameters()')
        assignments = [n for n in main.body if isinstance(n, ast.Assign)
                       and ast.unparse(n.targets[0]) in ('optimizer', 'optimizer_fast', 'optimizer_extremefast')]
        for variant in ('gamed', 'gamed_without_veto'):
            model = fixture(variant)
            for fine_tuning in (False, True):
                ns = {'model': model, 'torch': torch, 'fine_tuning': fine_tuning}
                for speed in ('normal', 'fast', 'extremefast'):
                    ns['optim_params_'+speed] = []; ns['name_params_'+speed] = []
                execute([loop, *assignments], ns)
                seen = []
                for suffix, rate in [('', 1e-5), ('_fast', 1e-5 if fine_tuning else 5e-5),
                                     ('_extremefast', 1e-5 if fine_tuning else 1e-4)]:
                    group = ns['optimizer'+suffix].param_groups[0]
                    self.assertEqual((group['lr'], group['weight_decay'], group['betas']), (rate, .01, (.9, .999)))
                    ids = {id(p) for p in group['params']}; seen.extend(ids)
                    for name, parameter in model.named_parameters():
                        wanted = '' if name.startswith(('image_encoder.', 'text_encoder.')) else (
                            '_extremefast' if name.startswith('ip_backbone.') or name == 'irrelevant_tensor' else '_fast')
                        self.assertEqual(id(parameter) in ids, parameter.requires_grad and wanted == suffix, name)
                self.assertEqual(len(seen), len(set(seen)))
                self.assertEqual(set(seen), {id(p) for p in model.parameters() if p.requires_grad})


if __name__ == '__main__':
    unittest.main()
