"""Verify module-level and control-flow boundaries around source contracts.

Function and class fragments are reduced to markers only after their digests
match fixed expectations. The remaining AST is checked for imports, defaults,
execution order and module-level statements.
"""
import ast
import copy
import hashlib
import json
import unittest

import test_computation_contracts as contracts

ROOT = contracts.ROOT
EXPECTED_COMPUTATION_FIXTURE_SHA256 = '2151cd0698fca61f3c322c651e51ff33de9b65726bfcd9ce250b282e8d3fe95e'


def marker(name):
    return ast.Expr(value=ast.Constant(value='verified code block: ' + name))


def checked_fragment(node, expected, name):
    if contracts.digest(node) != expected:
        raise AssertionError('Changed protected computation: ' + name)
    return marker(name)


def checked_body(body, expected, name):
    module = ast.Module(body=body, type_ignores=[])
    return [checked_fragment(module, expected, name)]


def runtime_envelope(tree, source, expected_contract):
    """Keep all unprotected AST, reducing only digest-verified computation blocks.

    Paths are deliberately fixed. Moving, duplicating or extending a protected
    region fails either its hash check or the complete envelope comparison.
    """
    tree = copy.deepcopy(tree)
    if any(isinstance(node, ast.Constant) and isinstance(node.value, str)
           and node.value.startswith('verified code block: ')
           for node in ast.walk(tree)):
        raise AssertionError('Contract markers are not valid research source')
    if source == contracts.ENTRY:
        for index, node in enumerate(tree.body):
            if isinstance(node, ast.FunctionDef) and node.name in expected_contract['functions']:
                tree.body[index] = checked_fragment(
                    node, expected_contract['functions'][node.name], 'function:' + node.name)
        main = contracts.getfn(tree, 'main')
        blocks = expected_contract['main_blocks']
        # Verify nested regions before replacing the two outer list slices.
        loop = main.body[71]
        loop.body[2] = checked_fragment(loop.body[2], blocks['training'], 'training')
        loop.body[3].body = checked_body(
            loop.body[3].body, blocks['validation_and_save'], 'validation_and_save')
        for index, kind in ((34, 'train'), (35, 'val')):
            branch = main.body[index]
            for owner, dataset in ((branch, 'fakeddit'), (branch.orelse[0], 'yang')):
                name = f"loader:setting['{kind}_dataname'] == '{dataset}'"
                owner.body = checked_body(owner.body, blocks[name], name)
        main.body[44:64] = checked_body(
            main.body[44:64], blocks['loss_optimisers_schedulers'], 'loss_optimisers_schedulers')
        main.body[:11] = checked_body(main.body[:11], blocks['seed_and_backend'], 'seed_and_backend')
    elif source in contracts.MODELS.values():
        variant = next(key for key, path in contracts.MODELS.items() if path == source)
        for index, node in enumerate(tree.body):
            if isinstance(node, ast.ClassDef):
                key = 'MODEL' if node.name in ('GAMED', 'GAMEDWithoutVeto') else node.name
                expected = expected_contract['model_classes'][variant][key]
                tree.body[index] = checked_fragment(
                    contracts.normalise_model(node, source), expected, 'class:' + node.name)
    elif source in expected_contract['datasets']:
        for index, node in enumerate(tree.body):
            if isinstance(node, ast.ClassDef):
                tree.body[index] = checked_fragment(
                    node, expected_contract['datasets'][source], 'dataset:' + node.name)
    elif source != 'data/util.py':
        raise AssertionError('Unknown runtime boundary: ' + source)
    return tree


def validate_source(source, text, fixture, expected_contract):
    tree = ast.parse(text)
    if source in contracts.PRESENTATION_SITES:
        tree = contracts.normalise_presentation(tree, source)
    try:
        actual = runtime_envelope(tree, source, expected_contract)
    except (AttributeError, IndexError, KeyError, StopIteration) as error:
        raise AssertionError('Changed runtime boundary structure: ' + source) from error
    expected = ast.parse('\n'.join(fixture['sources'][source]))
    if contracts.canonical_dump(actual) != contracts.canonical_dump(expected):
        raise AssertionError('Changed runtime boundary: ' + source)


class RuntimeBoundaryContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = json.loads((ROOT/'tests/runtime_boundaries.json').read_text(encoding='utf-8'))
        cls.expected_contract = json.loads((ROOT/'tests/behaviour_contract.json').read_text(encoding='utf-8'))

    def test_fixture_integrity_and_source_inventory(self):
        self.assertEqual(hashlib.sha256((ROOT/'tests/behaviour_contract.json').read_bytes()).hexdigest(),
                         EXPECTED_COMPUTATION_FIXTURE_SHA256)
        self.assertEqual(set(self.fixture['sources']), {
            contracts.ENTRY, *contracts.MODELS.values(), *self.expected_contract['datasets'], 'data/util.py'})

    def test_complete_runtime_boundaries(self):
        for source in self.fixture['sources']:
            with self.subTest(source=source):
                validate_source(source, (ROOT/source).read_text(encoding='utf-8'),
                                self.fixture, self.expected_contract)

    def assert_rejected(self, source, old, new):
        text = (ROOT/source).read_text(encoding='utf-8')
        self.assertEqual(text.count(old), 1, (source, old))
        with self.assertRaises(AssertionError):
            validate_source(source, text.replace(old, new), self.fixture, self.expected_contract)

    def test_entry_and_image_changes_are_rejected(self):
        changes = [
            ('GT_size = 224', 'GT_size = 256'),
            ('word_token_length = 197', 'word_token_length = 128'),
            ('image_token_length = 197', 'image_token_length = 128'),
            ("setting['thresh'] = 0.5", "setting['thresh'] = 0.9"),
            ("setting['with_ambiguity'] = False", "setting['with_ambiguity'] = True"),
            ('batch_size=custom_batch_size,\n                      thresh', 'batch_size=1,\n                      thresh'),
            ("if len(setting['checkpoint_path'])!=0:", "if len(setting['checkpoint_path'])==0:"),
            ("if setting['val']!=0:", "if setting['val']==0:"),
            ('for epoch in range(custom_num_epochs):', 'for epoch in range(custom_num_epochs + 1):'),
            ('seed = 25', 'seed = 26'),
            ('import models_mae', 'import models_mae\n    torch.manual_seed(987)'),
        ]
        for old, new in changes:
            with self.subTest(change=old):
                self.assert_rejected(contracts.ENTRY, old, new)
        self.assert_rejected('data/util.py', 'astype(np.float32) / 255.', 'astype(np.float32) / 128.')
        self.assert_rejected('data/util.py', '3), dtype=np.uint8)', '3), dtype=np.float32)')

    def test_model_names_imports_and_rng_are_not_normalised_away(self):
        for source, name, wrong in (
            ('models/gamed.py', 'GAMED', 'GAMEDWithoutVeto'),
            ('models/gamed_without_veto.py', 'GAMEDWithoutVeto', 'GAMED'),
        ):
            changes = [
                (f'super({name}, self)', f'super({wrong}, self)'),
                (f'super({name}, self)', 'super(UnrelatedModel, self)'),
                ('class AdaIN(nn.Module):', 'torch.manual_seed(987)\n\nclass AdaIN(nn.Module):'),
                ('from timm.models.vision_transformer import Block',
                 'from timm.models.vision_transformer import Mlp as Block'),
                ('self.mm_score = None', 'self.mm_score = None\n        torch.rand(1)'),
                ('self.mm_score = None', f'self.mm_score = None\n        self.extra_class_reference = {name}'),
                ('self.mm_score = None', 'self.mm_score = None\n        from CNN_architectures.pytorch_inceptionet import RNG_SIDE_EFFECT'),
                ('self.mm_score = None', 'self.mm_score = None\n        from CNN_architectures.pytorch_inceptionet_v3 import Inception3 as Block'),
                ('self.num_expert = 2', 'self.num_expert = 3'),
            ]
            for old, new in changes:
                with self.subTest(source=source, change=new):
                    self.assert_rejected(source, old, new)

    def test_class_mapping_changes_only_two_ast_fields(self):
        for source, name in (('models/gamed.py', 'GAMED'),
                             ('models/gamed_without_veto.py', 'GAMEDWithoutVeto')):
            original = contracts.getfn(contracts.contract_tree(source), name)
            before = contracts.canonical_dump(original)
            mapped = contracts.normalise_model(original, source)
            self.assertEqual(contracts.canonical_dump(original), before)
            # Reverse only the two explicitly mapped fields; no import is removed.
            mapped.name = name
            contracts.getfn(mapped, '__init__').body[13].value.func.value.args[0].id = name
            self.assertEqual(contracts.canonical_dump(mapped), before)
            extra = copy.deepcopy(original)
            contracts.getfn(extra, '__init__').body.append(ast.parse(f'x = {name}').body[0])
            mapped_extra = contracts.normalise_model(extra, source)
            self.assertEqual(contracts.getfn(mapped_extra, '__init__').body[-1].value.id, name)

    def test_exact_display_corrections_reject_non_display_changes(self):
        changes = [
            ("logs.append(('IP_loss', ip_loss.item()))", "logs.append(('IP_loss', is_loss.item()))"),
            ("logs.append(('IP_loss', ip_loss.item()))", "other.append(('IP_loss', ip_loss.item()))"),
            ("logs.append(('IP_loss', val_ip_loss.item()))", "logs.append(('IP_loss', val_ip_loss.item() + 1))"),
            ("logs.append(('IP_loss', val_ip_loss.item()))", "logs.append((labels['IP_loss'], val_ip_loss.item()))"),
            ("setting['is_filter']))", "setting['with_ambiguity']))"),
        ]
        for old, new in changes:
            with self.subTest(change=new):
                self.assert_rejected(contracts.ENTRY, old, new)

    def test_covered_regions_cannot_be_extended_reordered_or_rebound(self):
        changes = [
            ('model.train()\n\n    print(get_parameter_number(model))',
             'model.train()\n    torch.rand(1)\n    print(get_parameter_number(model))'),
            ('torch.manual_seed(seed)\n    np.random.seed(seed)',
             'np.random.seed(seed)\n    torch.manual_seed(seed)'),
            ('main(args)\n', 'main(args)\n    torch.manual_seed(999)\n'),
        ]
        for old, new in changes:
            with self.subTest(change=new):
                self.assert_rejected(contracts.ENTRY, old, new)
        for source in self.fixture['sources']:
            text = (ROOT/source).read_text(encoding='utf-8')
            with self.subTest(source=source), self.assertRaises(AssertionError):
                validate_source(source, text + '\nimport random as np\n', self.fixture, self.expected_contract)

    def test_fixture_markers_cannot_replace_real_code(self):
        for source, name in (('models/gamed.py', 'GAMED'),
                             ('run_gamed.py', 'evaluate')):
            tree = ast.parse((ROOT/source).read_text(encoding='utf-8'))
            for index, node in enumerate(tree.body):
                if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name == name:
                    tree.body[index] = marker(('class:' if source.startswith('models/') else 'function:') + name)
                    break
            with self.subTest(source=source), self.assertRaises(AssertionError):
                validate_source(source, ast.unparse(ast.fix_missing_locations(tree)),
                                self.fixture, self.expected_contract)


if __name__ == '__main__':
    unittest.main()
