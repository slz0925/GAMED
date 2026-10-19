"""Check source-level computation and parameter-registration contracts.

These AST checks do not replace tensor-level or numerical evaluation in the
full model environment.
"""
from pathlib import Path
import ast
import copy
import hashlib
import inspect
import json
import unittest

ROOT=Path(__file__).resolve().parents[1]
ENTRY='run_gamed.py'
MODELS={'gamed':'models/gamed.py','gamed_without_veto':'models/gamed_without_veto.py'}

def getfn(tree,name):
    return next(n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==name)

def canonical_dump(node):
    """Keep the Python 3.12 empty-field representation on Python 3.13.

    Empty lists remain part of the contract; AST fields are not discarded.
    """
    options = {'include_attributes': False}
    if 'show_empty' in inspect.signature(ast.dump).parameters:
        options['show_empty'] = True
    return ast.dump(node, **options)


# Each text-only comparison site has an exact source, scope and AST location.
# Calls must match completely before static strings are normalised.
# Every listed site must occur once; all other AST content is preserved.
PRESENTATION_SITES = {'models/gamed.py': {'docstrings': ((('AdaIN', 'mu'),
                                     'Return the mean across the feature dimension for each sample.',
                                     ' Takes a (n,c,h,w) tensor as input and returns the average across\n'
                                     "        it's spatial dimensions as (h,w) tensor [See eq. 5 of paper]"),
                                    (('AdaIN', 'sigma'),
                                     'Return the per-sample feature standard deviation with an epsilon.',
                                     ' Takes a (n,c,h,w) tensor as input and returns the standard deviation\n'
                                     "        across it's spatial dimensions as (h,w) tensor [See eq. 6 of "
                                     'paper] Note\n'
                                     '        the permutations are required for broadcasting'),
                                    (('AdaIN', 'forward'),
                                     'Apply the existing prediction-guided feature adjustment.',
                                     ' Takes a content embeding x and a style embeding y and changes\n'
                                     '        transforms the mean and standard deviation of the content '
                                     'embedding to\n'
                                     '        that of the style. [See eq. 8 of paper] Note the permutations '
                                     'are\n'
                                     '        required for broadcasting'),
                                    (('TokenAttention',),
                                     'Aggregate token features using learned, unnormalised attention scores.',
                                     '\n    Compute attention layer\n    ')),
                     'logs': ((('GAMED', '__init__'),
                               ('body', 7, 'value'),
                               "print('Using AdaIN-based feature adjustment')",
                               ((('args', 0), 'Using AdaIN-based feature adjustment', 'we are using adaIN'),)),
                              (('GAMED', '__init__'),
                               ('body', 19, 'value'),
                               "print('Text encoder: ERNIE 2.0 Base (English)')",
                               ((('args', 0), 'Text encoder: ERNIE 2.0 Base (English)', 'Embedding: using {}'),)))},
 'models/gamed_without_veto.py': {'docstrings': ((('AdaIN', 'mu'),
                                                  'Return the mean across the feature dimension for each '
                                                  'sample.',
                                                  ' Takes a (n,c,h,w) tensor as input and returns the average '
                                                  'across\n'
                                                  "        it's spatial dimensions as (h,w) tensor [See eq. 5 "
                                                  'of paper]'),
                                                 (('AdaIN', 'sigma'),
                                                  'Return the per-sample feature standard deviation with an '
                                                  'epsilon.',
                                                  ' Takes a (n,c,h,w) tensor as input and returns the standard '
                                                  'deviation\n'
                                                  "        across it's spatial dimensions as (h,w) tensor [See "
                                                  'eq. 6 of paper] Note\n'
                                                  '        the permutations are required for broadcasting'),
                                                 (('AdaIN', 'forward'),
                                                  'Apply the existing prediction-guided feature adjustment.',
                                                  ' Takes a content embeding x and a style embeding y and '
                                                  'changes\n'
                                                  '        transforms the mean and standard deviation of the '
                                                  'content embedding to\n'
                                                  '        that of the style. [See eq. 8 of paper] Note the '
                                                  'permutations are\n'
                                                  '        required for broadcasting'),
                                                 (('TokenAttention',),
                                                  'Aggregate token features using learned, unnormalised '
                                                  'attention scores.',
                                                  '\n    Compute attention layer\n    '),
                                                 (('GAMEDWithoutVeto', 'veto'),
                                                  "An unused veto helper; not called by this model's forward "
                                                  'pass.\n'
                                                  '\n'
                                                  '        Return the average with the first opposing branch '
                                                  'that meets the\n'
                                                  "        confidence threshold and exceeds the mixture's "
                                                  'maximum score; otherwise\n'
                                                  '        return the mixture unchanged.\n'
                                                  '        ',
                                                  '\n'
                                                  '        参数:\n'
                                                  '        - outputs: 包含image_only_output, text_only_output, '
                                                  'vgg_only_output, aux_output的列表\n'
                                                  '        - mix_output: 综合考虑所有分支后的输出结果\n'
                                                  '        - confidence_threshold: 置信度阈值，用于判断是否应该考虑反对意见\n'
                                                  '        \n'
                                                  '        返回:\n'
                                                  '        - 调整后的mix_output\n'
                                                  '        ')),
                                  'logs': ((('GAMEDWithoutVeto', '__init__'),
                                            ('body', 7, 'value'),
                                            "print('Using AdaIN-based feature adjustment')",
                                            ((('args', 0),
                                              'Using AdaIN-based feature adjustment',
                                              'we are using adaIN'),)),
                                           (('GAMEDWithoutVeto', '__init__'),
                                            ('body', 19, 'value'),
                                            "print('Text encoder: ERNIE 2.0 Base (English)')",
                                            ((('args', 0),
                                              'Text encoder: ERNIE 2.0 Base (English)',
                                              'Embedding: using {}'),)))},
 'run_gamed.py': {'docstrings': (),
                  'logs': ((('main',),
                            ('body', 9, 'value'),
                            "print('Initialising CUDA AMP gradient scaler')",
                            ((('args', 0), 'Initialising CUDA AMP gradient scaler', 'Using amp (Tempt)'),)),
                           (('main',),
                            ('body', 35, 'body', 0, 'value'),
                            "print('Preparing Fakeddit validation data')",
                            ((('args', 0),
                              'Preparing Fakeddit validation data',
                              'Using fakeddit as inference'),)),
                           (('main',),
                            ('body', 35, 'orelse', 0, 'body', 0, 'value'),
                            "print('Preparing Yang validation data')",
                            ((('args', 0), 'Preparing Yang validation data', 'Using fakeddit as inference'),)),
                           (('main',),
                            ('body', 52, 'value'),
                            "print(f'Fine-tuning mode: {fine_tuning}')",
                            ((('args', 0, 'values', 0),
                              'Fine-tuning mode: ',
                              'THE CURRENT MODE FOR FINETUNING:'),)),
                           (('main',),
                            ('body', 69, 'value'),
                            "print('Starting training or validation')",
                            ((('args', 0), 'Starting training or validation', 'training model'),)),
                           (('main',),
                            ('body', 72, 'body', 2, 'body', 4, 'value'),
                            "print('Training epoch completed')",
                            ((('args', 0), 'Training epoch completed', 'end training...'),)),
                           (('main',),
                            ('body', 72, 'body', 3, 'body', 3, 'value'),
                            "print('Starting validation')",
                            ((('args', 0), 'Starting validation', 'begin evaluate...'),)),
                           (('main',),
                            ('body', 72, 'body', 3, 'body', 13, 'value'),
                            "print('Epoch [%d/%d],  validation accuracy: %.4f at threshold index %d (best %.4f "
                            "at epoch %d)' % (epoch + 1, custom_num_epochs, validate_acc, val_thresh, "
                            'best_acc_so_far, best_epoch_record))',
                            ((('args', 0, 'left'),
                              'Epoch [%d/%d],  validation accuracy: %.4f at threshold index %d (best %.4f at '
                              'epoch %d)',
                              'Epoch [%d/%d],  Val_Acc: %.4f. at thresh %.4f (so far %.4f in Epoch %d) .'),)),
                           (('main',),
                            ('body', 72, 'body', 3, 'body', 14, 'value'),
                            "print(f'Branch accuracies: IS {is_acc}, T {t_acc}, IP {ip_acc}, MM "
                            "{mm_acc}')",
                            ((('args', 0, 'values', 0),
                              'Branch accuracies: IS ',
                              'Single Modalities Accuracy: Img '),
                             (('args', 0, 'values', 2), ', T ', ' Text '),
                             (('args', 0, 'values', 4), ', IP ', ' VGG '),
                             (('args', 0, 'values', 6), ', MM ', ' SSIM '))),
                           (('main',),
                            ('body', 72, 'body', 3, 'body', 26, 'value'),
                            "print('Validation completed')",
                            ((('args', 0), 'Validation completed', 'end evaluate...'),)),
                           (('evaluate',),
                            ('body', 52, 'value'),
                            "print(f'Excel saved to {pandas_file}')",
                            ((('args', 0, 'values', 0), 'Excel saved to ', 'Excel Saved at '),)))}}

REFERENCE_NOTE_SOURCE = 'models/gamed_without_veto.py'
REFERENCE_NOTE_SCOPE = ('GAMEDWithoutVeto', 'get_pretrain_features')
REFERENCE_NOTE_INDEX = 3
REFERENCE_NOTE_TEXT = '\n    veto投票反对\n    '

# Only literal display labels are mapped; computed expressions remain checked.
PRESENTATION_SITES[ENTRY]['logs'] += (
    (('main',), ('body', 31, 'value'),
     "print('Compatibility filter flag (unused by supported loaders): {}'.format(setting['is_filter']))",
     ((('args', 0, 'func', 'value'),
       'Compatibility filter flag (unused by supported loaders): {}', 'Filter the dataset? {}'),)),
    (('main',), ('body', 72, 'body', 2, 'body', 2, 'body', 0, 'body', 26, 'value'),
     "logs.append(('IP_loss', ip_loss.item()))",
     ((('args', 0, 'elts', 0), 'IP_loss', 'VGG'),)),
    (('evaluate',), ('body', 20, 'body', 17, 'body', 4, 'value'),
     "logs.append(('IP_loss', val_ip_loss.item()))",
     ((('args', 0, 'elts', 0), 'IP_loss', 'vgg_loss'),)),
)


def _scope(module, names):
    owner = module
    for name in names:
        matches = [item for item in owner.body
                   if isinstance(item, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
                   and item.name == name]
        if len(matches) != 1:
            raise AssertionError(('Expected one declaration', names, name, len(matches)))
        owner = matches[0]
    return owner


def _at(node, path):
    for step in path:
        node = node[step] if isinstance(step, int) else getattr(node, step)
    return node


def _reject_print_bindings(module):
    # Conservatively reject lexical rebinding anywhere in an approved file.
    for item in ast.walk(module):
        bound = (
            isinstance(item, ast.Name) and item.id == 'print'
            and isinstance(item.ctx, (ast.Store, ast.Del))
            or isinstance(item, ast.arg) and item.arg == 'print'
            or isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and item.name == 'print'
            or isinstance(item, ast.alias) and (item.asname or item.name.split('.')[0]) == 'print'
            or isinstance(item, ast.alias) and item.name == '*'
            or isinstance(item, ast.ExceptHandler) and item.name == 'print'
            or isinstance(item, (ast.Global, ast.Nonlocal)) and 'print' in item.names
            or isinstance(item, (ast.MatchAs, ast.MatchStar)) and item.name == 'print'
            or isinstance(item, ast.MatchMapping) and item.rest == 'print'
        )
        if bound:
            raise AssertionError('An approved source file must not bind or shadow print')


def normalise_presentation(node, source=None):
    """Normalise specific literal display text while preserving computations.

    No source context means no normalisation. Recognised contexts require the
    complete module, unique scopes, exact call expressions and one match per
    listed location. Only true docstrings and named static log leaves change.
    """
    node = copy.deepcopy(node)
    if source is None:
        return node
    if source not in PRESENTATION_SITES:
        raise ValueError('No approved presentation edits for source: ' + str(source))
    if not isinstance(node, ast.Module):
        raise AssertionError('Presentation normalisation requires a complete source module')
    _reject_print_bindings(node)
    sites = PRESENTATION_SITES[source]
    seen = set()
    for names, current, reference in sites['docstrings']:
        owner = _scope(node, names)
        if not (owner.body and isinstance(owner.body[0], ast.Expr)
                and isinstance(owner.body[0].value, ast.Constant)
                and isinstance(owner.body[0].value.value, str)
                and ast.get_docstring(owner, clean=False) == current):
            raise AssertionError(('Expected the approved leading docstring', source, names))
        key = ('docstring', names)
        if key in seen:
            raise AssertionError(('Duplicate approved site', key))
        seen.add(key)
        owner.body[0].value.value = reference
    for names, path, expected_call, leaves in sites['logs']:
        owner = _scope(node, names)
        try:
            call = _at(owner, path)
        except (AttributeError, IndexError, TypeError) as error:
            raise AssertionError(('Missing reviewed log site', source, names, path)) from error
        expected = ast.parse(expected_call).body[0].value
        expected_dump = canonical_dump(expected)
        occurrences = sum(isinstance(item, ast.Call)
                          and canonical_dump(item) == expected_dump
                          for item in ast.walk(owner))
        if occurrences != 1:
            raise AssertionError(('Expected one protected log occurrence', source, names, path, occurrences))
        if canonical_dump(call) != expected_dump:
            raise AssertionError(('Changed protected log expression', source, names, path))
        for leaf_path, current, reference in leaves:
            # Paths are explicit literal/template/JoinedStr-text positions in
            # the expected call, never a walk over arguments or keyword values.
            literal = _at(call, leaf_path)
            if not isinstance(literal, ast.Constant) or type(literal.value) is not str or literal.value != current:
                raise AssertionError(('Changed protected log text', source, names, path, leaf_path))
            key = ('log', names, path, leaf_path)
            if key in seen:
                raise AssertionError(('Duplicate approved site', key))
            seen.add(key)
            literal.value = reference
        if (source in MODELS.values() and names[-1] == '__init__'
                and path == ('body', 19, 'value')):
            # Apply the reference form only at the exact text-encoder log site.
            call.args[0] = ast.parse("'Embedding: using {}'.format(model_name)",
                                     mode='eval').body
    expected_count = len(sites['docstrings']) + sum(len(log[3]) for log in sites['logs'])
    if len(seen) != expected_count:
        raise AssertionError(('Approved site count mismatch', source))
    if source == REFERENCE_NOTE_SOURCE:
        # One reference-only, unreachable string after the method return. Insert
        # it only in the in-memory reference AST; do not ignore arbitrary Exprs.
        method = _scope(node, REFERENCE_NOTE_SCOPE)
        expected_return = ast.parse('return image_feature, text_feature').body[0]
        if (len(method.body) != REFERENCE_NOTE_INDEX
                or canonical_dump(method.body[-1]) != canonical_dump(expected_return)):
            raise AssertionError('Changed reference-note position or return statement')
        method.body.insert(REFERENCE_NOTE_INDEX, ast.Expr(value=ast.Constant(value=REFERENCE_NOTE_TEXT)))
    return node


def contract_tree(source):
    tree = ast.parse((ROOT/source).read_text(encoding='utf8'))
    return normalise_presentation(tree, source) if source in PRESENTATION_SITES else tree


def digest(node):
    return hashlib.sha256(canonical_dump(node).encode()).hexdigest()


def normalise_model(node, source):
    """Map only each model's class declaration and exact super() call.

    Import statements and all other names remain part of the source digest.
    """
    names = {'models/gamed.py': 'GAMED',
             'models/gamed_without_veto.py': 'GAMEDWithoutVeto'}
    if source not in names or not isinstance(node, ast.ClassDef):
        raise AssertionError('Model normalisation requires a known source and class')
    node = copy.deepcopy(node)
    expected_name = names[source]
    if node.name in ('AdaIN', 'SimpleGate', 'TokenAttention'):
        return node
    if node.name != expected_name:
        raise AssertionError(('Unexpected model class', source, node.name))
    constructor = _scope(node, ('__init__',))
    expected = ast.parse(f'super({expected_name}, self).__init__()').body[0]
    if (len(constructor.body) <= 13
            or canonical_dump(constructor.body[13]) != canonical_dump(expected)):
        raise AssertionError(('Changed model super() call', source))
    node.name = 'MODEL'
    constructor.body[13].value.func.value.args[0].id = 'MODEL'
    return node

def main_blocks(tree):
    main=getfn(tree,'main');result={}
    seed_end=next(i for i,n in enumerate(main.body) if isinstance(n,ast.Expr) and ast.unparse(n)=="print('loading data')")
    result['seed_and_backend']=ast.Module(body=main.body[:seed_end],type_ignores=[])
    optim_start=next(i for i,n in enumerate(main.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='loss_cross_entropy' for t in n.targets))
    optim_end=next(i for i,n in enumerate(main.body) if isinstance(n,ast.Expr) and ast.unparse(n)=="print('loader size ' + str(len(train_loader)))")
    result['loss_optimisers_schedulers']=ast.Module(body=main.body[optim_start:optim_end],type_ignores=[])
    loop=next(n for n in main.body if isinstance(n,ast.For) and ast.unparse(n.target)=='epoch')
    result['training']=next(n for n in loop.body if isinstance(n,ast.If) and ast.unparse(n.test)=="setting['val'] == 0")
    validation=next(n for n in loop.body if isinstance(n,ast.With))
    seq=validation.body
    result['validation_and_save']=ast.Module(body=seq,type_ignores=[])
    for n in ast.walk(main):
        if isinstance(n,ast.If) and ast.unparse(n.test) in [f"setting['{kind}_dataname'] == '{data}'" for kind in ('train','val') for data in ('fakeddit','yang')]:
            result['loader:'+ast.unparse(n.test)]=ast.Module(body=n.body,type_ignores=[])
    return result


class ComputationContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract=json.loads((ROOT/'tests/behaviour_contract.json').read_text())

    def test_collate_loading_and_evaluation_are_unchanged(self):
        tree=contract_tree(ENTRY)
        for name,expected in self.contract['functions'].items():
            with self.subTest(function=name):self.assertEqual(digest(getfn(tree,name)),expected)

    def test_training_rng_optimisers_loaders_and_saves_are_unchanged(self):
        actual=main_blocks(contract_tree(ENTRY))
        self.assertEqual(set(actual),set(self.contract['main_blocks']))
        for name,node in actual.items():
            with self.subTest(block=name):self.assertEqual(digest(node),self.contract['main_blocks'][name])

    def test_models_preserve_registration_order_and_every_computation(self):
        for key,path in MODELS.items():
            classes={n.name:digest(normalise_model(n,path)) for n in contract_tree(path).body if isinstance(n,ast.ClassDef)}
            classes={('MODEL' if n in ('GAMED','GAMEDWithoutVeto') else n):v for n,v in classes.items()}
            with self.subTest(variant=key):self.assertEqual(classes,self.contract['model_classes'][key])

    def test_dataset_classes_are_unchanged(self):
        for name,expected in self.contract['datasets'].items():
            tree=contract_tree(name)
            with self.subTest(dataset=name):self.assertEqual(digest(next(n for n in tree.body if isinstance(n,ast.ClassDef))),expected)

    def test_backbones_and_positions_match_original_bytes_except_exact_text(self):
        backbone = 'CNN_architectures/pytorch_InceptionResNetV2.py'
        for name, expected in self.contract['unchanged_files'].items():
            content = (ROOT/name).read_bytes()
            if name == backbone:
                for current, original in (
                    (b'print("Using Bayar filtering in InceptionResNetV2")',
                     b'print("We are using SRM in Inception")'),
                    (b'# Special attributes', b'# Special attributs'),
                ):
                    self.assertEqual(content.count(current), 1)
                    content = content.replace(current, original, 1)
            with self.subTest(file=name):
                self.assertEqual(hashlib.sha256(content).hexdigest(), expected)

    def test_backbone_display_changes_are_precisely_limited(self):
        path = ROOT/'CNN_architectures/pytorch_InceptionResNetV2.py'
        text = path.read_bytes()
        self.assertIn(b'print("Using Bayar filtering in InceptionResNetV2")', text)
        self.assertNotIn(b'print("We are using SRM in Inception")', text)

    def test_canonical_representation_retains_empty_fields(self):
        self.assertEqual(canonical_dump(ast.Module(body=[],type_ignores=[])),
                         'Module(body=[], type_ignores=[])')
        declaration=canonical_dump(ast.parse('def f(): pass'))
        self.assertIn('decorator_list=[]',declaration)
        self.assertIn('type_params=[]',declaration)

    def test_presentation_normalisation_does_not_hide_computation(self):
        # A matching phrase outside docstrings/print() remains part of the AST.
        expression = ast.parse("value = 'Using AdaIN-based feature adjustment'")
        self.assertIn('Using AdaIN-based feature adjustment',
                      canonical_dump(normalise_presentation(expression)))
        self.assertNotEqual(digest(expression),
                            digest(ast.parse("value = 'we are using adaIN'")))

    def test_contract_rejects_a_changed_calculation(self):
        node=getfn(contract_tree(ENTRY),'evaluate')
        self.assertEqual(digest(node),self.contract['functions']['evaluate'])
        changed=copy.deepcopy(node)
        literal=next(n for n in ast.walk(changed) if isinstance(n,ast.Constant) and type(n.value) in (int,float))
        literal.value+=1
        self.assertNotEqual(digest(changed),self.contract['functions']['evaluate'])

    def raw_tree(self, source=ENTRY):
        return ast.parse((ROOT/source).read_text(encoding='utf8'))

    def test_approved_locations_and_occurrence_counts_are_fixed(self):
        self.assertEqual(set(PRESENTATION_SITES), {ENTRY, *MODELS.values()})
        counts = {source: (len(sites['docstrings']), len(sites['logs']),
                           sum(len(log[3]) for log in sites['logs']))
                  for source, sites in PRESENTATION_SITES.items()}
        self.assertEqual(counts, {
            'models/gamed.py': (4, 2, 2),
            'models/gamed_without_veto.py': (5, 2, 2),
            ENTRY: (0, 14, 17),
        })
        for source in PRESENTATION_SITES:
            tree = self.raw_tree(source)
            before = canonical_dump(tree)
            normalise_presentation(tree, source)
            self.assertEqual(canonical_dump(tree), before)

    def test_ast_counterexamples_keep_distinct_digests(self):
        pairs = [
            ("print(scores[', MM '])", "print(scores[' SSIM '])"),
            ('print(f"{scores[\', MM \']}")', 'print(f"{scores[\' SSIM \']}")'),
            ("print('ok', file=open('Preparing Yang validation data', 'w'))",
             "print('ok', file=open('Using fakeddit as inference', 'w'))"),
            ("def score(print): return print('Preparing Yang validation data')",
             "def score(print): return print('Using fakeddit as inference')"),
        ]
        for left, right in pairs:
            with self.subTest(left=left):
                a = normalise_presentation(ast.parse(left))
                b = normalise_presentation(ast.parse(right))
                self.assertNotEqual(digest(a), digest(b))
                self.assertEqual(canonical_dump(a), canonical_dump(ast.parse(left)))
                self.assertEqual(canonical_dump(b), canonical_dump(ast.parse(right)))

    def test_registered_logs_reject_computed_arguments_and_keyword_paths(self):
        source = 'models/gamed.py'
        scope, path, _, _ = PRESENTATION_SITES[source]['logs'][1]
        replacements = [
            "print(scores[', MM '])",
            "print(scores[' SSIM '])",
            'print(f"{scores[\', MM \']}")',
            'print(f"{scores[\' SSIM \']}")',
            "print('Text encoder: {}'.format(paths['Preparing Yang validation data']))",
            "print('Text encoder: {}'.format(paths['Using fakeddit as inference']))",
            "print('Text encoder: {}'.format(model_name), file=open('Preparing Yang validation data', 'w'))",
            "print('Text encoder: {}'.format(model_name), file=open('Using fakeddit as inference', 'w'))",
            "print('Text encoder: {}'.format(model_name), flush=True)",
        ]
        for expression in replacements:
            with self.subTest(expression=expression):
                tree = self.raw_tree(source)
                call = _at(_scope(tree, scope), path)
                replacement = ast.parse(expression).body[0].value
                call.func, call.args, call.keywords = replacement.func, replacement.args, replacement.keywords
                with self.assertRaises(AssertionError):
                    normalise_presentation(tree, source)

    def test_registered_fstring_preserves_evaluated_values_and_format_specs(self):
        for expression in ("scores[', MM ']", "scores[' SSIM ']"):
            tree = self.raw_tree()
            spec = next(s for s in PRESENTATION_SITES[ENTRY]['logs']
                        if s[2].startswith("print(f'Branch accuracies:"))
            call = _at(_scope(tree, spec[0]), spec[1])
            call.args[0].values[1].value = ast.parse(expression, mode='eval').body
            with self.assertRaises(AssertionError):
                normalise_presentation(tree, ENTRY)
        tree = self.raw_tree()
        call = _at(_scope(tree, spec[0]), spec[1])
        call.args[0].values[1].format_spec = ast.JoinedStr(values=[ast.Constant(value='.3f')])
        with self.assertRaises(AssertionError):
            normalise_presentation(tree, ENTRY)

    def test_approved_files_reject_print_shadowing(self):
        bindings = [
            'print = transform', 'del print', 'import transform as print',
            'from package import print', 'from package import *',
            'def print(value): return value', 'class print: pass',
            'def helper(print): return print(1)',
            'def helper():\n print = transform',
            'try:\n pass\nexcept Exception as print:\n pass',
        ]
        for binding in bindings:
            tree = self.raw_tree()
            tree.body.extend(ast.parse(binding).body)
            with self.subTest(binding=binding), self.assertRaises(AssertionError):
                normalise_presentation(tree, ENTRY)

    def test_source_scope_text_and_occurrence_limits_fail_closed(self):
        with self.assertRaises(ValueError):
            normalise_presentation(self.raw_tree(), 'unknown/run_gamed.py')
        with self.assertRaises(AssertionError):
            normalise_presentation(getfn(self.raw_tree(), 'main'), ENTRY)
        for change in ('rename', 'duplicate', 'extra', 'appended-log', 'wrong-text'):
            tree = self.raw_tree()
            main = getfn(tree, 'main')
            if change == 'rename':
                main.name = 'other_main'
            elif change == 'duplicate':
                tree.body.append(copy.deepcopy(main))
            elif change == 'extra':
                main.body.insert(9, ast.parse("print('Initialising CUDA AMP gradient scaler')").body[0])
            elif change == 'appended-log':
                main.body.append(ast.parse("print('Initialising CUDA AMP gradient scaler')").body[0])
            else:
                call = _at(main, PRESENTATION_SITES[ENTRY]['logs'][2][1])
                call.args[0].value = 'Preparing Fakeddit validation data'
            with self.subTest(change=change), self.assertRaises(AssertionError):
                normalise_presentation(tree, ENTRY)

    def test_only_leading_docstrings_at_approved_scopes_are_normalised(self):
        source = 'models/gamed.py'
        tree = self.raw_tree(source)
        method = _scope(tree, ('AdaIN', 'mu'))
        method.body.append(copy.deepcopy(method.body[0]))
        normalised = normalise_presentation(tree, source)
        actual = _scope(normalised, ('AdaIN', 'mu'))
        self.assertNotEqual(actual.body[0].value.value, actual.body[-1].value.value)
        self.assertEqual(actual.body[-1].value.value,
                         'Return the mean across the feature dimension for each sample.')
        self.assertNotEqual(digest(normalised), digest(contract_tree(source)))
        method.body[0], method.body[1] = method.body[1], method.body[0]
        with self.assertRaises(AssertionError):
            normalise_presentation(tree, source)

    def test_reference_note_exception_is_one_exact_unreachable_position(self):
        source = REFERENCE_NOTE_SOURCE
        normalised = contract_tree(source)
        method = _scope(normalised, REFERENCE_NOTE_SCOPE)
        self.assertEqual(len(method.body), 4)
        self.assertIsInstance(method.body[2], ast.Return)
        self.assertEqual(method.body[3].value.value, REFERENCE_NOTE_TEXT)
        self.assertFalse(any(isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant)
                             and n.value.value == REFERENCE_NOTE_TEXT
                             for n in ast.walk(self.raw_tree(source))))
        for change in ('return', 'extra-statement'):
            tree = self.raw_tree(source)
            method = _scope(tree, REFERENCE_NOTE_SCOPE)
            if change == 'return':
                method.body[-1].value.elts.reverse()
            else:
                method.body.append(ast.Pass())
            with self.subTest(change=change), self.assertRaises(AssertionError):
                normalise_presentation(tree, source)
        tree = self.raw_tree(source)
        _scope(tree, ('GAMEDWithoutVeto', 'mapping')).body.append(
            ast.Expr(value=ast.Constant(value=REFERENCE_NOTE_TEXT)))
        self.assertNotEqual(digest(normalise_presentation(tree, source)), digest(normalised))

if __name__=='__main__':unittest.main()
