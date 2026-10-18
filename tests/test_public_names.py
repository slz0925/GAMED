"""Public scope and invocation checks without model initialisation."""
import ast
import contextlib
import importlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import importlib.util
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import test_release_tools as routing
ROOT=routing.ROOT
import run_gamed as entry
from scripts import run as launcher

class PublicScopeTests(unittest.TestCase):
    setUp=routing.RoutingTests.setUp
    tearDown=routing.RoutingTests.tearDown
    configure=routing.RoutingTests.configure

    def args(self,variant='gamed',dataset='yang'):
        return ['-network_arch',variant,'-train_dataset',dataset,'-test_dataset',dataset,'-val','0']

    def test_two_variants_and_two_datasets_forward_every_argument(self):
        for variant in ('gamed','gamed_without_veto'):
            for dataset in ('fakeddit','yang'):
                args=self.args(variant,dataset)+['-output_file','a path/checkpoint']
                self.assertEqual(entry.validate_arguments(args),args)

    def test_invalid_names_missing_arguments_and_unknown_options_fail(self):
        bad=[[],['-train_dataset','yang'],self.args('unknown_model'),self.args('typo'),self.args(dataset='unknown_dataset'),
             self.args()+['--unsupported-option','19983'],self.args()+['--unknown-setting','1'],
             ['-network_arch','gamed','-train_dataset','yang','-test_dataset','unknown_dataset']]
        for args in bad:
            with self.subTest(args=args),contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):entry.validate_arguments(args)

    def test_selector_resolves_only_the_matching_real_model_class(self):
        names={'gamed':('models.gamed','GAMED'),
               'gamed_without_veto':('models.gamed_without_veto','GAMEDWithoutVeto')}
        modules={};classes={}
        for variant,(module,symbol) in names.items():
            stub=ModuleType(module);cls=type(variant,(),{});setattr(stub,symbol,cls)
            modules[module]=stub;classes[variant]=cls
        with patch.dict(sys.modules,modules):
            for variant in names:self.assertIs(entry.resolve_model_class(variant),classes[variant])
            with self.assertRaises(ValueError):entry.resolve_model_class('unknown_model')

    def test_launcher_has_only_the_public_entry(self):
        self.assertEqual(launcher.ENTRIES,('run_gamed.py',))
        with self.assertRaises(ValueError):launcher.build_plan('unknown_entry.py',self.args())

    def test_valid_plan_runs_once_from_project_root_with_exact_argv(self):
        args=self.args()+['-batch_size','32']
        plan=launcher.build_plan('run_gamed.py',args)
        before=Path.cwd();seen=[]
        try:
            os.chdir(self.home)
            with patch.object(sys,'argv',list(sys.argv)),patch.object(sys,'path',list(sys.path)), \
                 patch.object(launcher.runpy,'run_path',side_effect=lambda p,run_name:seen.append((p,run_name,Path.cwd(),list(sys.argv)))):
                launcher.execute_plan(plan)
            target=str(ROOT/plan['implementation_entry'])
            self.assertEqual(seen,[(target,'__main__',ROOT,[target,*args])])
        finally:os.chdir(before)

    def test_help_and_invalid_direct_cli_do_not_load_frameworks(self):
        for args,code in [(['--help'],0),([],2), (self.args(dataset='Mix'),2)]:
            result=subprocess.run([sys.executable,'-B',str(ROOT/'run_gamed.py'),*args],cwd=self.home,
                                  env=os.environ,capture_output=True,text=True)
            self.assertEqual(result.returncode,code,result.stderr)
            self.assertNotIn('ModuleNotFoundError',result.stderr)

    def test_dry_run_is_lightweight_and_creates_no_files(self):
        before=set(self.home.rglob('*'))
        result=subprocess.run([sys.executable,'-B',str(ROOT/'scripts/run.py'),'--dry-run','run_gamed.py',*self.args()],
                              cwd=self.home,env=os.environ,capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(json.loads(result.stdout)['argv'],['run_gamed.py',*self.args()])
        self.assertEqual(before,set(self.home.rglob('*')))

    def test_cli_initialises_runtime_once_before_gpu_and_main(self):
        before=Path.cwd();events=[]
        fake=SimpleNamespace(cuda=SimpleNamespace(set_device=lambda d:events.append(('gpu',d))))
        try:
            with patch.object(entry,'initialise_runtime',side_effect=lambda:events.append('runtime')), \
                 patch.object(entry,'torch',fake,create=True),patch.object(entry,'main',side_effect=lambda a:events.append(('main',a.network_arch))):
                entry.cli(self.args())
            self.assertEqual(events,['runtime',('gpu',0),('main','gamed')])
            self.assertEqual(Path.cwd(),ROOT)
        finally:os.chdir(before)

    def test_entry_import_is_framework_free(self):
        code="import sys; import run_gamed; assert 'torch' not in sys.modules; assert 'transformers' not in sys.modules"
        result=subprocess.run([sys.executable,'-B','-c',code],cwd=ROOT,env=os.environ,capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)

    def test_public_models_are_direct_module_definitions(self):
        for file, name in (('gamed.py','GAMED'),('gamed_without_veto.py','GAMEDWithoutVeto')):
            tree=ast.parse((ROOT/'models'/file).read_text())
            definition=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==name)
            self.assertEqual([ast.unparse(base) for base in definition.bases],['nn.Module'])

    @unittest.skipUnless(importlib.util.find_spec('numpy') and importlib.util.find_spec('PIL'),
                         'Tokeniser import-boundary check requires NumPy/Pillow')
    def test_actual_runtime_setup_loads_only_the_english_tokeniser_once(self):
        from PIL import ImageFile
        self.configure(resource_root=str(self.home/'assets'))
        calls=[]
        def module(name):
            value=ModuleType(name);value.__path__=[]
            return value
        torch=module('torch');nn=module('torch.nn');torch.nn=nn
        auto=module('torch.autograd');auto.Variable=object
        utils=module('torch.utils');data=module('torch.utils.data');data.DataLoader=object
        utils.data=data;torch.utils=utils
        transformers=module('transformers')
        tokenizer=object()
        transformers.AutoTokenizer=SimpleNamespace(from_pretrained=lambda path:(calls.append(path),tokenizer)[1])
        modules={'torch':torch,'torch.nn':nn,'torch.autograd':auto,'torch.utils':utils,
                 'torch.utils.data':data,'pytorch_warmup':module('pytorch_warmup'),
                 'transformers':transformers,'models_mae':module('models_mae')}
        with patch.dict(entry.__dict__,{}),patch.dict(sys.modules,modules), \
             patch.object(ImageFile,'LOAD_TRUNCATED_IMAGES',False):
            entry.initialise_runtime()
            self.assertIs(entry.token_uncased_fakeddit,tokenizer)
            self.assertTrue(ImageFile.LOAD_TRUNCATED_IMAGES)
        self.assertEqual(calls,[str(self.home/'assets/ernie/models--nghuyong--ernie-2.0-base-en')])

    def test_runtime_preloads_backbone_imports_before_tokenisation_and_research_seeds(self):
        tree=ast.parse((ROOT/'run_gamed.py').read_text())
        setup=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='initialise_runtime')
        preload=next(i for i,n in enumerate(setup.body) if isinstance(n,ast.Import) and any(a.name=='models_mae' for a in n.names))
        tokeniser=next(i for i,n in enumerate(setup.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='token_uncased_fakeddit' for t in n.targets))
        self.assertLess(preload,tokeniser)
        cli=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='cli')
        calls=[ast.unparse(n.value.func) for n in cli.body if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call)]
        self.assertLess(calls.index('initialise_runtime'),calls.index('main'))

if __name__=='__main__':unittest.main()
