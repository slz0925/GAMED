"""Check resource routing, launcher behaviour and release safeguards."""
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import resource_paths as paths


class RoutingTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='gamed-tests-')
        self.home=Path(self.tmp.name)
        self.config=self.home/'paths.json'
        self.config.write_text('{}',encoding='utf-8')
        env={k:v for k,v in os.environ.items() if not k.startswith(('FND_','GAMED_'))}
        env['FND_CONFIG']=str(self.config)
        self.environ=patch.dict(os.environ,env,clear=True);self.environ.start()
        paths._load_config.cache_clear()

    def tearDown(self):
        paths._load_config.cache_clear();self.environ.stop();self.tmp.cleanup()

    def configure(self, **kwargs):
        self.config.write_text(json.dumps(kwargs),encoding='utf-8')
        paths._load_config.cache_clear()

    def test_no_override_preserves_exact_values(self):
        values=['checkpoint//fakeddit/a.pkl','./ernie/a','/root/autodl-tmp/ssd/yyh/tmp/fnd-bootstrap-master/ernie/x',Path('checkpoint/a.pth')]
        for value in values:self.assertIs(paths.resolve_read_path(value),value)

    def test_remote_model_id_is_not_a_local_path(self):
        self.configure(resource_root=str(self.home/'assets'))
        self.assertEqual(paths.resolve_read_path('bert-base-chinese'),'bert-base-chinese')

    def test_three_legacy_source_roots(self):
        self.configure(resource_root=str(self.home/'assets'))
        for prefix in paths._ROOT_PREFIXES:
            self.assertEqual(paths.resolve_read_path(prefix+'/ernie/model/config.json'),str(self.home/'assets/ernie/model/config.json'))

    def test_relative_and_project_absolute_resources(self):
        self.configure(resource_root=str(self.home/'assets'))
        for value in ('./dataset/Yang/yang_train.tsv',ROOT/'dataset/Yang/yang_train.tsv'):
            self.assertEqual(paths.resolve_read_path(value),str(self.home/'assets/dataset/Yang/yang_train.tsv'))

    def test_both_groupshare_forms(self):
        self.configure(groupshare_root=str(self.home/'shared'))
        for prefix in ('/home/groupshare','/groupshare'):
            self.assertEqual(paths.resolve_read_path(prefix+'/weibo/a.xlsx'),str(self.home/'shared/weibo/a.xlsx'))

    def test_prefix_is_anchored_and_longest_first(self):
        self.configure(prefix_map={'/data':str(self.home/'general'),'/data/special':str(self.home/'specific')})
        self.assertEqual(paths.resolve_read_path('/data/special/a'),str(self.home/'specific/a'))
        self.assertEqual(paths.resolve_read_path('/database/a'),'/database/a')

    def test_environment_override_has_priority(self):
        self.configure(resource_root='unused')
        os.environ['FND_RESOURCE_ROOT']=str(self.home/'override')
        self.assertEqual(paths.resolve_read_path('checkpoint/a.pth'),str(self.home/'override/checkpoint/a.pth'))

    def test_gamed_environment_names_override_compatible_fnd_values(self):
        self.configure(resource_root='json-value')
        os.environ['FND_RESOURCE_ROOT']=str(self.home/'old')
        os.environ['GAMED_RESOURCE_ROOT']=str(self.home/'new')
        self.assertEqual(paths.resolve_read_path('checkpoint/a.pth'),str(self.home/'new/checkpoint/a.pth'))

    def test_config_names_and_resource_spellings_resolve_to_the_same_file(self):
        assets=self.home/'assets'
        file=assets/'ernie/models--nghuyong--ernie-2.0-base-en/config.json'
        file.parent.mkdir(parents=True)
        file.write_text('{}')
        self.configure(resource_root=str(assets))
        old_path=paths.resolve_read_path(paths._ROOT_PREFIXES[0]+'/ernie/models--nghuyong--ernie-2.0-base-en/config.json')
        os.environ['GAMED_CONFIG']=str(self.config)
        os.environ['FND_CONFIG']=str(self.home/'unused-missing-config.json')
        new_path=paths.resolve_read_path('ernie/models--nghuyong--ernie-2.0-base-en/config.json')
        self.assertTrue(Path(old_path).samefile(new_path))
        self.assertEqual(Path(new_path).read_text(),'{}')

    def test_explicit_missing_gamed_config_fails(self):
        os.environ['GAMED_CONFIG']=str(self.home/'missing.json')
        with self.assertRaises(FileNotFoundError):paths.settings()

    def test_cache_keys_and_filenames_are_not_modified(self):
        self.configure(cache_root=str(self.home/'cache'))
        self.assertEqual(paths.resolve_read_path('./train_weibo.pickle'),str(self.home/'cache/train_weibo.pickle'))
        self.assertEqual(paths.resolve_read_path('a text cache key'),'a text cache key')

    def test_existing_local_checkpoint_takes_priority_over_external_resources(self):
        clean=self.home/'clean'
        artifact=clean/'checkpoint/fakeddit/new.pkl'
        artifact.parent.mkdir(parents=True)
        artifact.write_bytes(b'local artifact, not a model tensor')
        self.configure(resource_root=str(self.home/'original'))
        with patch.object(paths,'PROJECT_ROOT',clean):
            self.assertEqual(paths.resolve_read_path('checkpoint/fakeddit/new.pkl'),'checkpoint/fakeddit/new.pkl')
            self.assertEqual(paths.resolve_read_path('checkpoint/mae_pretrain_vit_base.pth'),str(self.home/'original/checkpoint/mae_pretrain_vit_base.pth'))

    def test_filelike_and_nonpaths_passthrough(self):
        value=io.BytesIO(b'x')
        self.assertIs(paths.resolve_read_path(value),value)
        self.assertEqual(paths.resolve_read_path(0),0)

    def test_read_routing_does_not_create_or_modify_files(self):
        self.configure(resource_root=str(self.home/'absent'))
        before={p.relative_to(self.home) for p in self.home.rglob('*')}
        paths.resolve_read_path('dataset/Fakeddit/a.tsv')
        self.assertEqual(before,{p.relative_to(self.home) for p in self.home.rglob('*')})

    def test_writes_do_not_route_back_to_original_resources(self):
        self.configure(resource_root=str(self.home/'original'),output_root=str(self.home/'results'))
        self.assertEqual(paths.resolve_write_path('checkpoint/fakeddit/a.pkl'),'checkpoint/fakeddit/a.pkl')
        self.assertEqual(paths.resolve_write_path('/home/groupshare/mae-main/example/yang/experiment.xlsx'),str(self.home/'results/examples/yang/experiment.xlsx'))

    def test_english_encoder_defaults_to_repository_resource(self):
        with patch.object(paths, 'PROJECT_ROOT', self.home), \
             patch.object(paths, '_ROOT_PREFIXES', (str(self.home/'absent legacy'),)):
            self.assertEqual(paths.resolve_english_encoder_path(), paths.ENGLISH_ENCODER_PATH)

    def test_english_encoder_preserves_resource_root_mapping(self):
        for root in ('relative assets', str(self.home/'absolute assets')):
            self.configure(resource_root=root)
            expected=paths.resolve_read_path(paths._ROOT_PREFIXES[0]+'/'+paths.ENGLISH_ENCODER_PATH)
            self.assertEqual(paths.resolve_english_encoder_path(), expected)

    def test_english_encoder_preserves_legacy_prefix_precedence(self):
        legacy=paths._ROOT_PREFIXES[0]+'/'+paths.ENGLISH_ENCODER_PATH
        self.configure(resource_root=str(self.home/'other assets'), prefix_map={
            paths._ROOT_PREFIXES[0]:str(self.home/'general'),
            legacy:str(self.home/'specific encoder')})
        self.assertEqual(paths.resolve_english_encoder_path(), str(self.home/'specific encoder'))

    def test_english_encoder_supports_relative_prefix_mapping(self):
        self.configure(prefix_map={paths.ENGLISH_ENCODER_PATH:str(self.home/'selected encoder')})
        self.assertEqual(paths.resolve_english_encoder_path(), str(self.home/'selected encoder'))

    def test_explicit_english_mapping_overrides_general_resource_root(self):
        selected=self.home/'selected encoder'
        general=self.home/'assets'
        for directory, marker in ((selected,'selected'), (general/paths.ENGLISH_ENCODER_PATH,'general')):
            directory.mkdir(parents=True)
            (directory/'config.json').write_text(json.dumps({'identity':marker}),encoding='utf-8')
        self.configure(resource_root=str(general), prefix_map={paths.ENGLISH_ENCODER_PATH:str(selected)})
        before=set(self.home.rglob('*'))
        actual=paths.resolve_english_encoder_path()
        self.assertEqual(actual,str(selected))
        self.assertEqual(actual,paths.resolve_read_path(paths.ENGLISH_ENCODER_PATH))
        self.assertEqual(json.loads((Path(actual)/'config.json').read_text())['identity'],'selected')
        self.assertEqual(set(self.home.rglob('*')),before)

    def test_explicit_english_mapping_precedes_environment_resource_roots(self):
        selected=self.home/'selected encoder'
        self.configure(resource_root='config root',prefix_map={paths.ENGLISH_ENCODER_PATH:str(selected)})
        for overrides in ({'FND_RESOURCE_ROOT':'old environment root'},
                          {'FND_RESOURCE_ROOT':'old environment root','GAMED_RESOURCE_ROOT':'new environment root'}):
            with self.subTest(overrides=overrides),patch.dict(os.environ,overrides):
                self.assertEqual(paths.resolve_english_encoder_path(),str(selected))
                expected_root=overrides.get('GAMED_RESOURCE_ROOT',overrides['FND_RESOURCE_ROOT'])
                self.assertEqual(paths.resolve_read_path('checkpoint/mae_pretrain_vit_base.pth'),
                                 str(paths.PROJECT_ROOT/expected_root/'checkpoint/mae_pretrain_vit_base.pth'))

    def test_english_mapping_uses_longest_relative_prefix_and_relative_destination(self):
        self.configure(resource_root='general assets',prefix_map={
            'ernie':'less specific', paths.ENGLISH_ENCODER_PATH:'selected encoder'})
        with patch.object(paths,'PROJECT_ROOT',self.home):
            self.assertEqual(paths.resolve_english_encoder_path(),str(self.home/'selected encoder'))
            self.configure(resource_root='general assets',prefix_map={'ernie':'selected models'})
            self.assertEqual(paths.resolve_english_encoder_path(),
                             str(self.home/'selected models'/Path(paths.ENGLISH_ENCODER_PATH).name))

    def test_legacy_explicit_mapping_wins_when_both_spellings_are_mapped(self):
        legacy=paths._ROOT_PREFIXES[0]+'/'+paths.ENGLISH_ENCODER_PATH
        self.configure(resource_root=str(self.home/'general'),prefix_map={
            paths.ENGLISH_ENCODER_PATH:str(self.home/'relative selection'),
            paths._ROOT_PREFIXES[0]:str(self.home/'legacy root'),
            legacy:str(self.home/'legacy selection')})
        self.assertEqual(paths.resolve_english_encoder_path(),str(self.home/'legacy selection'))

    def test_disabled_and_nonmatching_english_prefixes_do_not_override_root(self):
        general=self.home/'general'
        for prefixes in ({paths.ENGLISH_ENCODER_PATH:None}, {paths.ENGLISH_ENCODER_PATH:''},
                         {paths.ENGLISH_ENCODER_PATH+'-sibling':str(self.home/'wrong')},
                         {'erni':str(self.home/'wrong')}):
            with self.subTest(prefixes=prefixes):
                self.configure(resource_root=str(general),prefix_map=prefixes)
                self.assertEqual(paths.resolve_english_encoder_path(),str(general/paths.ENGLISH_ENCODER_PATH))

    def test_explicit_mapping_is_not_replaced_when_destination_is_missing(self):
        selected=self.home/'not downloaded yet'
        self.configure(resource_root=str(self.home/'general'),prefix_map={paths.ENGLISH_ENCODER_PATH:str(selected)})
        self.assertEqual(paths.resolve_english_encoder_path(),str(selected))
        self.assertFalse(selected.exists())

    def test_english_prefix_does_not_change_dataset_checkpoint_or_write_routing(self):
        general=self.home/'assets'
        self.configure(resource_root=str(general),output_root=str(self.home/'outputs'),
                       prefix_map={paths.ENGLISH_ENCODER_PATH:str(self.home/'selected encoder')})
        for name in ('dataset/Yang/yang_train.tsv','checkpoint/mae_pretrain_vit_base.pth',
                     'checkpoint/trained.pkl','ernie/another-model'):
            self.assertEqual(paths.resolve_read_path(name),str(general/name))
        self.assertEqual(paths.resolve_write_path('checkpoint/trained.pkl'),'checkpoint/trained.pkl')
        self.assertEqual(paths.resolve_write_path('/home/groupshare/mae-main/example/yang/result.xlsx'),
                         str(self.home/'outputs/examples/yang/result.xlsx'))

    def test_english_encoder_existing_legacy_installation_is_fallback(self):
        legacy_root=self.home/'legacy installation'
        legacy=legacy_root/paths.ENGLISH_ENCODER_PATH
        legacy.mkdir(parents=True)
        with patch.object(paths, 'PROJECT_ROOT', self.home/'project'), \
             patch.object(paths, '_ROOT_PREFIXES', (str(legacy_root),)):
            self.assertEqual(paths.resolve_english_encoder_path(),
                             str(legacy_root)+'/'+paths.ENGLISH_ENCODER_PATH)
            (paths.PROJECT_ROOT/paths.ENGLISH_ENCODER_PATH).mkdir(parents=True)
            self.assertEqual(paths.resolve_english_encoder_path(), paths.ENGLISH_ENCODER_PATH)

    def test_english_encoder_routing_has_no_file_or_rng_side_effects(self):
        import random
        self.configure(resource_root=str(self.home/'uncreated assets'))
        before=set(self.home.rglob('*'));state=random.getstate()
        paths.resolve_english_encoder_path()
        self.assertEqual(set(self.home.rglob('*')), before)
        self.assertEqual(random.getstate(), state)

    def test_all_environment_aliases_preserve_precedence_and_empty_override(self):
        for key, env in paths._GAMED_ENV_KEYS.items():
            old=env.replace('GAMED_', 'FND_', 1)
            with self.subTest(key=key), patch.dict(os.environ, {old:'old value', env:'new value'}):
                self.assertEqual(paths.settings()[key], 'new value')
                os.environ[env]=''
                self.assertIsNone(paths.settings()[key])
                del os.environ[env]
                self.assertEqual(paths.settings()[key], 'old value')

    def test_launcher_ignores_unrelated_output_root_without_creating_directories(self):
        from scripts import run as launcher
        blocker=self.home/'not a directory';blocker.write_text('preserve this file')
        for output in (str(blocker), str(self.home/'uncreated outputs')):
            self.configure(output_root=output)
            args=['-network_arch','gamed','-train_dataset','yang','-test_dataset','yang',
                  '-val','1','-output_file','chosen output with spaces']
            plan=launcher.build_plan('run_gamed.py',args)
            before=set(self.home.rglob('*'));cwd=Path.cwd();seen=[]
            try:
                with patch.object(sys,'argv',list(sys.argv)),patch.object(sys,'path',list(sys.path)), \
                     patch.object(launcher.runpy,'run_path',side_effect=lambda p,run_name:seen.append((p,run_name,Path.cwd(),list(sys.argv)))):
                    launcher.execute_plan(plan)
                target=str(ROOT/'run_gamed.py')
                self.assertEqual(seen,[(target,'__main__',ROOT,[target,*args])])
                self.assertEqual(set(self.home.rglob('*')),before)
                self.assertEqual(blocker.read_text(),'preserve this file')
            finally:
                os.chdir(cwd)

    def test_compatibility_cli_flags_are_accepted_but_not_advertised(self):
        import run_gamed
        parser=run_gamed.build_parser()
        flags={'-is_filter':('2',2),'-duplicate_fake_times':('3',3),
               '-is_sample_positive':('0.25',0.25),'-not_on_12':('1',1)}
        args=['-network_arch','gamed','-train_dataset','yang','-test_dataset','yang']
        for flag,(text,value) in flags.items():
            self.assertNotIn(flag,parser.format_help())
            parsed=parser.parse_args(args+[flag,text])
            self.assertEqual(getattr(parsed,flag[1:]),value)
        defaults=parser.parse_args(args)
        self.assertEqual((defaults.is_filter,defaults.duplicate_fake_times,
                          defaults.is_sample_positive,defaults.not_on_12),(0,0,1.0,0))

    def test_models_and_tokenizer_share_the_english_resource_resolver(self):
        import ast
        for source in ('models/gamed.py','models/gamed_without_veto.py','run_gamed.py'):
            tree=ast.parse((ROOT/source).read_text(encoding='utf-8'))
            calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call)
                   and isinstance(n.func,ast.Name) and n.func.id=='resolve_english_encoder_path']
            self.assertEqual(len(calls),1,source)
            self.assertEqual((calls[0].args,calls[0].keywords),([],[]))

    def test_launcher_dry_run_is_cwd_independent_and_does_not_import_torch(self):
        before={p.relative_to(self.home) for p in self.home.rglob('*')}
        result=subprocess.run([sys.executable,'-B',str(ROOT/'scripts/run.py'),'--dry-run','run_gamed.py','-network_arch','gamed','-train_dataset','yang','-test_dataset','yang','-val','0'],cwd=self.home,env=os.environ,capture_output=True,text=True,encoding='utf-8')
        self.assertEqual(result.returncode,0,result.stderr)
        plan=json.loads(result.stdout)
        self.assertEqual(plan['argv'],['run_gamed.py','-network_arch','gamed','-train_dataset','yang','-test_dataset','yang','-val','0'])
        self.assertEqual(plan['working_directory'],str(ROOT))
        self.assertEqual(before,{p.relative_to(self.home) for p in self.home.rglob('*')})


if __name__=='__main__':unittest.main()
