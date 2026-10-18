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

    def test_launcher_dry_run_is_cwd_independent_and_does_not_import_torch(self):
        before={p.relative_to(self.home) for p in self.home.rglob('*')}
        result=subprocess.run([sys.executable,'-B',str(ROOT/'scripts/run.py'),'--dry-run','run_gamed.py','-network_arch','gamed','-train_dataset','yang','-test_dataset','yang','-val','0'],cwd=self.home,env=os.environ,capture_output=True,text=True,encoding='utf-8')
        self.assertEqual(result.returncode,0,result.stderr)
        plan=json.loads(result.stdout)
        self.assertEqual(plan['argv'],['run_gamed.py','-network_arch','gamed','-train_dataset','yang','-test_dataset','yang','-val','0'])
        self.assertEqual(plan['working_directory'],str(ROOT))
        self.assertEqual(before,{p.relative_to(self.home) for p in self.home.rglob('*')})


if __name__=='__main__':unittest.main()
