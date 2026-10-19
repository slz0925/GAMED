"""Check public Shell entrypoints without loading research frameworks or weights."""
import contextlib
import io
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import unittest
from unittest.mock import patch

import test_release_tools as routing
from scripts import run as launcher
import run_gamed as entry

ROOT = routing.ROOT
SCRIPTS = ROOT/'scripts'


def validation_program(dataset='fakeddit'):
    text = (SCRIPTS/('validate_'+dataset+'.sh')).read_text(encoding='utf-8')
    return text.split("<<'PY'\n", 1)[1].rsplit('\nPY', 1)[0]


class ValidationEntrypointTests(unittest.TestCase):
    dataset = 'fakeddit'

    setUp = routing.RoutingTests.setUp
    tearDown = routing.RoutingTests.tearDown
    configure = routing.RoutingTests.configure

    def invoke(self, arguments):
        # Execute the actual embedded preflight and real launcher plan builder.
        # Only the final research execution is intercepted.
        plans = []
        output = io.StringIO()
        errors = io.StringIO()
        previous = Path.cwd()
        try:
            os.chdir(ROOT)
            with patch.object(sys, 'argv', ['-', *arguments]), \
                 patch.object(launcher, 'execute_plan', side_effect=plans.append), \
                 patch.object(entry, 'initialise_runtime', side_effect=AssertionError('Unexpected model initialisation')), \
                 contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors):
                try:
                    exec(compile(validation_program(self.dataset), 'validate_'+self.dataset+'.sh:Python', 'exec'), {'__name__': '__main__'})
                    code = 0
                except SystemExit as error:
                    code = error.code
        finally:
            os.chdir(previous)
        return code, plans, output.getvalue(), errors.getvalue()

    def checkpoint(self):
        path = self.home/'trained weights'/'model $[]; name.pkl'
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(b'file-existence probe, never deserialised')
        return path

    def test_missing_empty_or_incomplete_checkpoint_never_launches(self):
        for args in ([], ['-checkpoint'], ['-checkpoint', ''], ['-checkpoint', '   ']):
            with self.subTest(args=args):
                code, plans, _, error = self.invoke(args)
                self.assertEqual(code, 2, error)
                self.assertEqual(plans, [])
                self.assertIn('checkpoint', error)

    def test_nonexistent_file_and_directory_never_launch(self):
        for path in (self.home/'absent.pkl', self.home):
            with self.subTest(path=path):
                code, plans, _, error = self.invoke(['-checkpoint', str(path)])
                self.assertEqual(code, 2, error)
                self.assertEqual(plans, [])
                self.assertIn('Checkpoint file not found', error)

    def test_valid_file_preserves_optional_arguments_and_defaults(self):
        args = ['-checkpoint', str(self.checkpoint()), '-output_file', 'output path/$[]; literal']
        code, plans, _, error = self.invoke(args)
        self.assertEqual(code, 0, error)
        self.assertEqual(len(plans), 1)
        self.assertEqual(plans[0]['argv'][1:1+len(args)], args)
        parsed = entry.build_parser().parse_args(plans[0]['argv'][1:])
        self.assertEqual((parsed.batch_size, parsed.epochs, parsed.val), (16, 100, 1))
        self.assertEqual(parsed.output_file, args[-1])

    def test_optional_overrides_reach_the_real_parser(self):
        args = ['-checkpoint', str(self.checkpoint()), '-batch_size', '7', '-epochs', '3',
                '-device', '2', '-finetune', '1']
        code, plans, _, error = self.invoke(args)
        self.assertEqual(code, 0, error)
        parsed = entry.build_parser().parse_args(plans[0]['argv'][1:])
        self.assertEqual((parsed.batch_size, parsed.epochs, parsed.device, parsed.finetune), (7, 3, 2, 1))

    def test_resource_root_mapping_checks_the_resolved_file(self):
        path = self.home/'assets'/'checkpoint'/'selected.pkl'
        path.parent.mkdir(parents=True)
        path.write_bytes(b'not loaded')
        self.configure(resource_root=str(self.home/'assets'))
        code, plans, _, error = self.invoke(['-checkpoint', 'checkpoint/selected.pkl'])
        self.assertEqual(code, 0, error)
        self.assertEqual(plans[0]['argv'][2], 'checkpoint/selected.pkl')

    def test_explicit_config_and_prefix_mapping(self):
        path = self.checkpoint()
        self.configure(prefix_map={'/shared/models': str(path.parent)})
        with patch.dict(os.environ, {'GAMED_CONFIG': str(self.config)}):
            code, plans, _, error = self.invoke(['-checkpoint', '/shared/models/'+path.name])
        self.assertEqual(code, 0, error)
        self.assertEqual(len(plans), 1)

    def test_relative_checkpoint_uses_repository_root(self):
        relative = os.path.relpath(self.checkpoint(), ROOT)
        code, plans, _, error = self.invoke(['-checkpoint', relative])
        self.assertEqual(code, 0, error)
        self.assertEqual(plans[0]['working_directory'], str(ROOT))

    def test_last_checkpoint_occurrence_is_validated(self):
        code, plans, _, error = self.invoke(['-checkpoint', str(self.checkpoint()), '-checkpoint', ''])
        self.assertEqual(code, 2, error)
        self.assertEqual(plans, [])

    def test_fixed_selectors_cannot_turn_validation_into_training(self):
        args = ['-checkpoint', str(self.checkpoint()), '-val', '0', '-network_arch', 'gamed_without_veto',
                '-train_dataset', ('yang' if self.dataset == 'fakeddit' else 'fakeddit'),
                '-test_dataset', ('yang' if self.dataset == 'fakeddit' else 'fakeddit')]
        code, plans, _, error = self.invoke(args)
        self.assertEqual(code, 0, error)
        parsed = entry.build_parser().parse_args(plans[0]['argv'][1:])
        self.assertEqual((parsed.val, parsed.network_arch, parsed.train_dataset, parsed.test_dataset),
                         (1, 'gamed', self.dataset, self.dataset))

    def test_help_needs_no_checkpoint_and_does_not_launch(self):
        code, plans, output, error = self.invoke(['--help'])
        self.assertEqual(code, 0, error)
        self.assertEqual(plans, [])
        self.assertIn('-checkpoint is required', output)

    def test_invalid_optional_arguments_never_launch(self):
        for extra in (['-batch_size', 'invalid'], ['--unknown'], ['--config', 'other.json']):
            with self.subTest(extra=extra):
                code, plans, _, error = self.invoke(['-checkpoint', str(self.checkpoint()), *extra])
                self.assertEqual(code, 2, error)
                self.assertEqual(plans, [])

    def test_missing_explicit_config_does_not_launch(self):
        with patch.dict(os.environ, {'GAMED_CONFIG': str(self.home/'absent.json')}), \
             patch.object(launcher, 'execute_plan') as execute:
            with self.assertRaises(FileNotFoundError):
                self.invoke(['-checkpoint', str(self.checkpoint())])
            execute.assert_not_called()


class YangValidationEntrypointTests(ValidationEntrypointTests):
    dataset = 'yang'


class ShellSourceTests(unittest.TestCase):
    def test_training_command_keeps_quoted_arguments_and_uses_entry_defaults(self):
        # Static token inspection is not a substitute for the Bash integration tests.
        for dataset in ('yang', 'fakeddit'):
            with self.subTest(dataset=dataset):
                text = (SCRIPTS/('train_'+dataset+'.sh')).read_text(encoding='utf-8')
                command = text[text.index('exec '):].replace('\\\n', '')
                tokens = shlex.split(command)
                self.assertEqual(tokens[:5], ['exec', '${PYTHON:-python}', '$PROJECT_ROOT/scripts/run.py', 'run_gamed.py', '$@'])
                self.assertIn('"$@"', command)
                parsed = entry.build_parser().parse_args(tokens[5:])
                self.assertEqual((parsed.network_arch, parsed.train_dataset, parsed.test_dataset, parsed.val),
                                 ('gamed', dataset, dataset, 0))
                self.assertEqual((parsed.batch_size, parsed.epochs, parsed.not_on_12), (16, 100, 0))
                self.assertEqual(set(tokens[5::2]), {'-train_dataset', '-test_dataset', '-network_arch', '-val'})

    def test_scripts_have_lf_and_validation_heredoc_is_literal(self):
        for name in ('train_yang.sh', 'train_fakeddit.sh', 'validate_yang.sh', 'validate_fakeddit.sh'):
            self.assertNotIn(b'\r', (SCRIPTS/name).read_bytes())
        for dataset in ('yang', 'fakeddit'):
            text = (SCRIPTS/('validate_'+dataset+'.sh')).read_text(encoding='utf-8')
            self.assertIn('exec "${PYTHON:-python}" - "$@" <<\'PY\'', text)
            compile(validation_program(dataset), 'validation preflight', 'exec')



def isolated_driver(home):
    driver = home/'entry driver.py'
    driver.write_text(
        'import json, os, runpy, sys\n'
        'from pathlib import Path\n'
        'root = Path('+repr(str(ROOT))+')\n'
        'sys.path.insert(0, str(root))\n'
        'original_run_path = runpy.run_path\n'
        'def capture(path, run_name):\n'
        '    assert Path(path).resolve() == root/"run_gamed.py", path\n'
        '    assert run_name == "__main__", run_name\n'
        '    print(json.dumps({"argv": sys.argv, "cwd": os.getcwd()}))\n'
        'runpy.run_path = capture\n'
        'sys.argv = sys.argv[1:]\n'
        'if sys.argv[0] == "-":\n'
        '    exec(compile(sys.stdin.read(), "<stdin>", "exec"), {"__name__": "__main__"})\n'
        'else:\n'
        '    assert Path(sys.argv[0]).resolve() == root/"scripts"/"run.py", sys.argv[0]\n'
        '    original_run_path(sys.argv[0], run_name="__main__")\n', encoding='utf-8')
    # Only stdlib and repository code are needed; bypass all startup hooks.
    return [Path(sys.executable).as_posix(), '-I', '-S', '-B', driver.as_posix()]


def competing_startup_environment(home):
    hook = home/'external python hooks'
    hook.mkdir(exist_ok=True)
    (hook/'sitecustomize.py').write_text(
        'raise RuntimeError("External startup hook must not run in the test driver")\n', encoding='utf-8')
    return dict(os.environ, PYTHONPATH=str(hook), PYTHONDONTWRITEBYTECODE='1')


class IsolatedDriverTests(unittest.TestCase):
    setUp = routing.RoutingTests.setUp
    tearDown = routing.RoutingTests.tearDown

    def test_file_dispatch_ignores_external_startup_hooks(self):
        arguments = ['-network_arch', 'gamed', '-train_dataset', 'yang', '-test_dataset', 'yang',
                     '-val', '0', '-output_file', 'literal $value; [*] path']
        result = subprocess.run([*isolated_driver(self.home), str(SCRIPTS/'run.py'), 'run_gamed.py', *arguments],
                                cwd=self.home, env=competing_startup_environment(self.home),
                                capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stderr, '')
        payload = json.loads(result.stdout)
        self.assertEqual(payload['argv'][1:], arguments)
        self.assertEqual(Path(payload['cwd']).resolve(), ROOT.resolve())

    def test_stdin_dispatch_checks_checkpoint_before_capture(self):
        path = self.home/'weight file $[];.pkl'
        path.write_bytes(b'not deserialised')
        for dataset in ('yang', 'fakeddit'):
            for arguments, expected in (([], 2), (['-checkpoint', str(self.home/'absent.pkl')], 2),
                                        (['-checkpoint', str(path), '-epochs', '3'], 0)):
                with self.subTest(arguments=arguments):
                    result = subprocess.run([*isolated_driver(self.home), '-', *arguments],
                                            input=validation_program(dataset), cwd=ROOT,
                                            env=competing_startup_environment(self.home),
                                            capture_output=True, text=True, timeout=15)
                    self.assertEqual(result.returncode, expected, result.stderr)
                    if expected:
                        self.assertEqual(result.stdout, '')
                        self.assertIn('checkpoint', result.stderr.lower())
                    else:
                        self.assertEqual(result.stderr, '')
                        payload = json.loads(result.stdout)
                        self.assertEqual(payload['argv'][1:1+len(arguments)], arguments)
                        args = entry.build_parser().parse_args(payload['argv'][1:])
                        self.assertEqual((args.epochs, args.val, args.network_arch, args.train_dataset, args.test_dataset),
                                         (3, 1, 'gamed', dataset, dataset))

    def test_capture_rejects_an_unexpected_research_target(self):
        result = subprocess.run([*isolated_driver(self.home), '-'],
                                input='import runpy\nrunpy.run_path("unexpected.py", run_name="__main__")\n',
                                cwd=ROOT, env=competing_startup_environment(self.home),
                                capture_output=True, text=True, timeout=15)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('AssertionError', result.stderr)
        self.assertEqual(result.stdout, '')


class BashIntegrationTests(unittest.TestCase):
    train_dataset = 'yang'
    validation_dataset = 'fakeddit'

    @classmethod
    def setUpClass(cls):
        cls.bash = os.environ.get('GAMED_TEST_BASH') or shutil.which('bash')
        if not cls.bash:
            raise unittest.SkipTest('Bash is not installed')
        try:
            probe = subprocess.run([cls.bash, '--noprofile', '--norc', '-c', 'exit 0'],
                                   capture_output=True, timeout=5)
        except (OSError, subprocess.TimeoutExpired) as error:
            raise unittest.SkipTest('Bash cannot start: '+str(error))
        if probe.returncode:
            raise unittest.SkipTest('Bash cannot start (exit '+str(probe.returncode)+')')

    setUp = routing.RoutingTests.setUp
    tearDown = routing.RoutingTests.tearDown

    def shell(self, name, arguments, environment=None):
        # Real Bash invokes this test-only interpreter wrapper through PYTHON.
        # The driver runs the actual launcher/preflight, then intercepts only run_gamed.py.
        command = isolated_driver(self.home)
        wrapper = self.home/'test python'
        wrapper.write_text('#!/usr/bin/env bash\nexec '+
                           ' '.join(shlex.quote(value) for value in command)+
                           ' "$@"\n', encoding='utf-8', newline='\n')
        wrapper.chmod(0o755)
        env = competing_startup_environment(self.home)
        env['PYTHON'] = wrapper.as_posix()
        if environment:
            env.update(environment)
        return subprocess.run([self.bash, '--noprofile', '--norc', str(SCRIPTS/name), *arguments],
                              cwd=self.home, env=env, capture_output=True, text=True, timeout=15)

    def test_bash_syntax(self):
        for name in ('train_'+self.train_dataset+'.sh', 'validate_'+self.validation_dataset+'.sh'):
            result = subprocess.run([self.bash, '--noprofile', '--norc', '-n', str(SCRIPTS/name)],
                                    capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_training_shell_forwards_literal_arguments_from_another_directory(self):
        value = 'output space; $literal [*]'
        result = self.shell('train_'+self.train_dataset+'.sh', ['-batch_size', '7', '-output_file', value,
                                            '-val', '1', '-network_arch', 'gamed_without_veto',
                                            '-train_dataset', self.validation_dataset, '-test_dataset', self.validation_dataset,
                                            '-epochs', '3', '-device', '2'])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stderr, '')
        payload = json.loads(result.stdout)
        args = entry.build_parser().parse_args(payload['argv'][1:])
        self.assertEqual((args.batch_size, args.output_file, args.val, args.epochs, args.device), (7, value, 0, 3, 2))
        self.assertEqual((args.network_arch, args.train_dataset, args.test_dataset), ('gamed', self.train_dataset, self.train_dataset))
        self.assertEqual(Path(payload['cwd']).resolve(), ROOT.resolve())

    def test_validation_shell_requires_file_and_forwards_options(self):
        for arguments in ([], ['-checkpoint', ''], ['-checkpoint', str(self.home/'absent.pkl')],
                          ['-checkpoint', str(self.home)]):
            with self.subTest(arguments=arguments):
                result = self.shell('validate_'+self.validation_dataset+'.sh', arguments)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertEqual(result.stdout, '')
                self.assertIn('checkpoint', result.stderr.lower())
        path = self.home/'weight file $[];.pkl'
        path.write_bytes(b'not deserialised')
        result = self.shell('validate_'+self.validation_dataset+'.sh', ['-checkpoint', str(path), '-epochs', '3',
                                                  '-val', '0', '-network_arch', 'gamed_without_veto',
                                                  '-train_dataset', self.train_dataset, '-test_dataset', self.train_dataset,
                                                  '-batch_size', '7', '-device', '2', '-output_file', 'literal $value; [*] path'])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stderr, '')
        payload = json.loads(result.stdout)
        args = entry.build_parser().parse_args(payload['argv'][1:])
        self.assertEqual((args.checkpoint, args.epochs, args.val), (str(path), 3, 1))
        self.assertEqual((args.network_arch, args.train_dataset, args.test_dataset), ('gamed', self.validation_dataset, self.validation_dataset))
        self.assertEqual(Path(payload['cwd']).resolve(), ROOT.resolve())
        self.assertEqual((args.batch_size, args.device, args.output_file),
                         (7, 2, 'literal $value; [*] path'))

    def test_validation_shell_resolves_configured_checkpoint(self):
        assets = self.home/'external assets'
        path = assets/'checkpoint'/'trained weight $[];.pkl'
        path.parent.mkdir(parents=True)
        path.write_bytes(b'not deserialised')
        config = self.home/'resource config.json'
        config.write_text(json.dumps({'resource_root': str(assets)}), encoding='utf-8')
        argument = 'checkpoint/'+path.name
        result = self.shell('validate_'+self.validation_dataset+'.sh', ['-checkpoint', argument],
                            {'GAMED_CONFIG': str(config)})
        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(result.stdout)
        args = entry.build_parser().parse_args(payload['argv'][1:])
        self.assertEqual(args.checkpoint, argument)
        self.assertEqual((args.val, args.train_dataset, args.test_dataset),
                         (1, self.validation_dataset, self.validation_dataset))
        path.unlink()
        result = self.shell('validate_'+self.validation_dataset+'.sh', ['-checkpoint', argument],
                            {'GAMED_CONFIG': str(config)})
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertEqual(result.stdout, '')
        self.assertIn('Checkpoint file not found', result.stderr)

    def test_shell_help_uses_real_python_without_checkpoint(self):
        # Run help through the real entry; argparse exits before research initialisation.
        env = dict(os.environ, PYTHON=Path(sys.executable).as_posix(), PYTHONDONTWRITEBYTECODE='1')
        for name in ('train_'+self.train_dataset+'.sh', 'validate_'+self.validation_dataset+'.sh'):
            with self.subTest(script=name):
                result = subprocess.run([self.bash, '--noprofile', '--norc', str(SCRIPTS/name), '--help'],
                                        cwd=self.home, env=env, capture_output=True, text=True, timeout=15)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stderr, '')
                self.assertIn('-checkpoint', result.stdout)
                self.assertIn('-batch_size', result.stdout)


class FakedditTrainingYangValidationBashTests(BashIntegrationTests):
    train_dataset = 'fakeddit'
    validation_dataset = 'yang'


if __name__ == '__main__':
    unittest.main()
