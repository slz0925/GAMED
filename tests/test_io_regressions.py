"""Execute actual filesystem blocks without importing CUDA research entries.

Only model/tensor saving is stubbed. Routing, directory creation and image
enumeration use temporary files. No weights, datasets or training are loaded.
"""
import ast
import datetime
import importlib.util
import os
import random
from pathlib import Path
from types import SimpleNamespace
import struct
import unittest
from unittest.mock import patch
import zlib

import test_release_tools as release_tests

ROOT, paths = release_tests.ROOT, release_tests.paths

def png_chunk(kind, data):
    return struct.pack('>I', len(data))+kind+data+struct.pack('>I', zlib.crc32(kind+data))

PNG = (b'\x89PNG\r\n\x1a\n'
       + png_chunk(b'IHDR', struct.pack('>IIBBBBB', 1, 1, 8, 2, 0, 0, 0))
       + png_chunk(b'IDAT', zlib.compress(b'\0\xff\0\0'))+png_chunk(b'IEND', b''))


def tree(filename):
    return ast.parse((ROOT/filename).read_text(encoding='utf-8-sig'))


def execute(statements, namespace):
    module = ast.fix_missing_locations(ast.Module(body=statements, type_ignores=[]))
    exec(compile(module, '<research-io-block>', 'exec'), namespace)


def checkpoint_block(filename):
    for node in ast.walk(tree(filename)):
        body = getattr(node, 'body', None)
        if not isinstance(body, list):
            continue
        for index, statement in enumerate(body):
            if isinstance(statement, ast.Assign) and any(
                    isinstance(target, ast.Name) and target.id == 'best_validate_dir'
                    for target in statement.targets):
                block = body[index-1:index+2]
                assert ast.unparse(block[-1].value.func) == 'torch.save'
                return block
    raise AssertionError('Checkpoint saving block not found: '+filename)






class IORegressionTests(unittest.TestCase):
    setUp = release_tests.RoutingTests.setUp
    tearDown = release_tests.RoutingTests.tearDown
    configure = release_tests.RoutingTests.configure

    def save_checkpoint(self, filename, output):
        writes = []
        state = object()
        def save(value, destination):
            self.assertIs(value, state)
            Path(destination).write_bytes(b'stub-state-dict')
            writes.append(Path(destination))
        execute(checkpoint_block(filename), {
            'os': os, '_fnd_read_path': paths.resolve_read_path,
            'args': SimpleNamespace(output_file=str(output)),
            'setting': {'train_dataname': 'yang'}, 'epoch': 0,
            'datetime': datetime, 'best_validate_acc': 0.5,
            'model': SimpleNamespace(state_dict=lambda: state),
            'torch': SimpleNamespace(save=save),
        })
        self.assertEqual(len(writes), 1)
        self.assertEqual(writes[0].parent, Path(output)/'yang')
        self.assertTrue(writes[0].is_file())

    def test_external_checkpoint_directory_does_not_hide_missing_local_directory(self):
        archive = self.home/'archive'
        (archive/'checkpoint/yang').mkdir(parents=True)
        clean = self.home/'clean'
        clean.mkdir()
        self.configure(resource_root=str(archive), output_root=str(self.home/'unrelated-output'))
        previous = Path.cwd()
        try:
            os.chdir(clean)
            with patch.object(paths, 'PROJECT_ROOT', clean):
                for filename in ('run_gamed.py',):
                    with self.subTest(entry=filename):
                        # Separate local roots make both entries encounter a missing directory.
                        output = 'checkpoint/'+filename.removesuffix('.py')
                        (archive/output).mkdir(parents=True)
                        self.assertFalse(Path(output).exists())
                        self.save_checkpoint(filename, output)
                        self.assertFalse((archive/output/'yang').exists())
        finally:
            os.chdir(previous)

    def test_custom_nested_checkpoint_output_creates_dataset_parent(self):
        self.configure(resource_root=str(self.home/'archive'))
        for filename in ('run_gamed.py',):
            with self.subTest(entry=filename):
                self.save_checkpoint(filename, self.home/filename/'nested/results')

    def test_unconfigured_checkpoint_target_and_existing_directory_are_preserved(self):
        for filename in ('run_gamed.py',):
            with self.subTest(entry=filename):
                output = self.home/filename/'checkpoint'
                (output/'yang').mkdir(parents=True)
                sentinel = output/'yang/old.pkl'
                sentinel.write_bytes(b'old-checkpoint')
                self.save_checkpoint(filename, output)
                self.assertEqual(sentinel.read_bytes(), b'old-checkpoint')





    def image_functions(self, include_reader=False):
        names = {'is_image_file', '_get_paths_from_images', 'get_image_paths'}
        if include_reader:
            names.add('read_img')
        namespace = {'os':os, '_fnd_read_path':paths.resolve_read_path,
                     'IMG_EXTENSIONS':['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tiff', '.tif']}
        if include_reader:
            import numpy as np
            from PIL import Image
            namespace.update(np=np, Image=Image, cv2=SimpleNamespace(IMREAD_COLOR=1))
        functions = [node for node in tree('data/util.py').body
                     if isinstance(node, ast.FunctionDef) and node.name in names]
        self.assertEqual({f.name for f in functions}, names)
        execute(functions, namespace)
        return namespace

    def prepare_images(self):
        directory = self.home/'assets/dataset/images'
        (directory/'nested').mkdir(parents=True)
        (directory/'a.png').write_bytes(PNG)
        (directory/'nested/b.PNG').write_bytes(PNG)
        (directory/'ignored.txt').write_text('not an image')
        return directory

    def test_mapped_image_enumeration_returns_readable_sorted_nested_paths(self):
        directory = self.prepare_images()
        self.configure(resource_root=str(self.home/'assets'))
        clean = self.home/'clean'
        clean.mkdir()
        with patch.object(paths, 'PROJECT_ROOT', clean):
            found, sizes = self.image_functions()['get_image_paths']('dataset/images')
        self.assertEqual(found, [str(directory/'a.png'), str(directory/'nested/b.PNG')])
        self.assertIsNone(sizes)
        self.assertTrue(all(Path(p).read_bytes() == PNG for p in found))

    def test_unconfigured_image_enumeration_retains_original_paths(self):
        directory = self.prepare_images()
        found = self.image_functions()['_get_paths_from_images'](str(directory))
        self.assertEqual(found, [str(directory/'a.png'), str(directory/'nested/b.PNG')])

    @unittest.skipUnless(importlib.util.find_spec('numpy') and importlib.util.find_spec('PIL'),
                         'Optional integration test requires NumPy and Pillow')
    def test_mapped_images_are_decoded_by_the_actual_read_img_function(self):
        directory = self.prepare_images()
        self.configure(prefix_map={'/old/images':str(directory)})
        functions = self.image_functions(include_reader=True)
        for filename in functions['_get_paths_from_images']('/old/images'):
            image = functions['read_img'](filename)
            self.assertEqual(image.shape[:2], (1, 1))
            self.assertEqual(str(image.dtype), 'float32')
            import numpy as np
            np.testing.assert_array_equal(image, np.array([[[1., 0., 0.]]], dtype=np.float32))

    def pixel_functions(self):
        import numpy as np
        from PIL import Image
        namespace = {'np': np, 'Image': Image, '_fnd_read_path': paths.resolve_read_path,
                     'cv2': SimpleNamespace(IMREAD_COLOR=1)}
        names = {'read_img', 'channel_convert', 'bgr2ycbcr'}
        execute([node for node in tree('data/util.py').body
                 if isinstance(node, ast.FunctionDef) and node.name in names], namespace)
        return namespace

    @unittest.skipUnless(importlib.util.find_spec('numpy') and importlib.util.find_spec('PIL'),
                         'Pixel regression tests require NumPy and Pillow')
    def test_exact_pixels_scaling_alpha_and_greyscale(self):
        import numpy as np
        from PIL import Image
        functions = self.pixel_functions()
        cases = {
            'RGB': np.array([[[0, 64, 255], [128, 192, 32]]], dtype=np.uint8),
            'RGBA': np.array([[[0, 64, 255, 20], [128, 192, 32, 240]]], dtype=np.uint8),
            'L': np.array([[0, 128, 255]], dtype=np.uint8),
            'LA': np.array([[[64, 255], [128, 0]]], dtype=np.uint8),
        }
        for mode, pixels in cases.items():
            with self.subTest(mode=mode):
                path = self.home/(mode + '.png')
                Image.fromarray(pixels).save(path)
                actual = functions['read_img'](str(path))
                expected = pixels.astype(np.float32) / np.float32(255)
                if mode == 'RGBA':
                    expected = expected[:, :, :3]
                elif mode == 'L':
                    expected = expected[:, :, None]
                np.testing.assert_array_equal(actual, expected)
                self.assertEqual(actual.dtype, np.float32)

    @unittest.skipUnless(importlib.util.find_spec('numpy') and importlib.util.find_spec('PIL'),
                         'Pixel regression tests require NumPy and Pillow')
    def test_channel_conversion_preserves_existing_values_and_two_channel_quantisation(self):
        import numpy as np
        functions = self.pixel_functions()
        convert = functions['channel_convert']
        rgb = np.array([[[0.25, 0.5, 1.0]]], dtype=np.float32)
        result = convert(3, 'RGB', [rgb])[0]
        np.testing.assert_array_equal(result, rgb)
        self.assertIs(result, rgb)
        two = np.array([[[0.25, 1.0], [0.5, 0.0]]], dtype=np.float32)
        result = convert(2, 'RGB', [two])[0]
        np.testing.assert_array_equal(result, np.array([[[0, 1, 0], [0, 0, 0]]], dtype=np.uint8))
        self.assertEqual(result.dtype, np.uint8)
        # BGR primary colours, evaluated by the real NumPy-only luminance path.
        bgr = np.eye(3, dtype=np.uint8).reshape(1, 3, 3) * 255
        luminance = convert(3, 'y', [bgr])[0]
        np.testing.assert_array_equal(luminance, np.array([[[41], [145], [81]]], dtype=np.uint8))

    @unittest.skipUnless(importlib.util.find_spec('numpy') and importlib.util.find_spec('PIL'),
                         'Pixel regression tests require NumPy and Pillow')
    def test_actual_dataset_getitem_reverses_rgb_to_bgr_chw(self):
        import numpy as np
        from PIL import Image
        functions = self.pixel_functions()
        pixels = np.array([[[255, 64, 0], [16, 128, 240]],
                           [[32, 96, 224], [48, 160, 208]]], dtype=np.uint8)
        Image.fromarray(pixels).save(self.home/'pixels.png')
        expected = pixels[:, :, [2, 1, 0]].transpose(2, 0, 1).astype(np.float32) / np.float32(255)

        class TensorAdapter:
            def __init__(self, array):
                self.array = array
            def float(self):
                return self.array.astype(np.float32)

        def resize_same_size(array, size, interpolation):
            # No OpenCV interpolation claim: this isolates the real loader's
            # channel/axis operations using an already correctly sized image.
            self.assertEqual(size, (2, 2))
            self.assertEqual(array.shape[:2], (2, 2))
            self.assertEqual(interpolation, 1)
            return array

        for name in ('Fakeddit_dataset', 'Yang_dataset'):
            namespace = dict(functions)
            namespace.update(data=SimpleNamespace(Dataset=object),
                             random=random.Random(25),
                             np=SimpleNamespace(random=np.random.RandomState(25), copy=np.copy,
                                                transpose=np.transpose, ascontiguousarray=np.ascontiguousarray),
                             cv2=SimpleNamespace(resize=resize_same_size, INTER_LINEAR=1),
                             torch=SimpleNamespace(from_numpy=TensorAdapter))
            declaration = next(node for node in tree('data/'+name+'.py').body if isinstance(node, ast.ClassDef))
            execute([declaration], namespace)
            dataset = object.__new__(namespace[name])
            dataset.image_size = 2
            dataset.image_path = str(self.home)
            dataset.image_listdir = {}
            dataset.label_dict = [{'images': 'pixels.png', 'label': 1, 'content': ['synthetic']}]
            with self.subTest(dataset=name):
                item = dataset[0]
                np.testing.assert_array_equal(item[1][0], expected)
                self.assertEqual(item[1][2], 1)


if __name__ == '__main__':
    unittest.main()
