"""Tensor tests for Veto Voting, without encoder imports or model resources."""
import ast
import copy
import math
from pathlib import Path
import unittest

try:
    import torch
except ModuleNotFoundError as error:
    if error.name != 'torch':
        raise
    torch = None

ROOT = Path(__file__).resolve().parents[1]


def source_tree(path):
    return ast.parse((ROOT / path).read_text(encoding='utf-8'))


def model_class():
    return next(n for n in source_tree('models/gamed.py').body
                if isinstance(n, ast.ClassDef) and n.name == 'GAMED')


def method(name):
    return next(n for n in model_class().body
                if isinstance(n, ast.FunctionDef) and n.name == name)


def execute(nodes, namespace):
    tree = ast.fix_missing_locations(ast.Module(body=copy.deepcopy(nodes), type_ignores=[]))
    exec(compile(tree, '<actual-source-fragment>', 'exec'), namespace)


class VetoInterfaceSourceTests(unittest.TestCase):
    def test_forward_uses_ip_is_t_mm_and_returns_the_voted_logit(self):
        forward = method('forward')
        calls = [n for n in ast.walk(forward) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Attribute) and n.func.attr == 'veto_voting']
        self.assertEqual(len(calls), 1)
        expected = ast.parse('self.veto_voting(mix_output, torch.cat('
                             '(ip_output, is_output, t_output, mm_aux_output), dim=1))',
                             mode='eval').body
        self.assertEqual(ast.dump(calls[0]), ast.dump(expected))
        assignment = next(n for n in forward.body if isinstance(n, ast.Assign) and n.value is calls[0])
        self.assertEqual(ast.dump(assignment.targets[0]), ast.dump(ast.Name(id='mix_output', ctx=ast.Store())))
        self.assertFalse(any(isinstance(n, ast.Attribute) and n.attr == 'training'
                             for n in ast.walk(forward)))
        returned = [n.value for n in ast.walk(forward) if isinstance(n, ast.Return)
                    and isinstance(n.value, ast.Tuple) and len(n.value.elts) in (6, 7)]
        self.assertEqual(len(returned), 2)
        self.assertTrue(all(isinstance(n.elts[0], ast.Name) and n.elts[0].id == 'mix_output'
                            for n in returned))

    def test_voting_has_no_detach_scalar_extraction_or_parameter_registration(self):
        helper = method('veto_voting')
        self.assertEqual([ast.unparse(n) for n in helper.decorator_list], ['staticmethod'])
        forbidden = {'detach', 'item', 'tensor', 'Parameter', 'register_buffer', 'register_parameter'}
        for node in ast.walk(helper):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                self.assertNotIn(node.func.attr, forbidden)


@unittest.skipIf(torch is None, 'PyTorch is required for Veto tensor and gradient tests')
class VetoTensorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Execute the actual method body; do not recreate its voting logic here.
        helper = copy.deepcopy(method('veto_voting'))
        helper.decorator_list = []
        namespace = {'torch': torch}
        execute([helper], namespace)
        cls.vote = staticmethod(namespace['veto_voting'])

    def logits(self, probabilities, dtype=None, grad=False):
        value = torch.logit(torch.tensor(probabilities, dtype=dtype or torch.float64))
        return value.requires_grad_(grad)

    def assert_probability(self, mixture, experts, expected):
        m, e = self.logits([[mixture]]), self.logits([experts])
        actual = self.vote(m, e)
        torch.testing.assert_close(actual.sigmoid(), torch.tensor([[expected]], dtype=m.dtype),
                                   rtol=1e-12, atol=1e-12)
        return m, e, actual

    def test_no_rule_preserves_exact_logit(self):
        m, e, actual = self.assert_probability(.4, [.2, .3, .6, .7], .4)
        self.assertTrue(torch.equal(actual, m))

    def test_veto1_replaces_an_opposing_mixture(self):
        self.assert_probability(.2, [.98, .2, .3, .4], .98)

    def test_veto1_also_applies_to_same_class(self):
        self.assert_probability(.7, [.98, .6, .7, .8], .98)

    def test_veto1_requires_exceeding_mixture(self):
        self.assert_probability(.99, [.98, .6, .7, .8], .99)
        self.assert_probability(.98, [.98, .6, .7, .8], .98)

    def test_veto1_requires_high_threshold(self):
        self.assert_probability(.3, [.94, .6, .7, .8], .3)

    def test_veto2_averages_with_highest_expert(self):
        self.assert_probability(.2, [.01, .2, .3, .8], .5)

    def test_low_minority_does_not_trigger(self):
        self.assert_probability(.2, [.01, .6, .7, .8], .2)

    def test_two_two_tie_does_not_trigger_even_if_mixture_breaks_tie(self):
        self.assert_probability(.2, [.01, .02, .8, .9], .2)

    def test_probability_half_counts_as_positive(self):
        self.assert_probability(.2, [.01, .2, .5, .5], .2)

    def test_multiple_veto1_selects_highest_probability(self):
        self.assert_probability(.2, [.97, .99, .98, .7], .99)

    def test_each_low_majority_expert_updates_once(self):
        # Three averages: (initial + 7 * maximum) / 8, not one average.
        self.assert_probability(.1, [.01, .02, .03, .8], .7125)

    def test_high_and_low_rules_in_same_sample(self):
        self.assert_probability(.1, [.01, .02, .03, .98], .98)
        self.assert_probability(.1, [.98, .01, .02, .03], .98)

    def test_all_negative_keeps_a_nonempty_maximum_set(self):
        # Four averages with maximum=.04: (.8 + 15*.04) / 16.
        self.assert_probability(.8, [.01, .02, .03, .04], .0875)

    def test_maximum_excludes_mixture(self):
        self.assert_probability(.9, [.01, .2, .3, .7], .8)

    def test_strict_threshold_boundaries_in_logit_space(self):
        for dtype in (torch.float32, torch.float64):
            m = self.logits([[.2]], dtype)
            for probability, rest in ((.05, [.2, .3, .8]), (.95, [.2, .3, .8])):
                e = self.logits([[probability, *rest]], dtype)
                self.assertTrue(torch.equal(self.vote(m, e), m))
                below, above = e.clone(), e.clone()
                below[:, 0] = torch.nextafter(e[:, 0], e.new_tensor(-math.inf))
                above[:, 0] = torch.nextafter(e[:, 0], e.new_tensor(math.inf))
                if probability == .05:
                    torch.testing.assert_close(self.vote(m, below).sigmoid(), m.new_tensor([[.5]]))
                    self.assertTrue(torch.equal(self.vote(m, above), m))
                else:
                    self.assertTrue(torch.equal(self.vote(m, below), m))
                    torch.testing.assert_close(self.vote(m, above), above[:, :1])

    def test_batch_shape_dtype_device_and_input_immutability(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            m = self.logits([[.2], [.1], [.4]], dtype)
            e = self.logits([[.98, .2, .3, .4], [.01, .02, .03, .8], [.2, .3, .6, .7]], dtype)
            before_m, before_e = m.clone(), e.clone()
            result = self.vote(m, e)
            self.assertEqual((result.shape, result.dtype, result.device), (m.shape, dtype, m.device))
            self.assertTrue(torch.equal(m, before_m))
            self.assertTrue(torch.equal(e, before_e))
            self.assertTrue(torch.isfinite(result).all())
            probabilities = result.sigmoid()
            self.assertTrue(((probabilities >= 0) & (probabilities <= 1)).all())
            for index in range(3):
                torch.testing.assert_close(result[index:index+1], self.vote(m[index:index+1], e[index:index+1]))
            torch.testing.assert_close(probabilities, m.new_tensor([[.98], [.7125], [.4]]),
                                       rtol=.02 if dtype in (torch.float16, torch.bfloat16) else 1e-5,
                                       atol=.002 if dtype in (torch.float16, torch.bfloat16) else 1e-6)

    def test_extreme_finite_logits_do_not_need_probability_clipping(self):
        m = torch.tensor([[1000.], [-1000.]], dtype=torch.float64, requires_grad=True)
        e = torch.tensor([[-1000., -900., -800., 500.], [-1000., -900., -800., -700.]],
                         dtype=torch.float64, requires_grad=True)
        result = self.vote(m, e)
        self.assertTrue(torch.isfinite(result).all())
        # Three/four averages leave probabilities asymptotic to these logits.
        torch.testing.assert_close(result, m.new_tensor([[500. + math.log(8/7)],
                                                        [-700. + math.log(15/16)]]))
        loss = torch.nn.functional.binary_cross_entropy_with_logits(result, torch.tensor([[0.], [1.]]))
        loss.backward()
        self.assertTrue(torch.isfinite(m.grad).all())
        self.assertTrue(torch.isfinite(e.grad).all())

    def test_no_rule_gradient_remains_on_mixture(self):
        m, e = self.logits([[.4]], grad=True), self.logits([[.2, .3, .6, .7]], grad=True)
        torch.nn.functional.binary_cross_entropy_with_logits(self.vote(m, e), torch.ones_like(m)).backward()
        torch.testing.assert_close(m.grad, m.new_tensor([[-.6]]))
        torch.testing.assert_close(e.grad, torch.zeros_like(e))

    def test_veto1_gradient_routes_to_selected_expert(self):
        m, e = self.logits([[.3]], grad=True), self.logits([[.98, .6, .7, .8]], grad=True)
        torch.nn.functional.binary_cross_entropy_with_logits(self.vote(m, e), torch.ones_like(m)).backward()
        torch.testing.assert_close(m.grad, torch.zeros_like(m))
        torch.testing.assert_close(e.grad, e.new_tensor([[-.02, 0, 0, 0]]))

    def test_veto2_gradient_connects_mixture_and_mm_maximum(self):
        m, e = self.logits([[.2]], grad=True), self.logits([[.01, .2, .3, .8]], grad=True)
        torch.nn.functional.binary_cross_entropy_with_logits(self.vote(m, e), torch.ones_like(m)).backward()
        torch.testing.assert_close(m.grad, m.new_tensor([[-.16]]))
        torch.testing.assert_close(e.grad, e.new_tensor([[0, 0, 0, -.16]]))

    def test_equal_maxima_use_first_expert_for_gradient(self):
        m, e = self.logits([[.8]], grad=True), self.logits([[.01, .02, .2, .2]], grad=True)
        self.vote(m, e).sigmoid().sum().backward()
        torch.testing.assert_close(m.grad, m.new_tensor([[.04]]))
        torch.testing.assert_close(e.grad, e.new_tensor([[0, 0, .12, 0]]))

    def test_train_and_inference_values_match(self):
        m, e = self.logits([[.2]], grad=True), self.logits([[.01, .2, .3, .8]], grad=True)
        training = self.vote(m, e)
        with torch.no_grad():
            inference = self.vote(m, e)
        self.assertTrue(torch.equal(training, inference))
        self.assertTrue(training.requires_grad)
        self.assertFalse(inference.requires_grad)

    def test_autograd_matches_finite_differences_away_from_boundaries(self):
        m, e = self.logits([[.2]], grad=True), self.logits([[.01, .2, .3, .8]], grad=True)
        self.assertTrue(torch.autograd.gradcheck(self.vote, (m, e), fast_mode=True))

    def test_actual_loss_and_evaluation_fragments_receive_logits(self):
        entry = source_tree('run_gamed.py')
        main = next(n for n in entry.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
        evaluate = next(n for n in entry.body if isinstance(n, ast.FunctionDef) and n.name == 'evaluate')
        def assignment(owner, target):
            return next(n for n in ast.walk(owner) if isinstance(n, ast.Assign)
                        and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)
                        and n.targets[0].id == target
                        and (target != 'val_loss' or isinstance(n.value, ast.Call)))
        sigmoid = next(n for n in ast.walk(evaluate) if isinstance(n, ast.Assign)
                       and isinstance(n.targets[0], ast.Tuple)
                       and isinstance(n.value, ast.Tuple)
                       and isinstance(n.targets[0].elts[0], ast.Name)
                       and n.targets[0].elts[0].id == 'mix_output')
        m = self.logits([[.2], [.2]])
        e = self.logits([[.98, .2, .3, .4], [.01, .2, .3, .8]])
        voted = self.vote(m, e)
        ns = dict(torch=torch, criterion=torch.nn.BCEWithLogitsLoss(), mix_output=voted,
                  labels=torch.tensor([[1.], [0.]], dtype=m.dtype), is_output=e[:, :1],
                  t_output=e[:, 1:2], ip_output=e[:, 2:3], mm_aux_output=e[:, 3:4],
                  setting={'thresh': .5}, thresh=.5)
        execute([assignment(main, 'mix_loss')], ns)
        execute([assignment(evaluate, 'val_loss'), assignment(evaluate, 'threshold'), assignment(evaluate, 'THRESH'),
                 sigmoid, assignment(evaluate, 'validate_argmax')], ns)
        expected_loss = (-math.log(.98) - math.log(.5)) / 2
        self.assertAlmostEqual(ns['mix_loss'].item(), expected_loss, places=12)
        self.assertAlmostEqual(ns['val_loss'].item(), expected_loss, places=12)
        torch.testing.assert_close(ns['mix_output'], m.new_tensor([[.98], [.5]]))
        self.assertEqual(ns['THRESH'], [.5+i/1000 for i in range(-50, 50)])
        torch.testing.assert_close(ns['validate_argmax'], torch.ones((2, 1), dtype=torch.int64))

    def test_invalid_shapes_dtypes_and_thresholds_are_rejected(self):
        m, e = self.logits([[.2]]), self.logits([[.01, .2, .3, .8]])
        for bad_m, bad_e in ((m.flatten(), e), (m, e[:, :3]), (m, e.float()),
                             (m.long(), e.long()), (m, e.to('meta'))):
            with self.assertRaises(ValueError):
                self.vote(bad_m, bad_e)
        for low, high in ((0, .95), (.05, 1), (.6, .95), (.05, .4)):
            with self.assertRaises(ValueError):
                self.vote(m, e, low, high)


if __name__ == '__main__':
    unittest.main()
