"""Tensor tests for Veto Voting, without encoder imports or model resources."""
import ast
import copy
import math
import itertools
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

    def test_figure6_case1_retains_real_mixture(self):
        m, _, out = self.assert_probability(.306, [.230, .278, .737, .408], .306)
        self.assertTrue(torch.equal(out, m))

    def test_figure6_case2_selects_text_expert(self):
        self.assert_probability(.487, [.251, .177, .978, .424], .978)

    def test_figure6_case3_averages_once_with_is(self):
        self.assert_probability(.791, [.496, .596, .462, .488], .6935)

    def test_no_rule_preserves_exact_logit(self):
        m, _, out = self.assert_probability(.4, [.2, .3, .6, .7], .4)
        self.assertTrue(torch.equal(out, m))

    def test_veto1_replaces_opposing_mixture_for_both_classes(self):
        self.assert_probability(.2, [.98, .2, .3, .4], .98)
        self.assert_probability(.8, [.02, .8, .7, .6], .02)

    def test_veto1_also_applies_to_same_class(self):
        self.assert_probability(.7, [.98, .6, .7, .8], .98)
        self.assert_probability(.3, [.02, .4, .3, .2], .02)

    def test_veto1_requires_exceeding_mixture_confidence(self):
        self.assert_probability(.99, [.98, .6, .7, .8], .99)
        self.assert_probability(.98, [.98, .6, .7, .8], .98)
        self.assert_probability(.01, [.98, .6, .7, .8], .01)

    def test_veto1_requires_high_threshold(self):
        self.assert_probability(.3, [.94, .6, .7, .8], .3)
        self.assert_probability(.7, [.06, .4, .3, .2], .7)

    def test_veto1_selects_highest_class_confidence_not_fake_probability(self):
        self.assert_probability(.6, [.97, .99, .98, .7], .99)
        self.assert_probability(.6, [.97, .005, .98, .7], .005)

    def test_veto1_opposing_exact_tie_retains_mixture_and_gradient(self):
        m = self.logits([[.6]], grad=True)
        e = torch.tensor([[5., -5., 1., -1.]], dtype=m.dtype, requires_grad=True)
        out = self.vote(m, e)
        self.assertTrue(torch.equal(out, m))
        out.sum().backward()
        torch.testing.assert_close(m.grad, torch.ones_like(m))
        torch.testing.assert_close(e.grad, torch.zeros_like(e))

    def test_veto1_same_class_ties_share_gradient(self):
        for sign in (1, -1):
            m = self.logits([[.6]], grad=True)
            e = torch.tensor([[5.*sign, 5.*sign, 1., -1.]], dtype=m.dtype, requires_grad=True)
            self.assertTrue(torch.equal(self.vote(m, e), e[:, :1]))
            self.vote(m, e).sum().backward()
            torch.testing.assert_close(m.grad, torch.zeros_like(m))
            torch.testing.assert_close(e.grad, e.new_tensor([[.5, .5, 0., 0.]]))

    def test_veto1_ties_below_stronger_expert_do_not_block_replacement(self):
        m = torch.zeros((1, 1), dtype=torch.float64)
        e = m.new_tensor([[4., -4., 5., 1.]])
        self.assertTrue(torch.equal(self.vote(m, e), e[:, 2:3]))

    def test_veto1_precedes_veto2(self):
        self.assert_probability(.8, [.49, .48, .47, .99], .99)

    def test_veto2_is_single_average_not_three_updates(self):
        self.assert_probability(.8, [.49, .48, .47, .6], .7)

    def test_veto2_selects_real_minority_symmetrically(self):
        self.assert_probability(.2, [.51, .52, .53, .4], .3)

    def test_high_confidence_real_majority_is_not_low_confidence(self):
        # Veto 1 legitimately selects .01; Veto 2 must not average with .8.
        self.assert_probability(.8, [.01, .02, .03, .8], .01)

    def test_one_reliable_majority_member_blocks_veto2(self):
        self.assert_probability(.8, [.49, .48, .2, .6], .8)
        self.assert_probability(.2, [.51, .52, .8, .4], .2)

    def test_one_low_confidence_expert_does_not_invalidate_majority(self):
        self.assert_probability(.8, [.49, .2, .3, .6], .8)

    def test_mix_agrees_with_low_majority(self):
        self.assert_probability(.4, [.49, .48, .47, .6], .4)
        self.assert_probability(.6, [.51, .52, .53, .4], .6)

    def test_minority_must_be_reliable(self):
        self.assert_probability(.8, [.49, .48, .47, .54], .8)
        self.assert_probability(.2, [.51, .52, .53, .46], .2)

    def test_unanimous_majority_has_no_minority(self):
        self.assert_probability(.8, [.49, .48, .47, .46], .8)
        self.assert_probability(.2, [.51, .52, .53, .54], .2)

    def test_two_two_tie_does_not_trigger_even_if_mixture_breaks_tie(self):
        self.assert_probability(.8, [.49, .48, .6, .7], .8)

    def test_probability_half_counts_as_fake(self):
        # Three Fake votes (including two exact halves) oppose a Real mixture.
        self.assert_probability(.2, [.5, .5, .51, .4], .3)

    def test_high_confidence_mix_is_protected_from_veto2(self):
        self.assert_probability(.99, [.49, .48, .47, .6], .99)
        self.assert_probability(.01, [.51, .52, .53, .4], .01)

    def test_high_threshold_strictness_and_mix_protection(self):
        for dtype in (torch.float32, torch.float64):
            h = torch.logit(torch.tensor(.95, dtype=dtype))
            for sign in (1, -1):
                m = torch.zeros((1, 1), dtype=dtype)
                e = m.new_tensor([[0., .4, -.4, .2]])
                e[:, 0] = sign*h
                self.assertTrue(torch.equal(self.vote(m, e), m))
                e[:, 0] = sign*torch.nextafter(h, h.new_tensor(math.inf))
                self.assertTrue(torch.equal(self.vote(m, e), e[:, :1]))
                e[:, 0] = sign*torch.nextafter(h, h.new_tensor(-math.inf))
                self.assertTrue(torch.equal(self.vote(m, e), m))
                e = self.logits([[.49, .48, .47, .6]], dtype)*sign
                m = (sign*h).reshape(1, 1)
                self.assertTrue(torch.equal(self.vote(m, e), m))
                below = (sign*torch.nextafter(h, h.new_tensor(-math.inf))).reshape(1, 1)
                self.assertFalse(torch.equal(self.vote(below, e), below))

    def test_low_threshold_strict_for_every_majority_member(self):
        for dtype in (torch.float32, torch.float64):
            low = torch.logit(torch.tensor(.55, dtype=dtype))
            m = self.logits([[.8]], dtype)
            e = self.logits([[.49, .48, .47, .6]], dtype)
            for column in range(3):
                changed = e.clone(); changed[:, column] = -low
                self.assertTrue(torch.equal(self.vote(m, changed), m))
                changed[:, column] = -torch.nextafter(low, low.new_tensor(-math.inf))
                torch.testing.assert_close(self.vote(m, changed).sigmoid(), m.new_tensor([[.7]]))
                changed[:, column] = -torch.nextafter(low, low.new_tensor(math.inf))
                self.assertTrue(torch.equal(self.vote(m, changed), m))

    def test_low_threshold_strict_for_minority(self):
        for dtype in (torch.float32, torch.float64):
            low = torch.logit(torch.tensor(.55, dtype=dtype))
            m = self.logits([[.8]], dtype)
            e = self.logits([[.49, .48, .47, .6]], dtype)
            e[:, 3] = low
            self.assertTrue(torch.equal(self.vote(m, e), m))
            e[:, 3] = torch.nextafter(low, low.new_tensor(math.inf))
            torch.testing.assert_close(self.vote(m, e).sigmoid(), m.new_tensor([[.675]]))
            e[:, 3] = torch.nextafter(low, low.new_tensor(-math.inf))
            self.assertTrue(torch.equal(self.vote(m, e), m))

    def test_custom_thresholds_are_effective(self):
        m, e = self.logits([[.6]]), self.logits([[.93, .7, .6, .4]])
        self.assertTrue(torch.equal(self.vote(m, e), m))
        self.assertTrue(torch.equal(self.vote(m, e, .6, .9), e[:, :1]))
        m, e = self.logits([[.8]]), self.logits([[.44, .43, .42, .7]])
        self.assertTrue(torch.equal(self.vote(m, e), m))
        torch.testing.assert_close(self.vote(m, e, .6, .95).sigmoid(), m.new_tensor([[.75]]))

    def test_permutations_preserve_values_and_corresponding_gradients(self):
        cases = [(.6, [5., -5., 1., -1.]), (.6, [5., 5., 1., -1.]),
                 (.6, [5., -6., 1., -1.]),
                 (.8, self.logits([.49, .48, .47, .6]).tolist())]
        for probability, row in cases:
            m, e = self.logits([[probability]], grad=True), torch.tensor([row], dtype=torch.float64, requires_grad=True)
            reference = self.vote(m, e)
            gm, ge = torch.autograd.grad(reference.sum(), (m, e))
            for permutation in itertools.permutations(range(4)):
                pm, pe = m.detach().clone().requires_grad_(), e.detach()[:, permutation].clone().requires_grad_()
                actual = self.vote(pm, pe)
                am, ae = torch.autograd.grad(actual.sum(), (pm, pe))
                self.assertTrue(torch.equal(actual, reference))
                torch.testing.assert_close(am, gm)
                torch.testing.assert_close(ae, ge[:, permutation])

    def test_sign_reversal_symmetry_away_from_zero(self):
        for mixture, row in ((.3, [.98, .6, .7, .8]), (.8, [.49, .48, .47, .6]),
                             (.4, [.2, .3, .6, .7]), (.8, [.49, .48, .47, .54])):
            m, e = self.logits([[mixture]]), self.logits([row])
            torch.testing.assert_close(self.vote(-m, -e), -self.vote(m, e), rtol=0, atol=0)

    def test_batch_shape_dtype_device_and_input_immutability(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            m = self.logits([[.2], [.8], [.4]], dtype)
            e = self.logits([[.98, .2, .3, .4], [.49, .48, .47, .6], [.2, .3, .6, .7]], dtype)
            before_m, before_e = m.clone(), e.clone()
            result = self.vote(m, e)
            self.assertEqual((result.shape, result.dtype, result.device), (m.shape, dtype, m.device))
            self.assertTrue(torch.equal(m, before_m)); self.assertTrue(torch.equal(e, before_e))
            self.assertTrue(torch.isfinite(result).all())
            probabilities = result.sigmoid()
            self.assertTrue(((probabilities >= 0) & (probabilities <= 1)).all())
            for index in range(3):
                torch.testing.assert_close(result[index:index+1], self.vote(m[index:index+1], e[index:index+1]))
            torch.testing.assert_close(probabilities, m.new_tensor([[.98], [.7], [.4]]),
                                       rtol=.02 if dtype in (torch.float16, torch.bfloat16) else 1e-5,
                                       atol=.002 if dtype in (torch.float16, torch.bfloat16) else 1e-6)

    def test_extreme_finite_logits_and_gradients(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            large = torch.finfo(dtype).max
            m = torch.tensor([[large], [-large], [0.], [0.]], dtype=dtype, requires_grad=True)
            e = torch.tensor([[-large, -large/2, -1., 1.], [large, large/2, -1., 1.],
                              [large, -1., 1., 2.], [-large, -1., 1., 2.]], dtype=dtype, requires_grad=True)
            out = self.vote(m, e)
            torch.testing.assert_close(out, m.new_tensor([[large], [-large], [large], [-large]]))
            self.assertTrue(torch.isfinite(out).all())
            out.sum().backward()
            self.assertTrue(torch.isfinite(m.grad).all()); self.assertTrue(torch.isfinite(e.grad).all())

    def test_half_precision_backward_and_cpu_autocast(self):
        for dtype in (torch.float16, torch.bfloat16):
            m, e = self.logits([[.8]], dtype, True), self.logits([[.49, .48, .47, .6]], dtype, True)
            with torch.autocast('cpu', dtype=torch.bfloat16):
                out = self.vote(m, e)
            out.float().sigmoid().sum().backward()
            self.assertTrue(torch.isfinite(m.grad).all()); self.assertTrue(torch.isfinite(e.grad).all())
            self.assertGreater(m.grad.abs().sum(), 0)
            self.assertGreater(e.grad[:, 3].abs().sum(), 0)
            self.assertEqual(out.dtype, dtype)

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

    def test_veto2_gradient_connects_only_mix_and_selected_minority(self):
        m, e = self.logits([[.8]], grad=True), self.logits([[.49, .48, .47, .6]], grad=True)
        out = self.vote(m, e)
        # d BCE(y=1)/d p_new = -1/.7, with each probability weighted by 1/2.
        torch.nn.functional.binary_cross_entropy_with_logits(out, torch.ones_like(m)).backward()
        torch.testing.assert_close(m.grad, m.new_tensor([[-.5*.8*.2/.7]]))
        torch.testing.assert_close(e.grad, e.new_tensor([[0, 0, 0, -.5*.6*.4/.7]]))

    def test_train_and_inference_values_match(self):
        m, e = self.logits([[.8]], grad=True), self.logits([[.49, .48, .47, .6]], grad=True)
        training = self.vote(m, e)
        with torch.no_grad():
            inference = self.vote(m, e)
        self.assertTrue(torch.equal(training, inference))
        self.assertTrue(training.requires_grad); self.assertFalse(inference.requires_grad)

    def test_autograd_matches_finite_differences_away_from_boundaries(self):
        for probability, row in ((.8, [.49, .48, .47, .6]), (.2, [.51, .52, .53, .4]),
                                 (.3, [.98, .6, .7, .8]), (.4, [.2, .3, .6, .7])):
            m, e = self.logits([[probability]], grad=True), self.logits([row], grad=True)
            self.assertTrue(torch.autograd.gradcheck(self.vote, (m, e), fast_mode=True))

    def test_empty_batch(self):
        m = torch.empty((0, 1)); e = torch.empty((0, 4))
        self.assertEqual(self.vote(m, e).shape, (0, 1))

    def test_voting_does_not_consume_torch_rng(self):
        m, e = self.logits([[.8]]), self.logits([[.49, .48, .47, .6]])
        state = torch.random.get_rng_state().clone()
        self.vote(m, e)
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))

    def test_invalid_shapes_dtypes_devices_and_thresholds_are_rejected(self):
        m, e = self.logits([[.2]]), self.logits([[.49, .48, .47, .6]])
        for bad_m, bad_e in ((m.flatten(), e), (m, e[:, :3]), (m, e.float()),
                             (m.long(), e.long()), (m, e.to('meta'))):
            with self.assertRaises(ValueError):
                self.vote(bad_m, bad_e)
        for low, high in ((0, .95), (.55, 1), (.5, .95), (.95, .55), (.6, .6),
                          (float('nan'), .95), (.55, float('inf'))):
            with self.assertRaises(ValueError):
                self.vote(m, e, low, high)

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
        m = self.logits([[.2], [.8]])
        e = self.logits([[.98, .2, .3, .4], [.49, .48, .47, .6]])
        voted = self.vote(m, e)
        ns = dict(torch=torch, criterion=torch.nn.BCEWithLogitsLoss(), mix_output=voted,
                  labels=torch.tensor([[1.], [0.]], dtype=m.dtype), is_output=e[:, :1],
                  t_output=e[:, 1:2], ip_output=e[:, 2:3], mm_aux_output=e[:, 3:4],
                  setting={'thresh': .5}, thresh=.5)
        execute([assignment(main, 'mix_loss')], ns)
        execute([assignment(evaluate, 'val_loss'), assignment(evaluate, 'threshold'), assignment(evaluate, 'THRESH'),
                 sigmoid, assignment(evaluate, 'validate_argmax')], ns)
        expected_loss = (-math.log(.98) - math.log(.3)) / 2
        self.assertAlmostEqual(ns['mix_loss'].item(), expected_loss, places=12)
        self.assertAlmostEqual(ns['val_loss'].item(), expected_loss, places=12)
        torch.testing.assert_close(ns['mix_output'], m.new_tensor([[.98], [.7]]))
        self.assertEqual(ns['THRESH'], [.5+i/1000 for i in range(-50, 50)])
        torch.testing.assert_close(ns['validate_argmax'], torch.ones((2, 1), dtype=torch.int64))


if __name__ == '__main__':
    unittest.main()
