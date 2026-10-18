"""Checkpoint key compatibility for the two GAMED variants."""
from collections import OrderedDict
import warnings


MODULE_NAMES = {
    'image_model': 'image_encoder', 'text_model': 'text_encoder',
    'vgg_net': 'ip_backbone',
    'image_attention': 'is_attention', 'text_attention': 't_attention',
    'image_experts': 'is_experts', 'text_experts': 't_experts',
    'image_gate_mae': 'is_gate', 'text_gate': 't_gate',
    'mm_gate_1': 'mm_aux_gate',
    'image_gate_mae_1': 'is_aux_gate', 'text_gate_1': 't_aux_gate',
    'fusion_SE_network_main_task': 'fusion_gate',
    'final_attention': 'fusion_attention', 'final_fusing_experts': 'fusion_experts',
    'image_trim': 'is_trim', 'text_trim': 't_trim', 'vgg_trim': 'ip_trim',
    'aux_trim': 'mm_aux_trim',
    'image_alone_classifier': 'is_classifier', 'text_alone_classifier': 't_classifier',
    'vgg_alone_classifier': 'ip_classifier', 'aux_classifier': 'mm_aux_classifier',
    'mapping_IS_MLP_mu': 'is_adain_mu', 'mapping_IS_MLP_sigma': 'is_adain_sigma',
    'mapping_T_MLP_mu': 't_adain_mu', 'mapping_T_MLP_sigma': 't_adain_sigma',
    'mapping_IP_MLP_mu': 'ip_adain_mu', 'mapping_IP_MLP_sigma': 'ip_adain_sigma',
    'mapping_CC_MLP_mu': 'irrelevant_adain_mu',
    'mapping_CC_MLP_sigma': 'irrelevant_adain_sigma',
}


def canonical_key(key):
    """Map a root module name, preserving every descendant component."""
    if not isinstance(key, str):
        raise TypeError('State-dict keys must be strings')
    if key.startswith('module.'):
        key = key[7:]
    root, separator, suffix = key.partition('.')
    return MODULE_NAMES.get(root, root) + separator + suffix


def remap_state_dict(state_dict):
    """Return an ordered key mapping without copying or changing tensors."""
    converted = OrderedDict()
    for key, value in state_dict.items():
        target = canonical_key(key)
        if target in converted:
            raise ValueError('Checkpoint key collision at ' + target)
        converted[target] = value
    if hasattr(state_dict, '_metadata'):
        metadata = OrderedDict()
        wrapped = 'module' in state_dict._metadata
        for key, value in state_dict._metadata.items():
            if wrapped and key == '':
                continue  # DataParallel wrapper metadata, not model metadata.
            target = '' if wrapped and key == 'module' else canonical_key(key)
            if target in metadata:
                raise ValueError('Checkpoint metadata collision at ' + target)
            metadata[target] = value
        converted._metadata = metadata
    return converted


def prepare_state_dict(model, state_dict, strict=True):
    """Validate keys and tensor shapes before any parameter is overwritten.

    Explicit partial loading reports all missing and unexpected keys. Shape
    mismatches and ambiguous key mappings always fail, including strict=False.
    """
    converted = remap_state_dict(state_dict)
    expected = model.state_dict()
    missing = sorted(set(expected) - set(converted))
    unexpected = sorted(set(converted) - set(expected))
    incompatible = []
    for key in expected.keys() & converted.keys():
        value = converted[key]
        if not hasattr(value, 'shape') or tuple(value.shape) != tuple(expected[key].shape):
            incompatible.append(key)
    if incompatible:
        raise RuntimeError('Checkpoint tensor shape mismatch: ' + ', '.join(sorted(incompatible)))
    if missing or unexpected:
        message = f'Checkpoint keys: missing={missing}; unexpected={unexpected}'
        if strict:
            raise RuntimeError(message)
        warnings.warn(message, RuntimeWarning, stacklevel=2)
    return converted
