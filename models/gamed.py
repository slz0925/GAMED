"""GAMED research implementation; see docs/THIRD_PARTY.md for provenance."""
from models.checkpoint_compat import prepare_state_dict
from resource_paths import resolve_read_path as _read_path, resolve_english_encoder_path
from positional_encodings.torch_encodings import PositionalEncoding1D
import torch
import torch.nn as nn
from transformers import AutoModel
import models_mae
from timm.models.vision_transformer import Block

class AdaIN(nn.Module):
    """Prediction-guided affine adjustment with a signed scale."""

    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    def mu(self, x):
        return x.mean(dim=1)

    def sigma(self, x):
        return (x.var(dim=1, unbiased=False) + self.eps).sqrt()

    def forward(self, x, mu, sigma):
        # Accumulate feature statistics in float32 for low-precision inputs.
        work = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
        variance, mean = torch.var_mean(work, dim=1, unbiased=False, keepdim=True)
        normalised = (work - mean) * torch.rsqrt(variance + self.eps)
        return (sigma.to(work.dtype) * normalised + mu.to(work.dtype)).to(x.dtype)

class TokenAttention(torch.nn.Module):
    """Aggregate token features using normalised token importance."""

    def __init__(self, input_shape):
        super(TokenAttention, self).__init__()
        self.attention_layer = nn.Sequential(
                            torch.nn.Linear(input_shape, input_shape),
                            nn.SiLU(),
                            torch.nn.Linear(input_shape, 1),
        )

    def forward(self, inputs):
        scores = self.attention_layer(inputs).view(-1, inputs.size(1))
        work = inputs.float() if inputs.dtype in (torch.float16, torch.bfloat16) else inputs
        with torch.autocast(device_type=inputs.device.type, enabled=False):
            scores = torch.softmax(scores.to(work.dtype), dim=-1).unsqueeze(1)
            outputs = torch.matmul(scores, work).squeeze(1)
        return outputs.to(inputs.dtype), scores.to(inputs.dtype)


class GAMED(nn.Module):
    def __init__(self):
        model_size = 'base'
        self.model_size = model_size
        print("Using AdaIN-based feature adjustment")

        self.unified_dim = 768
        out_dim = 1
        self.num_expert = 2
        self.depth = 1
        super(GAMED, self).__init__()
        self.image_encoder = models_mae.__dict__["mae_vit_{}_patch16".format(self.model_size)](norm_pix_loss=False)
        checkpoint = torch.load(_read_path('checkpoint/mae_pretrain_vit_{}.pth'.format(self.model_size)), map_location='cpu')
        self.image_encoder.load_state_dict(checkpoint['model'], strict=False)


        model_name = resolve_english_encoder_path()
        print("Text encoder: ERNIE 2.0 Base (English)")
        self.text_encoder = AutoModel.from_pretrained(model_name)

        self.t_attention = TokenAttention(self.unified_dim)

        from CNN_architectures.pytorch_InceptionResNetV2 import InceptionResNetV2
        self.ip_backbone = InceptionResNetV2(num_classes=self.unified_dim, use_SRM=True)


        self.is_attention = TokenAttention(self.unified_dim)

        self.mm_attention = TokenAttention(self.unified_dim)
        is_expert_list, t_expert_list, mm_expert_list = [], [], []

        for i in range(self.num_expert):
            is_expert = []
            for j in range(self.depth):
                is_expert.append(Block(dim=self.unified_dim, num_heads=8))

            is_expert = nn.ModuleList(is_expert)
            is_expert_list.append(is_expert)

        for i in range(self.num_expert):
            t_expert = []
            mm_expert = []
            for j in range(self.depth):
                t_expert.append(Block(dim=self.unified_dim, num_heads=8))
                mm_expert.append(Block(dim=self.unified_dim, num_heads=8))

            t_expert = nn.ModuleList(t_expert)
            t_expert_list.append(t_expert)
            mm_expert = nn.ModuleList(mm_expert)
            mm_expert_list.append(mm_expert)

        self.is_experts = nn.ModuleList(is_expert_list)
        self.t_experts = nn.ModuleList(t_expert_list)
        self.mm_experts = nn.ModuleList(mm_expert_list)
        self.is_gate = nn.Sequential(nn.Linear(self.unified_dim, self.unified_dim),
                                            nn.SiLU(),
                                            nn.Linear(self.unified_dim, self.num_expert),
                                            )

        self.t_gate = nn.Sequential(nn.Linear(self.unified_dim, self.unified_dim),
                                       nn.SiLU(),
                                       nn.Linear(self.unified_dim, self.num_expert),
                                       )
        self.mm_gate = nn.Sequential(nn.Linear(self.unified_dim, self.unified_dim),
                                     nn.SiLU(),
                                     nn.Linear(self.unified_dim, self.num_expert),
                                     )
        self.mm_aux_gate = nn.Sequential(nn.Linear(self.unified_dim, self.unified_dim),
                                     nn.SiLU(),
                                     nn.Linear(self.unified_dim, self.num_expert),
                                     )

        self.fusion_attention = TokenAttention(self.unified_dim)

        self.fusion_gate = nn.Sequential(nn.Linear(self.unified_dim, self.unified_dim),
                                                         nn.SiLU(),
                                                         nn.Linear(self.unified_dim, self.num_expert),
                                                         )

        self.irrelevant_tensor = nn.Parameter(torch.ones((1,self.unified_dim)),requires_grad=True)


        self.mix_trim = nn.Sequential(
            nn.Linear(self.unified_dim, 64),
            nn.SiLU(),
        )
        self.mix_classifier = nn.Sequential(
            nn.Linear(64, out_dim),
        )

        self.t_trim = nn.Sequential(
            nn.Linear(self.unified_dim, 64),
            nn.SiLU(),
        )
        self.t_classifier = nn.Sequential(
            nn.Linear(64, out_dim),
        )

        self.is_trim = nn.Sequential(
            nn.Linear(self.unified_dim, 64),
            nn.SiLU(),
        )
        self.is_classifier = nn.Sequential(
            nn.Linear(64, out_dim),
        )

        self.ip_trim = nn.Sequential(
            nn.Linear(self.unified_dim, 64),
            nn.SiLU(),
        )
        self.ip_classifier = nn.Sequential(
            nn.Linear(64, out_dim),
        )

        self.mm_aux_trim = nn.Sequential(
            nn.Linear(self.unified_dim, 64),
            nn.SiLU(),
        )
        self.mm_aux_classifier = nn.Sequential(
            nn.Linear(64, out_dim),
        )

        self.is_adain_mu = nn.Sequential(
            nn.Linear(1, self.unified_dim),
            nn.SiLU(),
            nn.Linear(self.unified_dim, 1),
        )
        self.is_adain_sigma = nn.Sequential(
            nn.Linear(1, self.unified_dim),
            nn.SiLU(),
            nn.Linear(self.unified_dim,1),
        )
        self.t_adain_mu = nn.Sequential(
            nn.Linear(1, self.unified_dim),
            nn.SiLU(),
            nn.Linear(self.unified_dim, 1),
        )
        self.t_adain_sigma = nn.Sequential(
            nn.Linear(1, self.unified_dim),
            nn.SiLU(),
            nn.Linear(self.unified_dim, 1),
        )
        self.ip_adain_mu = nn.Sequential(
            nn.Linear(1, self.unified_dim),
            nn.SiLU(),
            nn.Linear(self.unified_dim, 1),
        )
        self.ip_adain_sigma = nn.Sequential(
            nn.Linear(1, self.unified_dim),
            nn.SiLU(),
            nn.Linear(self.unified_dim, 1),
        )
        self.irrelevant_adain_mu = nn.Sequential(
            nn.Linear(1, self.unified_dim),
            nn.SiLU(),
            nn.Linear(self.unified_dim, 1),
        )
        self.irrelevant_adain_sigma = nn.Sequential(
            nn.Linear(1, self.unified_dim),
            nn.SiLU(),
            nn.Linear(self.unified_dim, 1),
        )
        self.adaIN = AdaIN()

        final_fusing_expert = []
        for i in range(self.num_expert):
            fusing_expert = []
            for j in range(self.depth):
                fusing_expert.append(Block(dim=self.unified_dim, num_heads=8))
            fusing_expert = nn.ModuleList(fusing_expert)
            final_fusing_expert.append(fusing_expert)

        self.fusion_experts = nn.ModuleList(final_fusing_expert)


    def load_state_dict(self, state_dict, strict=True, assign=False):
        """Load current or compatible checkpoint names without duplicate modules."""
        state_dict = prepare_state_dict(self, state_dict, strict=strict)
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def get_pretrain_features(self, input_ids, attention_mask, token_type_ids, image):
        image_feature = self.image_encoder.forward_ying(image)
        text_feature = self.text_encoder(input_ids=input_ids,
                                       attention_mask=attention_mask,
                                       token_type_ids=token_type_ids)[0]

        return image_feature, text_feature

    def forward(self, input_ids, attention_mask, token_type_ids, image, *,
                calc_ambiguity=False, image_feature=None, text_feature=None,
                return_features=False):


        if image_feature is None:
            image_feature = self.image_encoder.forward_ying(image)

        ip_feature = self.ip_backbone(image)

        if text_feature is None:
            text_feature = self.text_encoder(input_ids=input_ids,
                                           attention_mask=attention_mask,
                                           token_type_ids=token_type_ids)[0]

        # All samples share the same positions; broadcast one encoding per sequence.
        self.positional_image = PositionalEncoding1D(self.unified_dim).to(image_feature.device)(image_feature[:1])
        self.positional_text = PositionalEncoding1D(self.unified_dim).to(text_feature.device)(text_feature[:1])
        mm_tokens = torch.cat((image_feature, text_feature), dim=1)
        self.positional_mm = PositionalEncoding1D(self.unified_dim).to(mm_tokens.device)(mm_tokens[:1])

        text_atn_feature, _ = self.t_attention(text_feature)

        image_atn_feature, _ = self.is_attention(image_feature)

        mm_atn_feature, _ = self.mm_attention(torch.cat((image_feature, text_feature), dim=1))

        is_gate_weights = self.is_gate(image_atn_feature)
        t_gate_weights = self.t_gate(text_atn_feature)

        mm_gate_weights = self.mm_gate(mm_atn_feature)
        mm_aux_gate_weights = self.mm_aux_gate(mm_atn_feature)


        is_feature = 0
        for i in range(self.num_expert):
            is_expert = self.is_experts[i]
            tmp_image_feature = image_feature
            for j in range(self.depth):
                tmp_image_feature = is_expert[j](tmp_image_feature + self.positional_image)
            is_feature += (tmp_image_feature * is_gate_weights[:, i].unsqueeze(1).unsqueeze(1))
        is_feature = is_feature[:, 0]

        t_feature = 0
        for i in range(self.num_expert):
            t_expert = self.t_experts[i]
            tmp_text_feature = text_feature
            for j in range(self.depth):
                tmp_text_feature = t_expert[j](tmp_text_feature + self.positional_text)
            t_feature += (tmp_text_feature * t_gate_weights[:, i].unsqueeze(1).unsqueeze(1))
        t_feature = t_feature[:, 0]

        mm_feature = torch.cat((image_feature, text_feature), dim=1)
        shared_mm_feature, mm_consistency_feature = 0, 0
        for i in range(self.num_expert):
            mm_expert = self.mm_experts[i]
            tmp_mm_feature = mm_feature
            for j in range(self.depth):
                tmp_mm_feature = mm_expert[j](tmp_mm_feature + self.positional_mm)
            shared_mm_feature += (tmp_mm_feature * mm_gate_weights[:, i].unsqueeze(1).unsqueeze(1))
            mm_consistency_feature += (tmp_mm_feature * mm_aux_gate_weights[:, i].unsqueeze(1).unsqueeze(1))
        shared_mm_feature = shared_mm_feature[:, 0]
        mm_consistency_feature = mm_consistency_feature[:, 0]


        mm_aux_feature = self.mm_aux_trim(mm_consistency_feature)
        mm_aux_output = self.mm_aux_classifier(mm_aux_feature)

        if calc_ambiguity:
            return mm_aux_output, mm_aux_output, mm_aux_output

        ip_feature_lite = self.ip_trim(ip_feature)
        is_feature_lite = self.is_trim(is_feature)
        t_feature_lite = self.t_trim(t_feature)

        ip_output = self.ip_classifier(ip_feature_lite)
        is_output = self.is_classifier(is_feature_lite)
        t_output = self.t_classifier(t_feature_lite)

        irrelevance_score = 1 - torch.sigmoid(mm_aux_output).clone()
        is_mu = self.is_adain_mu(torch.sigmoid(is_output).clone().detach())
        t_mu = self.t_adain_mu(torch.sigmoid(t_output).clone().detach())
        ip_mu = self.ip_adain_mu(torch.sigmoid(ip_output).clone().detach())
        irrelevant_mu = self.irrelevant_adain_mu(irrelevance_score.clone())
        is_sigma = self.is_adain_sigma(torch.sigmoid(is_output).clone().detach())
        t_sigma = self.t_adain_sigma(torch.sigmoid(t_output).clone().detach())
        ip_sigma = self.ip_adain_sigma(torch.sigmoid(ip_output).clone().detach())
        irrelevant_sigma = self.irrelevant_adain_sigma(irrelevance_score.clone())

        is_feature = self.adaIN(is_feature,is_mu,is_sigma)
        t_feature = self.adaIN(t_feature,t_mu,t_sigma)
        ip_feature = self.adaIN(ip_feature,ip_mu,ip_sigma)
        irrelevant_feature = torch.ones_like(shared_mm_feature)*self.irrelevant_tensor
        irrelevant_token = self.adaIN(irrelevant_feature,irrelevant_mu,irrelevant_sigma)
        fusion_tokens = torch.stack((is_feature,
                                                  t_feature,
                                                  shared_mm_feature,
                                                  ip_feature,
                                                  irrelevant_token
                                                  ), dim=1)


        self.positional_modal_representation = PositionalEncoding1D(self.unified_dim).to(fusion_tokens.device)(fusion_tokens[:1])
        fusion_attention_feature, _ = self.fusion_attention(fusion_tokens)
        fusion_gate_weights = self.fusion_gate(fusion_attention_feature)

        mix_feature = 0
        for i in range(self.num_expert):
            fusing_expert = self.fusion_experts[i]
            tmp_fusion_feature = fusion_tokens
            for j in range(self.depth):
                tmp_fusion_feature = fusing_expert[j](tmp_fusion_feature + self.positional_modal_representation)
            tmp_fusion_feature = tmp_fusion_feature[:, 0]
            mix_feature += (tmp_fusion_feature * fusion_gate_weights[:, i].unsqueeze(1))

        mix_feature_lite = self.mix_trim(mix_feature)
        mix_output = self.mix_classifier(mix_feature_lite)

        mix_output = self.veto_voting(
            mix_output,
            torch.cat((ip_output, is_output, t_output, mm_aux_output), dim=1),
        )
        if return_features:
            return mix_output, is_output, t_output, ip_output, mm_aux_output, \
                   torch.mean(self.irrelevant_tensor), \
                   (mix_feature_lite, is_feature_lite, t_feature_lite, ip_feature_lite, mm_aux_feature)
        

        return mix_output, is_output, t_output, ip_output, mm_aux_output, \
               torch.mean(self.irrelevant_tensor)

    @staticmethod
    def veto_voting(mix_logits, expert_logits, low_threshold=0.55, high_threshold=0.95):
        """Apply confidence-aware Veto Voting and return binary logits.

        IP, IS, T and MM vote with class confidence max(p, 1-p). Veto 1 selects
        the most confident eligible expert; opposing tied maxima retain the
        mixture. Veto 2 averages once with a reliable minority expert when all
        members of the opposing strict majority have low confidence.
        Thresholds refer to class confidence. See docs/MODEL_MAP.md.
        """
        if (mix_logits.ndim != 2 or mix_logits.shape[1] != 1
                or expert_logits.shape != (mix_logits.shape[0], 4)):
            raise ValueError('Veto expects mixture [batch, 1] and experts [batch, 4]')
        if (not mix_logits.is_floating_point() or not expert_logits.is_floating_point()
                or mix_logits.dtype != expert_logits.dtype
                or mix_logits.device != expert_logits.device):
            raise ValueError('Veto logits must share a floating dtype and device')
        if not 0.5 < low_threshold < high_threshold < 1:
            raise ValueError('Veto thresholds must satisfy 0.5 < low < high < 1')

        # C = sigmoid(abs(logit)); compare magnitudes without sigmoid saturation.
        # Half-precision inputs use float32 for thresholds and stable averaging.
        mix = mix_logits.float() if mix_logits.dtype in (torch.float16, torch.bfloat16) else mix_logits
        experts = expert_logits.float() if expert_logits.dtype in (torch.float16, torch.bfloat16) else expert_logits
        low_logit = torch.logit(experts.new_tensor(low_threshold))
        high_logit = torch.logit(experts.new_tensor(high_threshold))
        confidence = experts.abs()
        mix_confidence = mix.abs()
        positive = experts >= 0
        maximum = confidence.amax(dim=1, keepdim=True)
        strongest = confidence == maximum
        strongest_positive = (strongest & positive).any(dim=1, keepdim=True)
        strongest_negative = (strongest & ~positive).any(dim=1, keepdim=True)
        conflict = strongest_positive & strongest_negative
        veto1 = (maximum > high_logit) & (maximum > mix_confidence) & ~conflict
        # Same-class tied maxima share gradients, independently of column order.
        replacement = torch.where(strongest_positive,
                                  expert_logits.amax(dim=1, keepdim=True),
                                  expert_logits.amin(dim=1, keepdim=True))

        positive_count = positive.sum(dim=1, keepdim=True)
        majority_positive = positive_count > 2
        strict_majority = (positive_count > 2) | (positive_count < 2)
        in_majority = positive == majority_positive
        all_majority_low = ((confidence < low_logit) | ~in_majority).all(dim=1, keepdim=True)
        minority = ~in_majority & strict_majority
        minority_confidence = confidence.masked_fill(~minority, -1).amax(dim=1, keepdim=True)
        selected = minority & (confidence == minority_confidence)
        # With four voters and a strict majority, there is at most one minority member.
        minority_logit = torch.where(selected, experts, torch.zeros_like(experts)).sum(dim=1, keepdim=True)
        veto2 = (~veto1 & ~conflict & strict_majority & all_majority_low
                 & ((mix >= 0) != majority_positive)
                 & (minority_confidence > low_logit) & (mix_confidence < high_logit))
        # logit((sigmoid(mix) + sigmoid(minority_logit)) / 2), without clipping.
        log_positive = torch.logaddexp(torch.nn.functional.logsigmoid(mix),
                                      torch.nn.functional.logsigmoid(minority_logit))
        log_negative = torch.logaddexp(torch.nn.functional.logsigmoid(-mix),
                                      torch.nn.functional.logsigmoid(-minority_logit))
        average = (log_positive - log_negative).to(dtype=mix_logits.dtype)
        return torch.where(veto1, replacement, torch.where(veto2, average, mix_logits))
