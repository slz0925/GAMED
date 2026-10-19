"""GAMEDWithoutVeto research implementation; see docs/THIRD_PARTY.md for provenance."""
from models.checkpoint_compat import prepare_state_dict
from resource_paths import resolve_read_path as _read_path, resolve_english_encoder_path
from positional_encodings.torch_encodings import PositionalEncoding1D
import torch
import torch.nn as nn
from transformers import AutoModel
import models_mae
from timm.models.vision_transformer import Block

class AdaIN(nn.Module):
    def __init__(self):
        super().__init__()

    def mu(self, x):
        """Return the mean across the feature dimension for each sample."""
        return torch.sum(x,(1))/(x.shape[1])

    def sigma(self, x):
        """Return the per-sample feature standard deviation with an epsilon."""
        return torch.sqrt((torch.sum((x.permute([1,0])-self.mu(x)).permute([1,0])**2,(1))+0.000000023)/(x.shape[1]))

    def forward(self, x, mu, sigma):
        """Apply the existing prediction-guided feature adjustment."""
        x_mean = self.mu(x)
        x_std = self.sigma(x)
        x_reduce_mean = x.permute([1, 0]) - x_mean
        x_norm = x_reduce_mean/x_std
        return (sigma.squeeze(1)*(x_norm + mu.squeeze(1))).permute([1,0])

class SimpleGate(nn.Module):
    def __init__(self, dim=1):
        super(SimpleGate, self).__init__()
        self.dim=dim

    def forward(self,x):
        x1,x2 = x.chunk(2,dim=self.dim)
        return x1*x2


class TokenAttention(torch.nn.Module):
    """Aggregate token features using learned, unnormalised attention scores."""

    def __init__(self, input_shape):
        super(TokenAttention, self).__init__()
        self.attention_layer = nn.Sequential(
                            torch.nn.Linear(input_shape, input_shape),
                            nn.SiLU(),
                            torch.nn.Linear(input_shape, 1),
        )

    def forward(self, inputs):
        scores = self.attention_layer(inputs).view(-1, inputs.size(1))
        scores = scores.unsqueeze(1)
        outputs = torch.matmul(scores, inputs).squeeze(1)
        return outputs, scores


class GAMEDWithoutVeto(nn.Module):
    def __init__(self,
                 batch_size=64, dataset='weibo', text_token_len=197, image_token_len=197, is_use_bce=True,
                 thresh=0.5
                 ):
        self.thresh = thresh
        self.batch_size = batch_size
        self.text_token_len, self.image_token_len = text_token_len, image_token_len
        model_size = 'base'
        self.model_size = model_size
        self.dataset = dataset
        self.LOW_BATCH_SIZE_AND_LR = ['Twitter', 'politi']
        print("Using AdaIN-based feature adjustment")

        self.unified_dim, self.text_dim = 768, 768
        self.is_use_bce = is_use_bce
        out_dim = 1 if self.is_use_bce else 2
        self.num_expert = 2
        self.depth = 1
        super(GAMEDWithoutVeto, self).__init__()
        self.image_encoder = models_mae.__dict__["mae_vit_{}_patch16".format(self.model_size)](norm_pix_loss=False)
        checkpoint = torch.load(_read_path('checkpoint/mae_pretrain_vit_{}.pth'.format(self.model_size)), map_location='cpu')
        self.image_encoder.load_state_dict(checkpoint['model'], strict=False)



        english_lists = ['gossip', 'Twitter', 'politi']
        model_name = resolve_english_encoder_path()
        print("Text encoder: ERNIE 2.0 Base (English)")
        self.text_encoder = AutoModel.from_pretrained(model_name)

        self.t_attention = TokenAttention(self.unified_dim)

        from CNN_architectures.pytorch_InceptionResNetV2 import InceptionResNetV2
        self.ip_backbone = InceptionResNetV2(num_classes=self.unified_dim, use_SRM=True).cuda()

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

        self.is_aux_gate = nn.Sequential(nn.Linear(self.unified_dim, self.unified_dim),
                                              nn.SiLU(),
                                              nn.Linear(self.unified_dim, self.num_expert),
                                              )

        self.t_aux_gate = nn.Sequential(nn.Linear(self.unified_dim, self.unified_dim),
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

        self.mm_score = None


    def load_state_dict(self, state_dict, strict=True, assign=False):
        """Load current or compatible checkpoint names without duplicate modules."""
        state_dict = prepare_state_dict(self, state_dict, strict=strict)
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def get_pretrain_features(self, input_ids, attention_mask, token_type_ids, image, no_ambiguity, category=None,
                              calc_ambiguity=False):
        image_feature = self.image_encoder.forward_ying(image)
        text_feature = self.text_encoder(input_ids=input_ids,
                                       attention_mask=attention_mask,
                                       token_type_ids=token_type_ids)[0]

        return image_feature, text_feature
    
    def veto(outputs, mix_output, confidence_threshold=0.9):
        """An unused veto helper; not called by this model's forward pass.

        Return the average with the first opposing branch that meets the
        confidence threshold and exceeds the mixture's maximum score; otherwise
        return the mixture unchanged.
        """
        mix_decision = mix_output.argmax()

        for output in outputs:
            decision = output.argmax()
            confidence = output.max()

            if decision != mix_decision and confidence >= confidence_threshold and confidence > mix_output.max():
                adjusted_mix_output = (mix_output + output) / 2
                return adjusted_mix_output

        return mix_output

    def forward(self, input_ids, attention_mask, token_type_ids, image, no_ambiguity,
                image_aug=None, category=None,
                calc_ambiguity=False, image_feature=None, text_feature=None,
                return_features=False):

        batch_size = image.shape[0]
        if image_aug is None:
            image_aug = image
        p_1d_mm = PositionalEncoding1D(self.unified_dim)
        x_mm = torch.rand(batch_size, self.image_token_len + self.text_token_len, self.unified_dim)
        self.positional_mm = p_1d_mm(x_mm).cuda()
        p_1d_image = PositionalEncoding1D(self.unified_dim)
        x_image = torch.rand(batch_size, self.image_token_len, self.unified_dim)
        self.positional_image = p_1d_image(x_image).cuda()
        p_1d_text = PositionalEncoding1D(self.unified_dim)
        x_text = torch.rand(batch_size, self.text_token_len, self.unified_dim)
        self.positional_text = p_1d_text(x_text).cuda()
        p_1d = PositionalEncoding1D(self.unified_dim)
        x = torch.rand(batch_size, 5, self.unified_dim)
        self.positional_modal_representation = p_1d(x).cuda()


        if image_feature is None:
            image_feature = self.image_encoder.forward_ying(image)

        ip_feature = self.ip_backbone(image)

        if text_feature is None:
            text_feature = self.text_encoder(input_ids=input_ids,
                                           attention_mask=attention_mask,
                                           token_type_ids=token_type_ids)[0]

        text_atn_feature, _ = self.t_attention(text_feature)

        image_atn_feature, _ = self.is_attention(image_feature)

        mm_atn_feature, _ = self.mm_attention(torch.cat((image_feature, text_feature), dim=1))

        is_gate_weights = self.is_gate(image_atn_feature)
        t_gate_weights = self.t_gate(text_atn_feature)

        mm_gate_weights = self.mm_gate(mm_atn_feature)
        mm_aux_gate_weights = self.mm_aux_gate(mm_atn_feature)


        is_feature, is_aux_feature = 0, 0
        for i in range(self.num_expert):
            is_expert = self.is_experts[i]
            tmp_image_feature = image_feature
            for j in range(self.depth):
                tmp_image_feature = is_expert[j](tmp_image_feature + self.positional_image)
            is_feature += (tmp_image_feature * is_gate_weights[:, i].unsqueeze(1).unsqueeze(1))
        is_feature = is_feature[:, 0]

        t_feature, t_aux_feature = 0, 0
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

        irrelevance_score = 1 - torch.sigmoid(mm_aux_output).clone().detach()
        is_mu = self.is_adain_mu(torch.sigmoid(is_output).clone().detach())
        t_mu = self.t_adain_mu(torch.sigmoid(t_output).clone().detach())
        ip_mu = self.ip_adain_mu(torch.sigmoid(ip_output).clone().detach())
        irrelevant_mu = self.irrelevant_adain_mu(irrelevance_score.clone().detach())
        is_sigma = self.is_adain_sigma(torch.sigmoid(is_output).clone().detach())
        t_sigma = self.t_adain_sigma(torch.sigmoid(t_output).clone().detach())
        ip_sigma = self.ip_adain_sigma(torch.sigmoid(ip_output).clone().detach())
        irrelevant_sigma = self.irrelevant_adain_sigma(irrelevance_score.clone().detach())

        is_feature = self.adaIN(is_feature,is_mu,is_sigma)
        t_feature = self.adaIN(t_feature,t_mu,t_sigma)
        shared_mm_feature = shared_mm_feature
        ip_feature = self.adaIN(ip_feature,ip_mu,ip_sigma)
        irrelevant_feature = torch.ones_like(shared_mm_feature)*self.irrelevant_tensor
        irrelevant_token = self.adaIN(irrelevant_feature,irrelevant_mu,irrelevant_sigma)
        fusion_tokens = torch.stack((is_feature,
                                                  t_feature,
                                                  shared_mm_feature,
                                                  ip_feature,
                                                  irrelevant_token
                                                  ), dim=1)


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
        if return_features:
            return mix_output, is_output, t_output, ip_output, mm_aux_output, \
                   torch.mean(self.irrelevant_tensor), \
                   (mix_feature_lite, is_feature_lite, t_feature_lite, ip_feature_lite, mm_aux_feature)

        return mix_output, is_output, t_output, ip_output, mm_aux_output, \
               torch.mean(self.irrelevant_tensor)

    def mapping(self, score):
        diff_with_thresh = torch.abs(score-self.thresh)
        interval = torch.where(score-self.thresh>0, 1-self.thresh, self.thresh)
        return diff_with_thresh/interval



