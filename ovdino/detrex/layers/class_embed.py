import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassEmbed(nn.Module):
    def __init__(
        self,
        lang_embed_dim: int = 768,
        embed_dim: int = 256,
    ):
        super().__init__()
        self.image_embed_proj = nn.Identity()
        self.lang_embed_proj = nn.Linear(lang_embed_dim, embed_dim, bias=True)
        bias_value = -math.log((1 - 0.01) / 0.01)
        self.lang_bias = nn.ParameterList(
            [nn.Parameter(torch.zeros(lang_embed_dim), requires_grad=True)]
        )
        self.lang_bias0 = nn.ParameterList(
            [nn.Parameter(torch.Tensor([bias_value]), requires_grad=True)]
        )
        self.lang_log_scale = nn.ParameterList(
            [nn.Parameter(torch.Tensor([0.0]), requires_grad=True)]
        )

    def forward(self, image_embeds, lang_embeds):
        num_queries = image_embeds.shape[1]

        image_embeds = self.image_embed_proj(image_embeds)
        lang_embeds = F.normalize(lang_embeds, p=2, dim=-1)
        lang_embeds_proj = self.lang_embed_proj(lang_embeds / 2.0)
        lang_embeds_bias = (
            torch.einsum("bcd,d->bc", lang_embeds, self.lang_bias[0])
            + self.lang_bias0[0]
        )
        lang_embeds_bias = lang_embeds_bias.unsqueeze(1).repeat(1, num_queries, 1)
        dot_product_logit = (
            torch.einsum("bnd,bcd->bnc", image_embeds, lang_embeds_proj)
            / self.lang_log_scale[0].exp()
        ) + lang_embeds_bias
        dot_product_logit = torch.clamp(dot_product_logit, min=-500, max=500)

        return dot_product_logit


class SimpleClassEmbed(nn.Module):
    def __init__(
        self,
        lang_embed_dim: int = 768,
        embed_dim: int = 256,
    ):
        super().__init__()
        self.lang_embed_proj = nn.Linear(lang_embed_dim, embed_dim, bias=True)

    def forward(self, image_embeds, lang_embeds):
        lang_embeds = F.normalize(lang_embeds, p=2, dim=-1)
        lang_embeds_proj = self.lang_embed_proj(lang_embeds)
        dot_product_logit = torch.einsum("bnd,bcd->bnc", image_embeds, lang_embeds_proj)
        dot_product_logit = torch.clamp(dot_product_logit, min=-500, max=500)

        return dot_product_logit
