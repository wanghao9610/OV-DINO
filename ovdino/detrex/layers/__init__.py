# coding=utf-8
# Copyright 2022 The IDEA Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from .attention import (
    ConditionalCrossAttention,
    ConditionalSelfAttention,
    MultiheadAttention,
)
from .box_ops import (
    box_cxcywh_to_xyxy,
    box_iou,
    box_xyxy_to_cxcywh,
    generalized_box_iou,
    masks_to_boxes,
)
from .class_embed import ClassEmbed, SimpleClassEmbed
from .conv import ConvNorm, ConvNormAct
from .dcn_v3 import DCNv3, DCNv3Function, dcnv3_core_pytorch
from .denoising import GenerateDNQueries, apply_box_noise, apply_label_noise
from .layer_norm import LayerNorm
from .mlp import FFN, MLP
from .multi_scale_deform_attn import (
    MultiScaleDeformableAttention,
    multi_scale_deformable_attn_pytorch,
)
from .position_embedding import (
    PositionEmbeddingLearned,
    PositionEmbeddingSine,
    get_sine_pos_embed,
)
from .shape_spec import ShapeSpec
from .transformer import BaseTransformerLayer, TransformerLayerSequence
