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

from .cross_entropy_loss import CrossEntropyLoss, cross_entropy
from .dice_loss import DiceLoss, dice_loss
from .focal_loss import FocalLoss, sigmoid_focal_loss
from .giou_loss import GIoULoss, giou_loss
from .smooth_l1_loss import L1Loss, l1_loss, smooth_l1_loss
from .utils import reduce_loss, weight_reduce_loss
