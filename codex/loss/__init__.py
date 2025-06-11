# Copyright 2024 CoDeX authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Loss functions."""

from codex.loss.pretrained_features import load_vgg16_model
from codex.loss.pretrained_features import compute_vgg16_features
from codex.loss.wasserstein import lowpass
from codex.loss.wasserstein import multi_wasserstein_distortion
from codex.loss.wasserstein import wasserstein_distortion
