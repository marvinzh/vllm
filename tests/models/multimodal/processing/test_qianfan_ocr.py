# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for QianfanOCR-specific config fields.

The processor logic is shared with InternVL and tested in test_internvl.py.
"""

import pytest

from ...utils import build_model_context

MODEL_PATH = "baidu/QianfanOCR"


@pytest.mark.parametrize("model_id", [MODEL_PATH])
def test_config_loaded_correctly(model_id: str) -> None:
    """Smoke-test: QianfanOCRConfig fields are accessible as expected."""
    ctx = build_model_context(model_id, limit_mm_per_prompt={"image": 1})
    config = ctx.model_config.hf_config

    assert config.model_type == "qianfan_ocr"
    assert config.vision_config.model_type == "qianfan_ocr_vision"
    assert config.vision_config.image_size == 448
    assert config.vision_config.patch_size == 14
    assert config.downsample_ratio == 0.5
    assert config.text_config.model_type == "qwen3"
