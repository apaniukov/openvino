# -*- coding: utf-8 -*-
# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import sys
from typing import Any, Optional

from openvino.runtime.exceptions import OVTypeError
from .tokenizer_pipeline import TokenizerPipeline


def convert_tokenizer(tokenizer_object: Any, number_of_inputs: int = 1) -> TokenizerPipeline:
    if "transformers" in sys.modules:
        from transformers import PreTrainedTokenizerBase
        from .hf_parser import TransformersTokenizerPipelineParser

        if isinstance(tokenizer_object, PreTrainedTokenizerBase):
            return TransformersTokenizerPipelineParser(tokenizer_object).parse(number_of_inputs=number_of_inputs)


    raise OVTypeError(f"Tokenizer type is not supported: {type(tokenizer_object)}")
