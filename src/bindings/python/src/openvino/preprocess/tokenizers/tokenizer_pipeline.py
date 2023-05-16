# -*- coding: utf-8 -*-
# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from functools import singledispatchmethod
from typing import List, Optional, Any, Dict
from unittest.mock import MagicMock

from openvino.runtime.exceptions import UserInputError, OVTypeError
from openvino.runtime import Type, Shape, PartialShape, op, Model
from openvino.runtime.utils.types import as_node

import weakref


string_ops = MagicMock()


def string_to_bytes(value) -> bytes:
    return [byte for byte in bytes(value, 'utf-8')]


def pack_strings(values: List[str]):
    return string_to_bytes(values[0])


@dataclass
class BasePipelineStep:
    _pipeline: Optional[weakref.ReferenceType["TokenizerPipeline"]] = field(default=None, init=False, repr=False)

    def __str__(self) -> str:
        params_string = ", ".join(f"{key}={val!r}" for key, val in self.get_config().items())
        return f"{self.__class__.__name__}({params_string})"

    def get_config(self) -> Dict[str, Any]:
        config = {key: value for key, value in vars(self).items() if not key.startswith("_")}
        properties = {
            key: getattr(self, key)
            for key in dir(type(self))
            if not key.startswith("_") and isinstance(getattr(type(self), key), property)
        }
        config.update(properties)
        return config

    def get_pipeline(self) -> Optional["TokenizerPipeline"]:
        return self._pipeline()

    def set_pipeline(self, pipeline: "TokenizerPipeline") -> None:
        self._pipeline = weakref.ref(pipeline)

    def get_ov_subgraph(self, *input_nodes):
        raise NotImplementedError

    @staticmethod
    def create_string_constant_node(value: str) -> op.Constant:
        data = string_to_bytes(value)
        return op.Constant(
            Type.u8,
            Shape([len(data)]),
            data
        )

    @staticmethod
    def create_list_of_strings_constant_node(values: List[str]) -> op.Constant:
        data = pack_strings(values)
        return op.Constant(
            Type.u8,
            Shape([len(data)]),
            data,
        )

@dataclass
class NormalizationStep(BasePipelineStep):
    pass


@dataclass
class NormalizeUnicode(NormalizationStep):
    normalization_form: str = "NFD"

    def get_ov_subgraph(self, *input_nodes) -> "NormalizeUnicode":
        return string_ops.NormalizeUnicodeWithOffsets(*input_nodes, self.normalization_form)


@dataclass
class NMTNormalizationStep(NormalizationStep):
    """Normaization based on NMT task.

    https://github.com/huggingface/tokenizers/blob/28cd3dce2a75d106572392194ff2564574c33235/tokenizers/src/normalizers/unicode.rs#L44
    """


@dataclass
class CaseFoldStep(NormalizationStep):
    def get_ov_subgraph(self, *input_nodes) -> "CaseFoldStep":
        return string_ops.CaseFold(*input_nodes)


@dataclass
class RegExpNormalizationStep(NormalizationStep):
    regex_search_pattern: str
    replace_term: str

    @classmethod
    def strip_accents_regex(cls) -> "RegExpNormalizationStep":
        return cls(regex_search_pattern=r"\p{Mn}", replace_term="")

    @classmethod
    def del_control_chars_regex(cls) -> "RegExpNormalizationStep":
        return cls(regex_search_pattern=r"\p{Cc}|\p{Cf}", replace_term=" ")

    def get_ov_subgraph(self, *input_nodes) -> "RegExpNormalizationStep":

        return string_ops.RegExpNormalization(
            *input_nodes,
            self.create_string_constant_node(self.regex_search_pattern),
            self.create_string_constant_node(self.replace_term),
        )


@dataclass
class StripAccentsStep(NormalizationStep):
    def get_ov_subgraph(self, *input_nodes) -> "RegExpNormalizationStep":
        return RegExpNormalizationStep.strip_accents_regex().get_ov_subgraph(*input_nodes)


@dataclass
class DelControlCharsStep(NormalizationStep):
    def get_ov_subgraph(self, *input_nodes) -> "RegExpNormalizationStep":
        return RegExpNormalizationStep.del_control_chars_regex().get_ov_subgraph(*input_nodes)


@dataclass
class StripStringStep(NormalizationStep):
    left: bool
    right: bool


@dataclass
class PreTokenizatinStep(BasePipelineStep):
    pass


@dataclass
class RegExpSplitStep(PreTokenizatinStep):
    split_pattern: str
    invert: bool = False
    behaviour: str = "Remove"

    @classmethod
    def bert_splitter(cls) -> "RegExpSplitStep":
        """Generates a step with a standard BERT regex.

        The source:
        https://github.com/tensorflow/text/blob/4a098cd852c0b7ebee621e2d211c7f202dd679c2/tensorflow_text/python/ops/bert_tokenizer.py#L39
        """
        return cls(
            "|".join(
                [
                    r"\s+",
                    r"|".join(
                        [
                            r"[!-/]",
                            r"[:-@]",
                            r"[\[-`]",
                            r"[{-~]",
                            r"[\p{P}]",
                        ],
                    ),
                    r"|".join(
                        [
                            r"[\x{4E00}-\x{9FFF}]",
                            r"[\x{3400}-\x{4DBF}]",
                            r"[\x{20000}-\x{2A6DF}]",
                            r"[\x{2A700}-\x{2B73F}]",
                            r"[\x{2B740}-\x{2B81F}]",
                            r"[\x{2B820}-\x{2CEAF}]",
                            r"[\x{F900}-\x{FAFF}]",
                            r"[\x{2F800}-\x{2FA1F}]",
                        ],
                    ),
                ],
            ),
        )

    @classmethod
    def whitespace_splitter(cls) -> "RegExpSplitStep":
        return cls(r"\w+|[^\w\s]+")

    def get_ov_subgraph(self, *input_nodes) -> "RegExpNormalizationStep":
        return string_ops.RegExpSplit(
            *input_nodes,
            self.create_string_constant_node(self.split_pattern),
            self.behaviour,
            self.invert,
        )


@dataclass
class WhitespaceSplitStep(PreTokenizatinStep):
    """Works like python `str.split`."""
    def get_ov_subgraph(self, *input_nodes) -> "RegExpNormalizationStep":
        return RegExpSplitStep.whitespace_splitter().get_ov_subgraph(*input_nodes)


@dataclass
class PunctuationSplitStep(PreTokenizatinStep):
    """Splits string on punctuation chars."""

    behaviour: str = "Isolated"


@dataclass
class TokenizationModelStep(BasePipelineStep):
    pass


@dataclass
class WordPieceTokenizationStep(TokenizationModelStep):
    vocab: List[str] = field(repr=False)
    unk_token: str = "[UNK]"
    suffix_indicator: str = "##"
    max_bytes_per_word: int = 100
    unk_token_id: int = field(init=False)

    def __post_init__(self) -> None:
        try:
            self.unk_token_id = self.vocab.index(self.unk_token)
        except ValueError:
            raise UserInputError(f"Cannot find unknown token '{self.unk_token}' in the vocab")

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @classmethod
    def from_hf_json(cls, tokenizer_json: Dict[str, Any]) -> "WordPieceTokenizationStep":
        return cls(
            unk_token=tokenizer_json["model"]["unk_token"],
            suffix_indicator=tokenizer_json["model"]["continuing_subword_prefix"],
            vocab=[token for token, index in sorted(tokenizer_json["model"]["vocab"].items(), key=lambda x: x[1])],
        )

    def get_ov_subgraph(self, *input_nodes):
        return string_ops.WordPieceTokenization(
            *input_nodes,
            self.create_list_of_strings_constant_node(self.vocab),
            as_node(self.unk_token_id),
            self.suffix_indicator,
            self.max_bytes_per_word,
        )


@dataclass
class PostTokenizationStep(BasePipelineStep):
    pass


@dataclass
class TruncationStep(PostTokenizationStep):
    max_length: int
    truncate_right: bool = True
    axis: int = -1

    @classmethod
    def from_hf_json(cls, tokenizer_json: Dict[str, Any], num_of_added_tokens: int = 0) -> "TruncationStep":
        return cls(
            max_length=tokenizer_json["truncation"]["max_length"] - num_of_added_tokens,
            truncate_right=tokenizer_json["truncation"]["direction"] == "Right",
        )

    @classmethod
    def from_hf_object(cls, tokenizer: Any, num_of_added_tokens: int = 0) -> "TruncationStep":
        return cls(
            max_length=tokenizer.model_max_length - num_of_added_tokens,
            truncate_right=tokenizer.truncation_side == "right",
        )

    def get_ov_subgraph(self, *input_nodes):
        operation = string_ops.Truncation(
            *input_nodes,
            as_node(self.max_length),
            self.truncate_right,
            self.axis,
        )
        operation.configure_mock(**{"outputs.return_value": [MagicMock() for _ in range(len(input_nodes))]})
        return operation


@dataclass
class SpecialTokenWithId:
    token: str
    _token_id: Optional[int] = None

    def set_token_id(self, vocab: Optional[List[str]]) -> None:
        if vocab is not None:
            self._token_id = vocab.index(self.token)

    @property
    def token_id(self) -> Optional[int]:
        return self._token_id


@dataclass
class TokenWithTypeId:
    token_type_id: Optional[int] = None


@dataclass
class AddToken(TokenWithTypeId, SpecialTokenWithId):
    pass


@dataclass
class Sequence(TokenWithTypeId):
    pass


@dataclass
class CombineSegmentsStep(PostTokenizationStep):
    inputs: List[TokenWithTypeId] = field(default_factory=list)
    segment_ids: Optional[List[int]] = None
    axis: int = -1

    def __post_init__(self):
        if self.segment_ids is not None:
            return

        segment_ids_tensor = [node.token_type_id for node in self.inputs]
        if any(segment is None for segment in segment_ids_tensor):
            segment_ids_tensor = [0] * len(self.inputs)

        self.segment_ids = segment_ids_tensor

    def set_tokens_ids(self, vocab: Optional[List[int]]) -> None:
        for input_ in self.inputs:
            if isinstance(input_, AddToken):
                input_.set_token_id(vocab)

    @property
    def number_of_added_tokens(self) -> int:
        return sum(1 for input_ in self.inputs if isinstance(input_, AddToken))

    @classmethod
    def from_hf_json_template_postprocessor(
            cls, tokenizer_json: Dict[str, Any], number_of_inputs: int = 1
    ) -> "CombineSegmentsStep":
        inputs: List[TokenWithTypeId] = []
        if number_of_inputs == 1:
            post_processor = tokenizer_json["post_processor"]["single"]
        else:
            post_processor = tokenizer_json["post_processor"]["pair"]

        for template_dict in post_processor:
            if "SpecialToken" in template_dict:
                step = AddToken(
                    token=template_dict["SpecialToken"]["id"],
                    token_type_id=template_dict["SpecialToken"]["type_id"],
                )
                inputs.append(step)
            else:
                inputs.append(Sequence(token_type_id=template_dict["Sequence"]["type_id"]))

        return cls(inputs)

    @classmethod
    def from_hf_json_bert_postprocessor(cls, tokenizer_json: Dict[str, Any], number_of_inputs: int = 1):
        post_processor_dict = tokenizer_json["post_processor"]
        inputs: List[TokenWithTypeId] = [
            AddToken(
                token=post_processor_dict["cls"][0],
                token_type_id=0,
            ),
            Sequence(token_type_id=0),
            AddToken(
                token=post_processor_dict["sep"][0],
                token_type_id=0,
            ),
        ]

        if number_of_inputs == 2:
            inputs.extend(
                [
                    Sequence(token_type_id=1),
                    AddToken(
                        token=post_processor_dict["sep"][0],
                        token_type_id=1,
                    ),
                ]
            )
        return cls(inputs)

    def get_ov_subgraph(self, *input_nodes):
        number_of_sequence_inputs = sum(
            1 for input_ in self.inputs if isinstance(input_, Sequence)
        )
        if number_of_sequence_inputs != len(input_nodes):
            raise UserInputError(
                f"Number of input nodes: {len(input_nodes)}, must be equal to {number_of_sequence_inputs}"
            )

        input_nodes_iter = iter(input_nodes)
        op_inputs = [
            next(input_nodes_iter) if isinstance(node, Sequence) else as_node(node.token_type_id)
            for node in self.inputs
        ]

        operation = string_ops.CombineSegments(
            *op_inputs,
            self.segment_ids,
            self.axis,
        )
        operation.configure_mock(**{"outputs.return_value": [MagicMock()]})
        return operation


@dataclass
class PaddingStep(PostTokenizationStep, SpecialTokenWithId):
    pad_right: bool = True
    token_type_id: Optional[int] = None
    max_length: int = -1
    axis: int = -1

    @classmethod
    def from_hf_json(cls, tokenizer_json: Dict[str, Any]) -> "PaddingStep":
        padding_dict = tokenizer_json["padding"]
        return cls(
            token=padding_dict["pad_token"],
            pad_right=padding_dict["direction"] == "Right",
            token_type_id=padding_dict["pad_type_id"],
        )

    def get_ov_subgraph(self, *input_nodes):
        operation = string_ops.PaddingStep(
            *input_nodes,
            as_node(self.token_id),
            as_node(self.max_length),
            self.pad_right,
            self.axis,
        )
        operation.configure_mock(**{"outputs.return_value": [MagicMock()]})
        return operation


@dataclass
class TokenizerPipeline:
    steps: List[BasePipelineStep] = field(default_factory=list)
    vocab: Optional[List[str]] = field(default=None, repr=False)
    number_of_inputs: int = 1

    def get_config(self) -> Dict[str, Dict[str, Any]]:
        return {type(step).__name__: step.get_config() for step in self.steps}

    @singledispatchmethod
    def add_steps(self, steps: Any) -> None:
        raise OVTypeError(f"Type {type(steps)} is not supported")

    @add_steps.register
    def _(self, steps: BasePipelineStep) -> None:
        self.steps.append(steps)
        steps.set_pipeline(self)

    @add_steps.register
    def _(self, steps: list) -> None:
        for step in steps:
            self.steps.append(step)
            step.set_pipeline(self)

    def __getitem__(self, item: int) -> BasePipelineStep:
        return self.steps[item]

    @property
    def processing_steps(self) -> List[BasePipelineStep]:
        return [step for step in self.steps if not isinstance(step, PostTokenizationStep)]

    @property
    def post_tokenization_steps(self) -> List[PostTokenizationStep]:
        return [step for step in self.steps if isinstance(step, PostTokenizationStep)]

    @staticmethod
    def create_string_input():
        return op.Parameter(Type.u8, PartialShape(["?"]))

    def create_processing_pipeline(self, input_nodes: List[op.Parameter]) -> List[string_ops]:
        processing_pipelines_outputs = []

        for input_node in input_nodes:
            for step in self.processing_steps:
                input_node = step.get_ov_subgraph(input_node)
            processing_pipelines_outputs.append(input_node)

        return processing_pipelines_outputs

    def create_post_tokenization_pipeline(self, input_nodes: List[string_ops]) -> List[string_ops]:
        outputs = []
        for step in self.post_tokenization_steps:
            pipeline_step = step.get_ov_subgraph(*input_nodes)
            input_nodes = pipeline_step.outputs()

            if isinstance(step, CombineSegmentsStep):
                input_nodes.append(MagicMock(name="token_type_ids"))
                outputs.append(input_nodes.pop(-1))  # token_type_ids node
            if isinstance(step, PaddingStep):
                input_nodes.append(MagicMock(name="attention_mask"))
                outputs.append(input_nodes.pop(-1))  # attention_mask node

        outputs.insert(0, input_nodes[0])
        return outputs

    def get_ov_subgraph(self) -> Model:
        input_nodes = [self.create_string_input() for _ in range(self.number_of_inputs)]
        processing_outputs = self.create_processing_pipeline(input_nodes)
        outputs = self.create_post_tokenization_pipeline(processing_outputs)
        print(string_ops.method_calls)
        # return Model(outputs, input_nodes, name="tokenizer")
        return outputs, input_nodes
