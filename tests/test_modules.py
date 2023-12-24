import pytest
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from src.models.tensorized_module import (
    modify_with_tensorized_module,
    modify_with_tensorized_square_module,
)
import torch


def test_tensorized_square_module():
    class TensorizedSquareModuleConfig:
        def __init__(self):
            self.init_scale = 0.01
            self.lora_modules = ".*SelfAttention|.*EncDecAttention|.*DenseReluDense"
            self.lora_layers = "q|k|v|o"
            self.trainable_param_names = ".*layer_norm.*|.*leaf_tensor.*"
            self.order = 9
            self.embed2ket_rank = 2

    config = TensorizedSquareModuleConfig()
    model_name = "t5-small"  # bigscience/T0_3B
    # model_name = "bigscience/T0_3B"
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    input_seq = tokenizer(
        ["Applies a linear transformation to the incoming data."],
        return_tensors="pt",
    )
    target_seq = tokenizer(
        [
            "Parameters: in_features - size of each input sample. out_features - size of each output sample."
        ],
        return_tensors="pt",
    )

    model = modify_with_tensorized_square_module(model, config)
    with torch.no_grad():
        new_outputs = model(
            input_ids=input_seq.input_ids,
            decoder_input_ids=target_seq.input_ids[:, :-1],
            labels=target_seq.input_ids[:, 1:],
        )


def test_tensorized_module():
    class TensorizedModuleConfig:
        def __init__(self):
            self.init_scale = 0.01
            self.lora_modules = ".*SelfAttention|.*EncDecAttention|.*DenseReluDense"
            self.lora_layers = "q|k|v|o|w.*"
            self.trainable_param_names = ".*layer_norm.*|.*lora_[ab].*"
            self.rank = 2
            self.order_a = 4
            self.order_b = 10
            self.core = "adapter2ket"
            self.embed2ket_rank = 2
            # lora_modules and lora_layers are speicified with regular expressions
            # see https://www.w3schools.com/python/python_regex.asp for reference

    config = TensorizedModuleConfig()
    model_name = "t5-small"  # bigscience/T0_3B
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = modify_with_tensorized_module(model, config)

    input_seq = tokenizer(
        ["Applies a linear transformation to the incoming data."],
        return_tensors="pt",
    )
    target_seq = tokenizer(
        [
            "Parameters: in_features - size of each input sample. out_features - size of each output sample."
        ],
        return_tensors="pt",
    )

    with torch.no_grad():
        new_outputs = model(
            input_ids=input_seq.input_ids,
            decoder_input_ids=target_seq.input_ids[:, :-1],
            labels=target_seq.input_ids[:, 1:],
        )
    assert config.rank == 2
