import torch
import torch.nn as nn
import torch.nn.functional as F
import re

import sys
from os import path

sys.path.append(path.abspath(__file__))

# import src.models.word2ket as w2k
import math


class LoRALinear(nn.Module):
    def __init__(
        self, linear_layer, rank, scaling_rank, init_scale, order, core, embed2ket_rank
    ):
        super().__init__()
        self.in_features = linear_layer.in_features
        self.out_features = linear_layer.out_features
        self.rank = rank
        self.scaling_rank = scaling_rank
        self.weight = linear_layer.weight
        self.bias = linear_layer.bias

        self._scale = None
        self.order = order
        self.embed2ket_rank = embed2ket_rank

        self.use = core
        self.embedding_size_b = linear_layer.out_features
        self.embedding_dim_leaf_b = math.ceil(
            (self.embedding_size_b) ** (1 / self.order)
        )

        print(
            f"leaf_dim_leaf_b: {self.embedding_dim_leaf_b} original embedding:{self.embedding_size_b}"
        )

        self.embedding_size_a = linear_layer.in_features
        self.embedding_dim_leaf_a = math.ceil(
            (self.embedding_size_a) ** (1 / self.order)
        )

        print(
            f"leaf_dim_leaf_a: {self.embedding_dim_leaf_a} original embedding:{self.embedding_size_a}"
        )

        self.layerone_normalization_a = nn.LayerNorm(
            normalized_shape=[self.rank, self.embedding_dim_leaf_a**2]
        )

        self.layerone_normalization_b = nn.LayerNorm(
            normalized_shape=[self.rank, self.embedding_dim_leaf_b**2]
        )

        # if it is lora
        if self.rank > 0:
            self.tensor_rank = self.rank
        else:
            self.tensor_rank = self.scaling_rank
        if self.rank > 0:
            if self.use == "lora":
                self.lora_a = nn.Parameter(
                    torch.randn(rank, linear_layer.in_features) * init_scale
                )
            elif self.use == "adapter2ket":
                self.weight_leafs_a = nn.Parameter(
                    torch.randn(
                        self.order,
                        self.embed2ket_rank,
                        self.rank,
                        self.embedding_dim_leaf_a,
                    )
                )
            if init_scale < 0:
                self.lora_b = nn.Parameter(
                    torch.randn(linear_layer.out_features, rank) * init_scale
                )
            else:
                if self.use == "lora":
                    self.lora_b = nn.Parameter(
                        torch.zeros(linear_layer.out_features, rank)
                    )
                elif self.use == "adapter2ket":
                    self.weight_leafs_b = nn.Parameter(
                        torch.randn(
                            self.order,
                            self.embed2ket_rank,
                            self.rank,
                            self.embedding_dim_leaf_b,
                        )
                    )
                    print("self.weight_leafs_size", self.weight_leafs_b.size())

        if self.scaling_rank:
            self.multi_lora_a = nn.Parameter(
                torch.ones(self.scaling_rank, linear_layer.in_features)
                + torch.randn(self.scaling_rank, linear_layer.in_features) * init_scale
            )

            if init_scale < 0:
                self.multi_lora_b = nn.Parameter(
                    torch.ones(linear_layer.out_features, self.scaling_rank)
                    + torch.randn(linear_layer.out_features, self.scaling_rank)
                    * init_scale
                )
            else:
                if self.use == "ia3":
                    self.multi_lora_b = nn.Parameter(
                        torch.ones(linear_layer.out_features, self.scaling_rank)
                    )
                elif self.use == "emb2ket":
                    self.embedding_size = linear_layer.out_features
                    self.embedding_dim_leaf = math.ceil(
                        (self.embedding_size) ** (1 / self.order)
                    )
                    print(
                        f"leaf_dim :{self.embedding_dim_leaf} original embedding size: {linear_layer.out_features}",
                    )
                    self.weight_leafs = nn.Parameter(
                        torch.ones(
                            self.order,
                            self.embed2ket_rank,
                            1,
                            self.embedding_dim_leaf,
                        )
                    )
                    print("self.weight_leafs_size", self.weight_leafs.size())
        if self.use == "adapter2ket":
            self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain(nonlinearity="leaky_relu", param=math.sqrt(5))
        # std = gain / math.sqrt(self.in_features)
        std = gain / (self.in_features ** (1 / self.order))

        with torch.no_grad():
            self.weight_leafs_a.uniform_(-std, std)
            # torch.nn.init.kaiming_normal_(self.weight_leafs_a)

        with torch.no_grad():
            self.weight_leafs_b.uniform_(-std, std)
            # torch.nn.init.kaiming_normal_(self.weight_leafs_b)

    def tensor_product_represent(self, w, signal="output"):
        if self.order == 2:
            w01 = w[0, :, :, :, None] * w[1, :, :, None, :]
            w01 = w01.view(self.embed2ket_rank, self.tensor_rank, -1)

            if signal == "input":
                w01 = self.layerone_normalization_a(w01)
            elif signal == "output":
                w01 = self.layerone_normalization_b(w01)
            # w01 = nn.LayerNorm(w01.shape[-2:]).cuda()(w01)
            weight = w01.sum(0)
        elif self.order == 4:
            w01 = w[0, :, :, :, None] * w[1, :, :, None, :]
            w01 = w01.view(self.embed2ket_rank, self.tensor_rank, -1)
            # w01 = nn.LayerNorm(w01.shape[-2:]).cuda()(w01)
            w23 = w[2, :, :, :, None] * w[3, :, :, None, :]
            w23 = w23.view(self.embed2ket_rank, self.tensor_rank, -1)
            # w23 = nn.LayerNorm(w23.shape[-2:]).cuda()(w23)
            w0123 = w01[:, :, :, None] * w23[:, :, None, :]
            w0123 = w0123.view(self.embed2ket_rank, self.tensor_rank, -1)
            # w0123 = nn.LayerNorm(w0123.shape[-2:]).cuda()(w0123)
            weight = w0123.sum(0)
        elif self.order == 8:
            w01 = w[0, :, :, :, None] * w[1, :, :, None, :]
            w01 = w01.view(self.embed2ket_rank, self.tensor_rank, -1)
            w23 = w[2, :, :, :, None] * w[3, :, :, None, :]
            w23 = w23.view(self.embed2ket_rank, self.tensor_rank, -1)
            w45 = w[4, :, :, :, None] * w[5, :, :, None, :]
            w45 = w45.view(self.embed2ket_rank, self.tensor_rank, -1)
            w67 = w[6, :, :, :, None] * w[7, :, :, None, :]
            w67 = w67.view(self.embed2ket_rank, self.tensor_rank, -1)
            w0123 = w01[:, :, :, None] * w23[:, :, None, :]
            w0123 = w0123.view(self.embed2ket_rank, self.tensor_rank, -1)
            w4567 = w45[:, :, :, None] * w67[:, :, None, :]
            w4567 = w4567.view(self.embed2ket_rank, self.tensor_rank, -1)
            w01234567 = w0123[:, :, :, None] * w4567[:, :, None, :]
            w01234567 = w01234567.view(self.embed2ket_rank, self.tensor_rank, -1)
            weight = w01234567.sum(0)

        if signal == "output":
            tpr = weight[:, : self.out_features]
        elif signal == "input":
            tpr = weight[:, : self.in_features]
        else:
            raise ValueError("signal must be i)input or ii)output")
        return tpr

    def forward(self, input):
        if self.use == "emb2ket":
            # ia3 like
            self.multi_lora_b = self.tensor_product_represent(
                self.weight_leafs
            ).flatten()
            if self._scale is None:
                self._scale = 1.0 / self.multi_lora_b.mean()
            self.multi_lora_b = self._scale * self.multi_lora_b

        elif self.use == "adapter2ket":
            self.lora_a = self.tensor_product_represent(
                self.weight_leafs_a, signal="input"
            )
            self.lora_b = self.tensor_product_represent(
                self.weight_leafs_b, signal="output"
            )
            self.lora_b = self.lora_b.transpose(1, 0)

        if self.scaling_rank == 1 and self.rank == 0:
            # parsimonious implementation for ia3 and lora scaling
            if self.multi_lora_a.requires_grad:
                # print(self.multi_lora_a.flatten().size())
                hidden = F.linear(
                    (input * self.multi_lora_a.flatten()), self.weight, self.bias
                )
            else:
                hidden = F.linear(input, self.weight, self.bias)
            if self.multi_lora_b.requires_grad:
                # print(hidden.size(), self.multi_lora_b.size())
                hidden = hidden * self.multi_lora_b.flatten()

            return hidden
        else:
            # general implementation for lora (adding and scaling)
            weight = self.weight
            if self.scaling_rank:
                weight = (
                    weight
                    * torch.matmul(self.multi_lora_b, self.multi_lora_a)
                    / self.scaling_rank
                )
            if self.rank:
                weight = weight + torch.matmul(self.lora_b, self.lora_a) / self.rank
            return F.linear(input, weight, self.bias)

    def extra_repr(self):
        return (
            "in_features={}, out_features={}, bias={}, rank={}, scaling_rank={}".format(
                self.in_features,
                self.out_features,
                self.bias is not None,
                self.rank,
                self.scaling_rank,
            )
        )


def modify_with_lora(transformer, config):
    for m_name, module in dict(transformer.named_modules()).items():
        if re.fullmatch(config.lora_modules, m_name):
            for c_name, layer in dict(module.named_children()).items():
                if re.fullmatch(config.lora_layers, c_name):
                    # print(c_name, layer)
                    assert isinstance(
                        layer, nn.Linear
                    ), f"LoRA can only be applied to torch.nn.Linear, but {layer} is {type(layer)}."
                    setattr(
                        module,
                        c_name,
                        LoRALinear(
                            layer,
                            config.lora_rank,
                            config.lora_scaling_rank,
                            config.lora_init_scale,
                            config.order,
                            config.core,
                            config.embed2ket_rank,
                        ),
                    )
    return transformer


if __name__ == "__main__":
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    class LoRAConfig:
        def __init__(self):
            self.lora_rank = 4
            self.lora_init_scale = 0.01
            self.lora_modules = ".*SelfAttention|.*EncDecAttention|.*DenseReluDense"
            self.lora_layers = "q|k|v|o|w.*"
            self.trainable_param_names = ".*layer_norm.*|.*lora_[ab].*"
            self.lora_scaling_rank = 1
            # lora_modules and lora_layers are speicified with regular expressions
            # see https://www.w3schools.com/python/python_regex.asp for reference

    config = LoRAConfig()
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

    # print("Old model")
    # print(model)
    # with torch.no_grad():
    #     old_outputs = model(
    #         input_ids=input_seq.input_ids,
    #         decoder_input_ids=target_seq.input_ids[:, :-1],
    #         labels=target_seq.input_ids[:, 1:],
    #     )

    model = modify_with_lora(model, config)

    # print("New model")
    # print(model)
    # with torch.no_grad():
    #     new_outputs = model(
    #         input_ids=input_seq.input_ids,
    #         decoder_input_ids=target_seq.input_ids[:, :-1],
    #         labels=target_seq.input_ids[:, 1:],
    #     )

    print("Trainable parameters")
    for p_name in dict(model.named_parameters()).keys():
        print(p_name)

    # print(
    #     f"Logits diff {torch.abs(old_outputs.logits - new_outputs.logits).mean():.3f}"
    # )
    # print(f"Loss diff old={old_outputs.loss:.3f} new={new_outputs.loss:.3f}")
