import torch
import torch.nn as nn
import torch.nn.functional as F
import re
import math


class TensorizedSquareModule(nn.Module):
    def __init__(self, linear_layer, init_scale, order, embed2ket_rank):
        super().__init__()
        self.in_features = linear_layer.in_features
        self.out_features = linear_layer.out_features
        self.weight = linear_layer.weight
        self.bias = linear_layer.bias
        self._scale = None

        self.embed2ket_rank = embed2ket_rank
        # it is note that the a is the output size and b is the input size
        self.embedding_size_b = linear_layer.in_features
        self.embedding_size_a = linear_layer.out_features

        self.order = order

        self.embedding_dim_leaf = math.ceil((self.embedding_size_b) ** (1 / self.order))

        ###### the adapter is adapter2ket ######
        self.leaf_tensor = nn.Parameter(
            torch.randn(
                self.order,
                self.embed2ket_rank,
                self.embedding_dim_leaf,
                self.embedding_dim_leaf,
            )
        )
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain(nonlinearity="leaky_relu", param=math.sqrt(5))
        std = gain / (self.in_features ** (1 / self.order))

        with torch.no_grad():
            self.leaf_tensor.uniform_(-std, std)

    def tensor_product_represent(self, w, order, out_feature_dim, in_feature_dim):
        weight_leafs = w
        base = weight_leafs[0]

        leaf_dim = leaf_dim_begin = weight_leafs.shape[-1]
        for i in range(1, order):
            leaf_dim = leaf_dim * leaf_dim_begin
            base = torch.einsum("abc,ade->abdce", base, weight_leafs[i]).reshape(
                self.embed2ket_rank, leaf_dim, leaf_dim
            )
        base = base.sum(dim=0)
        print("tensor size", base.shape)
        tpr = base[:out_feature_dim, :in_feature_dim]
        return tpr

    def forward(self, input):
        self.tensoried_module = self.tensor_product_represent(
            self.leaf_tensor, self.order, self.out_features, self.in_features
        )
        weight = self.weight + self.tensoried_module
        return F.linear(input, weight, self.bias)

    def extra_repr(self):
        return "in_features={}, out_features={}, embed2ket_rank={}, order={} bias={}".format(
            self.in_features,
            self.out_features,
            self.embed2ket_rank,
            self.order,
            self.bias is not None,
        )


class TensorizedModule(nn.Module):
    def __init__(
        self, linear_layer, rank, init_scale, order_a, order_b, embed2ket_rank
    ):
        super().__init__()
        self.in_features = linear_layer.in_features
        self.out_features = linear_layer.out_features
        self.weight = linear_layer.weight
        self.bias = linear_layer.bias
        self.rank = rank
        self._scale = None

        self.embed2ket_rank = embed2ket_rank
        # it is note that the a is the output size and b is the input size
        self.embedding_size_b = linear_layer.in_features
        self.embedding_size_a = linear_layer.out_features

        self.order_a = order_a
        self.order_b = order_b

        self.embedding_dim_leaf_b = math.ceil(
            (self.embedding_size_b) ** (1 / self.order_b)
        )

        self.embedding_dim_leaf_a = math.ceil(
            (self.embedding_size_a) ** (1 / self.order_a)
        )
        ###### the adapter is adapter2ket ######
        self.leaf_tensor_a = nn.Parameter(
            torch.randn(
                self.order_a,
                self.embed2ket_rank,
                self.rank,
                self.embedding_dim_leaf_a,
            )
        )

        self.leaf_tensor_b = nn.Parameter(
            torch.randn(
                self.order_b,
                self.embed2ket_rank,
                self.rank,
                self.embedding_dim_leaf_b,
            )
        )
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain(nonlinearity="leaky_relu", param=math.sqrt(5))
        std = gain / (self.in_features ** (1 / self.order_a))

        with torch.no_grad():
            self.leaf_tensor_a.uniform_(-std, std)
            self.leaf_tensor_b.uniform_(-std, std)

    def tensor_product_represent(self, w, order, feature_dim):
        weight_leafs = w
        base = weight_leafs[0]

        for i in range(1, order):
            base = torch.einsum("abc,abd->abcd", base, weight_leafs[i]).reshape(
                self.embed2ket_rank, self.rank, -1
            )

        base = base.sum(dim=0)
        tpr = base[:, :feature_dim]
        return tpr

    def forward(self, input):
        self.tensoried_module_A = self.tensor_product_represent(
            self.leaf_tensor_a, self.order_a, self.out_features
        )
        self.tensoried_module_B = self.tensor_product_represent(
            self.leaf_tensor_b, self.order_b, self.in_features
        )
        weight = (
            self.weight
            + self.tensoried_module_A.transpose(0, 1) @ self.tensoried_module_B
        )
        return F.linear(input, weight, self.bias)

    def extra_repr(self):
        return "in_features={}, out_features={}, rank={},  bias={}".format(
            self.in_features,
            self.out_features,
            self.rank,
            self.bias is not None,
        )


def modify_with_tensorized_square_module(transformer, config):
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
                        TensorizedSquareModule(
                            layer,
                            config.init_scale,
                            config.order,
                            config.embed2ket_rank,
                        ),
                    )
    return transformer


def modify_with_tensorized_module(transformer, config):
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
                        TensorizedModule(
                            layer,
                            config.lora_rank,
                            config.init_scale,
                            config.order_a,
                            config.order_b,
                            config.embed2ket_rank,
                        ),
                    )
    return transformer


if __name__ == "__main__":
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    class TensorizedModuleConfig:
        def __init__(self):
            self.init_scale = 0.01
            self.lora_modules = ".*SelfAttention|.*EncDecAttention|.*DenseReluDense"
            self.lora_layers = "q|k|v|o|w.*"
            self.trainable_param_names = ".*layer_norm.*|.*lora_[ab].*"
            self.rank = 2
            self.order_a = 8
            self.order_b = 8
            self.core = "adapter2ket"
            self.embed2ket_rank = 2
            # lora_modules and lora_layers are speicified with regular expressions
            # see https://www.w3schools.com/python/python_regex.asp for reference

    class TensorizedSquareModuleConfig:
        def __init__(self):
            self.init_scale = 0.01
            self.lora_modules = ".*SelfAttention|.*EncDecAttention|.*DenseReluDense"
            self.lora_layers = "q|k|v|o"
            self.trainable_param_names = ".*layer_norm.*|.*leaf_tensor.*"
            self.order = 9
            self.embed2ket_rank = 2

    config = TensorizedModuleConfig()
    # model_name = "t5-small"  # bigscience/T0_3B
    model_name = "bigscience/T0_3B"
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    breakpoint()
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

    print("Old model")
    print(model)
    with torch.no_grad():
        old_outputs = model(
            input_ids=input_seq.input_ids,
            decoder_input_ids=target_seq.input_ids[:, :-1],
            labels=target_seq.input_ids[:, 1:],
        )

    model = modify_with_tensorized_module(model, config)

    print("New model")
    print(model)

    with torch.no_grad():
        new_outputs = model(
            input_ids=input_seq.input_ids,
            decoder_input_ids=target_seq.input_ids[:, :-1],
            labels=target_seq.input_ids[:, 1:],
        )

    # print("Trainable parameters")
    # for p_name in dict(model.named_parameters()).keys():
    #     print(p_name)

    # print(
    #     f"Logits diff {torch.abs(old_outputs.logits - new_outputs.logits).mean():.3f}"
    # )
    # print(f"Loss diff old={old_outputs.loss:.3f} new={new_outputs.loss:.3f}")
