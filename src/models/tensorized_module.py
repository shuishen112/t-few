import torch
import torch.nn as nn
import torch.nn.functional as F
import re
import math


class TensorizedModule(nn.Module):
    def __init__(self, linear_layer, init_scale, order, embed2ket_rank):
        super().__init__()
        self.in_features = linear_layer.in_features
        self.out_features = linear_layer.out_features
        self.weight = linear_layer.weight
        self.bias = linear_layer.bias
        self._scale = None
        self.order = 9
        self.embed2ket_rank = embed2ket_rank
        self.embedding_size_b = linear_layer.out_features
        self.embedding_size_a = linear_layer.in_features

        # self.embedding_dim_leaf_b = math.ceil(
        #     (self.embedding_size_b) ** (1 / self.order)
        # )
        self.embedding_dim_leaf_b = 2
        self.embedding_size_a = linear_layer.in_features
        self.embedding_dim_leaf_a = 2
        # self.embedding_dim_leaf_a = math.ceil(
        #     (self.embedding_size_a) ** (1 / self.order)
        # )
        ###### the adapter is adapter2ket ######
        self.leaf_tensor = nn.Parameter(
            torch.randn(
                self.order,
                self.embed2ket_rank,
                self.embedding_dim_leaf_a,
                self.embedding_dim_leaf_b,
            )
        )
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain(nonlinearity="leaky_relu", param=math.sqrt(5))
        std = gain / (self.in_features ** (1 / self.order))

        with torch.no_grad():
            self.leaf_tensor.uniform_(-std, std)

    def tensor_product_represent(self, w):
        weight_leafs = w
        base = weight_leafs[0]
        leaf_dim = 2
        for i in range(1, self.order):
            leaf_dim = leaf_dim * 2
            base = torch.einsum("abc,ade->abcde", base, weight_leafs[i]).reshape(
                self.embed2ket_rank, leaf_dim, leaf_dim
            )

        base = base.sum(dim=0)
        tpr = base[: self.out_features, : self.in_features]
        return tpr

    def forward(self, input):
        self.tensoried_module = self.tensor_product_represent(self.leaf_tensor)

        weight = self.weight + self.tensoried_module
        return F.linear(input, weight, self.bias)

    def extra_repr(self):
        return "in_features={}, out_features={}, bias={}".format(
            self.in_features,
            self.out_features,
            self.bias is not None,
        )


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
                            config.init_scale,
                            config.order,
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
            self.order = 11
            self.core = "adapter2ket"
            self.embed2ket_rank = 2
            # lora_modules and lora_layers are speicified with regular expressions
            # see https://www.w3schools.com/python/python_regex.asp for reference

    config = TensorizedModuleConfig()
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

    model = modify_with_tensorized_module(model, config)
    breakpoint()

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
