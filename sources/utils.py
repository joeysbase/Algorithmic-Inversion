import torch.nn as nn
import json
from collections import OrderedDict


def get_prompt_APPS(doc):
    """Generate prompts for APPS
    Finetuning setup: prompt=question  with some starter code and function name if they exist.
    We also specify the type of the prompt, i.e. whether it is call-based or standard input-based.
    """
    starter_code = None if len(doc["starter_code"]) == 0 else doc["starter_code"]
    try:
        input_outpout = json.loads(doc["input_output"])
        fn_name = None if not input_outpout.get("fn_name") else input_outpout["fn_name"]
    except ValueError:
        fn_name = None
    prompt = "QUESTION:\n"
    prompt += doc["question"]
    if starter_code:
        prompt += f"\nSTARTER CODE:\n{starter_code}"
    if not fn_name:
        call_format = "\nUse Standard Input format"
        prompt += call_format
    else:
        call_format = "\nUse Call-Based format"
        prompt += call_format
    prompt += "\nANSWER:\n"
    return prompt


def get_prompt_code_contest(doc):
    starter_code = None
    question = doc["description"]
    prompt = f"QUESTION:\n{question}\nSTARTER CODE:\n{starter_code}"
    prompt += "\nUse Standard Input format"
    prompt += "\nANSWER:\n"
    return prompt


def get_prompt_training(doc):
    starter_code = doc["starter_code"]
    question = doc["question"]
    prompt = f"QUESTION:\n{question}\nSTARTER CODE:\n{starter_code}"
    if starter_code is None:
        prompt += "\nUse Standard Input format"
    else:
        prompt += "\nUse Call-Based format"
    prompt += "\nANSWER:\n"
    return prompt


class DummyLayer(nn.Module):
    def __init__(self):
        super(DummyLayer, self).__init__()

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        return x


class MLP(nn.Module):
    def __init__(self, layers):
        """Multi layer percetron

        Args:
            layers (list): Specifying the number of nodes in each layer. e.g.,  [10, 20, 20, 10]
        """
        super(MLP, self).__init__()
        layer_dict = OrderedDict()
        last = len(layers) - 1
        for i in range(last):
            in_f, out_f = layers[i : i + 2]
            layer_dict[f"linear{i+1}"] = nn.Linear(in_f, out_f)
            if i != last - 1:
                layer_dict[f"selu{i+1}"] = nn.SELU()

        self.sequence = nn.Sequential(layer_dict)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        return self.sequence(x)


class AICoderConfig(dict):
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, dict):
                setattr(self, k, AICoderConfig(**v))
            else:
                setattr(self, k, v)
            self[k] = v

    def to_json(self, path):
        with open(path, "w") as f:
            json.dump(self, f)

    @staticmethod
    def from_file(file_path):
        with open(file_path, "r") as f:
            temp = json.load(f)

        return AICoderConfig(**temp)


class TagGeneratorConfig(dict):
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, dict):
                setattr(self, k, TagGeneratorConfig(**v))
            else:
                setattr(self, k, v)
            self[k] = v

    def to_json(self, path):
        with open(path, "w") as f:
            json.dump(self, f)

    @staticmethod
    def from_file(file_path):
        with open(file_path, "r") as f:
            temp = json.load(f)

        return TagGeneratorConfig(**temp)

MAPPER = {
    "APPS": get_prompt_APPS,
    "CodeContest": get_prompt_code_contest,
}

if __name__ == "__main__":
    pass
