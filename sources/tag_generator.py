import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from transformers import AutoModel, AutoTokenizer
from accelerate import load_checkpoint_in_model
from utils import TagGeneratorConfig, MLP


class TagGenerator(nn.Module):
    def __init__(self, config, freeze_encoder=True):
        super(TagGenerator, self).__init__()
        self.num_heads = int(len(config.label_mapping) / 2)
        self._indexing(label_mapping=config.label_mapping)

        # Classification heads can be configured here
        self.linear_heads = nn.ModuleList(
            [MLP([1024, 2048, 2048, 1024, 768, 512, 1]) for _ in range(self.num_heads)]
        )

        self.embedding_model = AutoModel.from_pretrained(config.embedding_model)
        if freeze_encoder:
            for p in self.embedding_model.parameters():
                p.requires_grad = False

    def __call__(self, x, y=None):
        return self.forward(x, y)

    def forward(self, x, y):
        embeddings = self.embedding_model(**x).last_hidden_state[:, 0]  # (B,D)
        raw = torch.cat(
            [linear(embeddings) for linear in self.linear_heads], dim=1
        )  # (B,L)
        logits = F.sigmoid(raw)  # (B,L)
        if y is None:
            return logits
        else:
            trans = torch.ones_like(y)  # (B,L)
            y_trans = trans - y
            s = y_trans - logits
            log_s = torch.log(torch.abs(s))
            loss = torch.sum(log_s) / (y.shape[0] * y.shape[1])
            return -loss

    @torch.no_grad()
    def generate(self, x, threshold=0.95):
        logits = self.forward(x, y=None)
        # print(logits)
        bool_selector_matrix = logits >= threshold
        tags = []
        for bool_selector_vector in bool_selector_matrix:
            tmp = []
            for i, v in enumerate(bool_selector_vector):
                if v:
                    tmp.append(self.label_text_ordered[i])
            tags.append(tmp)
        return tags

    @staticmethod
    def from_pretrained(path, torch_dtype=torch.bfloat16):
        config = TagGeneratorConfig.from_file(f"{path}/config.json")
        model = TagGenerator(config)
        if not os.path.exists(f"{path}/pytorch_model.bin.index.json"):
            state_dict = torch.load(f"{path}/pytorch_model.bin", map_location="cpu")
            model.load_state_dict(state_dict)
            model.to(torch_dtype)
        else:
            load_checkpoint_in_model(model, path, dtype=torch_dtype)
        return model

    def _indexing(self, label_mapping):
        self.label_text_ordered = [label_mapping[str(i)] for i in range(self.num_heads)]

    @property
    def device(self):
        return self.embedding_model.device


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(
        ""
    )
    config = TagGeneratorConfig.from_file("./configs/model_configs/tag_generator_config.json")
    model = TagGenerator(config)
    text = [
        "I felt quite hungry.",
        "We are outlaws, not idiots!",
    ]
    inp = tokenizer(text, padding=True, return_tensors="pt")
    a = model.generate(inp)
    print(a)
    
