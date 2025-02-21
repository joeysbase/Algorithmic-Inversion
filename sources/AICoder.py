import torch
import torch.nn as nn
import os
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
from accelerate import load_checkpoint_in_model
from model_sources.utils import AICoderConfig, DummyLayer, MLP


class AlgorithmEmbeddingLayer(nn.Module):
    def __init__(
        self,
        prefix_config_dict,
        dimension,
        tag_rep_scheme="full",
        MLP_layers=None,
        reparametrization=True,
        init_embedding=None,
    ):
        super(AlgorithmEmbeddingLayer, self).__init__()
        self.tag_to_info = prefix_config_dict
        self.tag_rep_scheme = tag_rep_scheme
        self.id_to_tag = None
        self.shape = None  # (L,D)
        self._indexing(dimension)
        if reparametrization:
            self.MLP = MLP(MLP_layers)
            self.embedding = nn.Parameter(torch.randn((self.shape[0], MLP_layers[0])))
        else:
            self.MLP = DummyLayer()

            # Initialise algorithm embedding randomly or from the decoder embedding
            if init_embedding is None:
                self.embedding = nn.Parameter(torch.randn(self.shape))
            else:
                # torch.manual_seed(0)
                embedding = init_embedding(torch.randint(0, 10000, (self.shape[0],)))
                self.embedding = nn.Parameter(embedding)

    def __call__(self, tags):
        return self.forward(tags)

    def forward(self, tags):
        embedding = self.MLP(self.embedding)
        embed_list = []  # (B,L,D)
        attention_mask = []
        for tag in tags:
            tmp = {}
            if self.tag_rep_scheme == "partial":
                for t in tag:
                    start = self.tag_to_info[t]["start"]
                    shift = self.tag_to_info[t]["shift"]
                    tmp[start] = embedding[start : start + shift]
                tmp = dict(sorted(tmp.items()))
                tmp = [v for k, v in tmp.items()]
                attention_mask = None
                if len(tmp) == 0:
                    embed_list.append(torch.empty((0, self.shape[1])).to(self.device))
                else:
                    embed_list.append(torch.cat(tmp, dim=0))
            elif self.tag_rep_scheme == "full":
                if len(tag) == 0:
                    attention_mask.append([])
                    embed_list.append(torch.empty((0, self.shape[1])).to(self.device))
                    continue
                mask = [0 for _ in range(self.shape[0])]
                for t in tag:
                    start = self.tag_to_info[t]["start"]
                    shift = self.tag_to_info[t]["shift"]
                    mask[start : start + shift] = [1 for _ in range(shift)]
                attention_mask.append(mask)
                embed_list.append(embedding)
            else:
                raise Exception("invalid representation scheme!")
        return embed_list, attention_mask

    @torch.no_grad()
    def dump_MLP(self):
        embedding = self.MLP(self.embedding)
        self.embedding = nn.Parameter(embedding)
        self.MLP = DummyLayer()

    def _indexing(self, dimension):
        id_to_tag = {}
        length = 0
        for k, v in self.tag_to_info.items():
            id_to_tag[v["abs_pos"]] = k
            length += v["shift"]

        accumulate = 0
        for k, v in id_to_tag.items():
            self.tag_to_info[v]["start"] = accumulate
            accumulate += self.tag_to_info[v]["shift"]

        self.id_to_tag = id_to_tag
        self.shape = (length, dimension)

    @property
    def device(self):
        return self.embedding.device


class AICoder(nn.Module):
    def __init__(self, config, load_decoder_paras=True, torch_dtype=torch.bfloat16):
        super(AICoder, self).__init__()
        decoder_config = AutoConfig.from_pretrained(
            config.decoder_path, trust_remote_code=True
        )
        if load_decoder_paras:
            self.decoder = AutoModelForCausalLM.from_pretrained(config.decoder_path).to(
                torch_dtype
            )
        else:
            self.decoder = AutoModelForCausalLM.from_config(decoder_config).to(
                torch_dtype
            )
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.decoder_path, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.embedding = self.decoder.get_input_embeddings()
        self.pad_embedding = self.embedding(torch.tensor(self.tokenizer.pad_token_id))
        self.position = config.position
        self.tag_rep_scheme = config.tag_rep_scheme

        for p in self.decoder.parameters():
            p.requires_grad = False

        dimension = (
            decoder_config.hidden_size
            if config.dimension == "auto"
            else config.dimension
        )
        init_from_embedding = self.embedding if config.init_from_embedding else None
        if config.require_MLP:
            self.AE = AlgorithmEmbeddingLayer(
                config.prefix_config,
                dimension,
                tag_rep_scheme=config.tag_rep_scheme,
                MLP_layers=config.MLP,
                reparametrization=True,
                init_embedding=init_from_embedding,
            ).to(torch_dtype)
        else:
            self.AE = AlgorithmEmbeddingLayer(
                config.prefix_config,
                dimension,
                tag_rep_scheme=config.tag_rep_scheme,
                reparametrization=False,
                init_embedding=init_from_embedding,
            ).to(torch_dtype)

    def __call__(self, input_ids, attention_mask, tags, labels, generate=False):
        return self.forward(input_ids, attention_mask, tags, labels, generate)

    def _rm_pad_embedding(self, input_embeddings, attention_mask):
        unpadded_embeddings = []
        for attnmsk, embed in zip(attention_mask, input_embeddings):
            last = 0
            for i, value in enumerate(attnmsk, start=1):
                if value == 0:
                    last = i
                else:
                    break
            unpadded_embeddings.append(embed[last:])
        return unpadded_embeddings

    def _pad_input_embeds(self, unpadded_embeddings, tag_embeds_attnmsk):
        self.pad_embedding = self.pad_embedding.to(self.device)
        embed_length = [embed.shape[0] for embed in unpadded_embeddings]
        max_length = max(embed_length)
        attention_mask = []
        padded_embeddings = []
        flag = True
        if tag_embeds_attnmsk is None:
            tag_embeds_attnmsk = [1 for _ in range(len(embed_length))]
            flag = False
        for l, embed, tea in zip(embed_length, unpadded_embeddings, tag_embeds_attnmsk):
            diff = max_length - l
            tmp = [1 for _ in range(max_length)]
            tmp[:diff] = [0 for _ in range(diff)]
            if flag:
                tmp[diff : diff + len(tea)] = tea
            padded_embeddings.append(
                torch.cat(
                    [self.pad_embedding.expand(diff, -1), embed], dim=0
                ).unsqueeze(dim=0)
            )
            attention_mask.append(tmp)
        padded_embeddings = torch.cat(padded_embeddings, dim=0).to(self.dtype)
        attention_mask = torch.tensor(attention_mask).to(self.device)
        return padded_embeddings, attention_mask

    def _prepare_labels(self, unpadded_labels, padded_input_embeddings):
        padded_labels = []
        for ul, pie in zip(unpadded_labels, padded_input_embeddings):
            diff = pie.shape[0] - len(ul)
            padded_labels.append([-100] * diff + ul)
        return torch.tensor(padded_labels).to(self.device)

    def forward(
        self, input_ids, attention_mask, tags, labels=None, generate=False, **kwargs
    ):
        question_embedding = self.embedding(input_ids)  # (B, L_inp, D), padded
        if tags is None:
            inputs_embeds = question_embedding
        else:
            stripped_input_embeds = self._rm_pad_embedding(
                question_embedding, attention_mask
            )
            tag_embeds, tag_embed_attnmsk = self.AE(tags)
            unpadded_input_embeddings = []
            for sie, te in zip(stripped_input_embeds, tag_embeds):
                if self.position == "prefix":
                    unpadded_input_embeddings.append(torch.cat([te, sie], dim=0))
                elif self.position == "infix":
                    unpadded_input_embeddings.append(torch.cat([sie, te], dim=0))
                else:
                    raise Exception("invalid position")
        inputs_embeds, new_attention_mask = self._pad_input_embeds(
            unpadded_input_embeddings,
            tag_embed_attnmsk,
        )
        if generate:
            output = self.decoder.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=new_attention_mask,
                **kwargs,
            )
        else:
            padded_labels = self._prepare_labels(labels, inputs_embeds)
            output = self.decoder(
                inputs_embeds=inputs_embeds,
                attention_mask=new_attention_mask,
                labels=padded_labels,
                return_dict=True,
            )
        return output

    def generate(self, input_ids, attention_mask, tags=None, **kwargs):
        return self.forward(input_ids, attention_mask, tags, generate=True, **kwargs)

    def dump_MLP(self):
        self.AE.dump_MLP()

    @staticmethod
    def from_pretrained(path, torch_dtype=torch.bfloat16, load_decoder_paras=False):
        config = AICoderConfig.from_file(f"{path}/config.json")
        model = AICoder(config, load_decoder_paras=load_decoder_paras)
        if not os.path.exists(f"{path}/pytorch_model.bin.index.json"):
            state_dict = torch.load(f"{path}/pytorch_model.bin")
            model.load_state_dict(state_dict)
            model.to(torch_dtype)
        else:
            load_checkpoint_in_model(model, path, dtype=torch_dtype)

        return model

    @property
    def device(self):
        return self.decoder.device

    @property
    def dtype(self):
        return self.decoder.dtype


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(
        ""
    )
    tokenizer.padding_side = "left"
    text = [
        "I felt quite hungry.",
        "We are outlaws, not idiots!",
    ]
    label = ["quite hungry.", "not idiots!"]
    inp = tokenizer(text, padding=True, return_tensors="pt")
    l = tokenizer(label).input_ids

    config = AICoderConfig.from_file(
        "./configs/model_configs/aicoder_config.json"
    )

    model = AICoder(config)
    a = model.generate(
        inp.input_ids,
        inp.attention_mask,
        tags=[
            ["math", "constructive algorithm", "implementation"],
            ["fft", "data structure"],
        ],
        max_new_tokens=10,
    )
    a = tokenizer.batch_decode(a)
    print(a)
