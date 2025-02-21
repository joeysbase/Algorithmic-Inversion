import jsonlines
import torch
import os
import argparse
import time
import logging
import math
import json
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from accelerate import Accelerator
from AICoder import AICoder
from utils import AICoderConfig, get_prompt_training
from torch.utils.data.distributed import DistributedSampler
from random import shuffle


class AICoderDataset(Dataset):
    def __init__(self, input_path):
        self.data = []
        with jsonlines.open(input_path, mode="r") as reader:
            raw_data = [o for o in reader]
        # raw_data = raw_data[:128]
        count = 0
        for o in raw_data:
            if len(o["solution"]) == 0:
                count += 1
            else:
                self.data.append(o)
        print(
            f"valid datapoint {len(self.data)}, {count} datapoints were discarded for the absence of solution codes"
        )
        shuffle(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class DataCollatorForAICoder:
    def __init__(self, tokenizer, prefix_length):
        self.tokenizer = tokenizer
        self.prefix_length = prefix_length

    def __call__(self, batch_data):
        x = []
        y = []
        tags = []
        for o in batch_data:
            sol_code = o["solution"][0]
            target = f"```python\n{sol_code}\n```\n"
            source = f"{get_prompt_training(o)}{target}"
            x.append(source)
            y.append(target)
            tags.append(o["first_tire_tags"])
        source_encoded = self.tokenizer(x, padding=True, return_tensors="pt")
        target_encoded = self.tokenizer(y).input_ids
        
        batch = {
            "input_ids": source_encoded.input_ids,
            "attention_mask": source_encoded.attention_mask,
            "labels": target_encoded,
            "tags": tags,
        }

        return batch


class Config:
    batch_size = 32
    max_epochs = 3
    learning_rate = 1e-5
    input_path = ""
    output_path = ""
    from_pretrained = ""

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class Dummy_scheduler:
    def __init__(self, lr):
        self.lr = lr

    def step(self):
        pass

    def get_lr(self):
        return self.lr


class Trainer:
    def __init__(
        self,
        config,
        model,
        optimizer,
        lr_scheduler,
        logger,
        rank,
        accelerator,
        avg_loss_steps=5,
    ):
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.epoch_run = 0
        self.logger = logger
        self.rank = rank
        self.accelerator = accelerator
        self.avg_loss_steps = avg_loss_steps

    def save_chk_point(self, time_step, suffix=""):
        msg = f"----------------------------------------saving checkpoint at step {time_step} to {self.config.output_path}/checkpoint-{time_step}----------------------------------------"
        if self.rank == 0:
            print(msg)
        self.logger.info(msg)
        self.accelerator.save_state(f"{config.output_path}/{suffix}-{time_step}")

    def train(self, dataloader, save_every=100):
        batch_num = math.ceil(len(dataloader) / self.config.gradient_accumulation_steps)
        max_epochs = self.config.max_epochs

        self.model.train()
        total_steps = 1
        start_time = time.time()
        for epoch in range(self.epoch_run, max_epochs):
            epoch_loss = 0
            loss_for_avg = 0
            batch_loss = 0
            iters = math.ceil(len(dataloader) / self.config.gradient_accumulation_steps)
            batch_start_time = time.time()
            for step, inp in enumerate(dataloader, start=1):
                if step % self.config.gradient_accumulation_steps != 0 and step != len(
                    dataloader
                ):
                    with self.accelerator.no_sync(self.model):
                        input_ids = inp["input_ids"].to(self.model.device)
                        attention_mask = inp["attention_mask"].to(self.model.device)
                        labels = inp["labels"]
                        tags = inp["tags"]
                        loss = self.model(input_ids, attention_mask, tags, labels).loss
                        loss /= self.config.gradient_accumulation_steps
                        self.accelerator.backward(loss, retain_graph=True)
                else:
                    input_ids = inp["input_ids"].to(self.model.device)
                    attention_mask = inp["attention_mask"].to(self.model.device)
                    labels = inp["labels"]
                    tags = inp["tags"]
                    loss = self.model(input_ids, attention_mask, tags, labels).loss
                    loss /= self.config.gradient_accumulation_steps
                    self.accelerator.backward(loss, retain_graph=True)
                    self.optimizer.step()
                    self.lr_scheduler.step(epoch + (step - 1) / iters)
                    self.optimizer.zero_grad()

                epoch_loss += loss.item()
                batch_loss += loss.item()
                loss_for_avg += loss.item()
                total_steps += 1

                if step % self.config.gradient_accumulation_steps == 0 or step == len(
                    dataloader
                ):
                    batch_end_time = time.time()
                    effective_steps = math.ceil(
                        step / self.config.gradient_accumulation_steps
                    )

                    msg = f"< {effective_steps} of {batch_num} done || time elapsed {batch_end_time-batch_start_time} || current lr -> {self.lr_scheduler.get_lr()} || current loss -> {batch_loss} >"
                    if self.rank == 0:
                        print(msg)
                    self.logger.info(msg)

                    if effective_steps % self.avg_loss_steps == 0:
                        avg_loss = loss_for_avg / self.avg_loss_steps
                        loss_msg = f"================================= {self.avg_loss_steps} batch avg loss is {avg_loss} ================================="
                        self.logger.info(loss_msg)
                        if self.rank == 0:
                            print(loss_msg)
                        loss_for_avg = 0

                    batch_loss = 0
                    batch_start_time = time.time()

            if (epoch + 1) % save_every == 0 and (epoch + 1) != max_epochs:
                self.save_chk_point(epoch + 1, "chk")

            msg = f"------------------------------------ {epoch+1} of {max_epochs} done, epoch loss -> {epoch_loss/batch_num} ------------------------------------"
            if self.rank == 0:
                print(msg)
            self.logger.info(msg)

        end_time = time.time()
        msg = f"------------------------------- training is done, time elapsed on proc {self.rank} is {end_time-start_time}, saving model to {self.config.output_path} now -------------------------------"
        if self.rank == 0:
            print(msg)
        self.logger.info(msg)

        self.accelerator.wait_for_everyone()
        self.model.dump_MLP()
        self.accelerator.save_model(
            self.model,
            config.output_path,
            max_shard_size="5GB",
            safe_serialization=False,
        )
        # self.accelerator.save_state(config.output_path)

    def load_chk_point(self):
        chkpoint = torch.load(f"{self.chk_path}/checkpoint.pt")
        self.model.load_state_dict(chkpoint["MODEL_STATE"])
        self.optimizer.load_state_dict(chkpoint["OPTIMIZER_STATE"])
        self.lr_scheduler.load_state_dict(chkpoint["SCHEDULER_STATE"])
        self.epoch_run = chkpoint["EPOCH_RUN"]
        print(
            f"---------------------restoring training process from {self.chk_path}/checkpoint.pt at epoch {self.epoch_run}---------------------"
        )


def main(config):
    rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    print(f"this is proc {rank}")

    logger = logging.getLogger(__name__)
    logging.basicConfig(
        filename=f"{config.log_path}/training-{rank}.log",
        filemode="w",
        encoding="utf-8",
        level=logging.DEBUG,
    )

    accelerator = Accelerator(project_dir=config.output_path)
    device = accelerator.device

    with open(
        f"./configs/algorithm_embedding_configs/{config.prefix_length}_{config.length_type}.json",
        "r",
    ) as f:
        prefix_config = json.load(f)

    try:
        model_config = AICoderConfig.from_file("./configs/model_configs/aicoder_config.json")
        model_config["decoder_path"] = config.base_model_path
        model_config["prefix_config"] = prefix_config
        model_config["position"] = config.position
        model = AICoder(model_config, torch_dtype=torch.bfloat16).to(device)
        model.decoder.gradient_checkpointing_enable()

        tokenizer = AutoTokenizer.from_pretrained(
            model_config["decoder_path"],
            trust_remote_code=True,
        )
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        training_data = AICoderDataset(config.input_path)

        collator = DataCollatorForAICoder(tokenizer, config.prefix_length)
        dataloader = DataLoader(
            training_data,
            collate_fn=collator,
            sampler=DistributedSampler(training_data),
            batch_size=config.per_device_batch_size,
            shuffle=False,
            num_workers=world_size,
        )
        optimizer = AdamW(model.parameters(), lr=config.learning_rate)
        lr_scheduler = CosineAnnealingWarmRestarts(optimizer, 2, 2)

        model, optimizer, lr_scheduler = accelerator.prepare(
            model, optimizer, lr_scheduler
        )
        # lr_scheduler = Dummy_scheduler(config.learning_rate)

        trainer = Trainer(
            config=config,
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            logger=logger,
            rank=rank,
            accelerator=accelerator,
            avg_loss_steps=config.avg_loss_steps,
        )
        trainer.train(dataloader=dataloader, save_every=config.save_every)

        model_config["require_MLP"] = False
        model_config["init_from_embedding"] = False
        with open(f"{config.output_path}/config.json", "w") as f:
            json.dump(model_config, f)
        tokenizer.save_pretrained(config.output_path)

    except Exception as e:
        logger.error(e)
        raise e


if __name__ == "__main__":
    argp = argparse.ArgumentParser()
    argp.add_argument("--input_path", default=None, type=str)
    argp.add_argument("--output_path", default=None, type=str)
    argp.add_argument("--base_model_path", default=None, type=str)
    argp.add_argument("--log_path", default="", type=str)
    argp.add_argument("--learning_rate", default=5e-4, type=float, help="learning rate")
    argp.add_argument("--max_epochs", default=3, type=int)
    argp.add_argument("--per_device_batch_size", default=16, type=int)
    argp.add_argument("--gradient_accumulation_steps", default=1, type=int)
    argp.add_argument("--save_every", default=100, type=int)
    argp.add_argument("--max_length", default=2048, type=int)
    argp.add_argument("--avg_loss_steps", default=10, type=int)
    argp.add_argument("--prefix_length", default=1, type=int)
    argp.add_argument("--length_type", default="ratio_codelen", type=str)
    argp.add_argument("--position", default="prefix", type=str)
    args = argp.parse_args()

    rank = int(os.environ["LOCAL_RANK"])

    if rank == 0:
        if not os.path.exists(args.output_path):
            os.mkdir(args.output_path)

        if not os.path.exists(args.log_path):
            os.mkdir(args.log_path)
    else:
        while not os.path.exists(args.log_path):
            pass

    config = Config(**{k: v for k, v in args._get_kwargs()})

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    main(config)


"""
CUDA_VISIBLE_DEVICES="6,7" accelerate launch ./sources/train_AICoder.py \
    --input_path ./datasets/training/train-dedup-7005.jsonl \
    --output_path ./models/qwen/AICoder-67-ratio-codelen-prefix \
    --base_model_path ./configs/model_configs/aicoder_config.json  \
    --log_path ./logs/qwen/AICoder-67-ratio-codelen-prefix \
    --learning_rate 1e-3 \
    --max_epochs 10 \
    --per_device_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --max_length 4096 \
    --prefix_length 67 \
    --avg_loss_steps 10 \
    --save_every 10
"""
