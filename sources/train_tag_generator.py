import jsonlines
import torch
import os
import torch.multiprocessing as mp
import argparse
import json
import time
import logging
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from transformers import AdamW
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from tag_generator import TagGenerator
from utils import TagGeneratorConfig


def pad_label(sequence, max_length, padding_value):
    padded = []
    for s in sequence:
        s = s.tolist()
        pending_pad_len = max_length - len(s)
        s = [padding_value for i in range(pending_pad_len)] + s
        padded.append(s)
    return torch.tensor(padded)

class APPS(Dataset):
    def __init__(self, input_path):
        self.data = []
        with jsonlines.open(input_path, mode="r") as reader:
            self.data = [o for o in reader]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class TagGeneratorDataset(Dataset):
    def __init__(self, input_path):
        self.data = []
        with jsonlines.open(input_path, mode="r") as reader:
            self.data = [o for o in reader]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class Dummy_scheduler:
    def __init__(self, lr):
        self.lr = lr

    def step(self):
        pass

    def get_lr(self):
        return self.lr


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


class Trainer:
    def __init__(self, config, model, optimizer, lr_scheduler, rank, logger):
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.epoch_run = 0
        self.rank = rank
        self.logger = logger

    def save_chk_point(self, time_step):
        os.mkdir(f"{self.config.output_path}/checkpoint-{time_step}")
        print(
            f"----------------------------------------saving checkpoint at step {time_step} to {self.config.output_path}/checkpoint-{time_step}----------------------------------------"
        )
        self.logger.info(
            f"----------------------------------------saving checkpoint at step {time_step} to {self.config.output_path}/checkpoint-{time_step}----------------------------------------"
        )
        self.model.save_pretrained(f"{self.config.output_path}/checkpoint-{time_step}")

    def train(self, dataloader):
        self.model.train()
        total_steps = 0
        start_time = time.time()
        for epoch in range(self.epoch_run, self.config.max_epochs):
            batch_num = len(dataloader)
            total_loss = 0
            for step, (source, target) in enumerate(dataloader, start=1):
                batch_start_time = time.time()
                self.optimizer.zero_grad()
                target = target.to(self.model.device)
                source = source.to(self.model.device)
                loss = self.model(source, target)
                batch_loss = loss.item()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                self.lr_scheduler.step()

                total_steps += step

                if self.rank == 0:
                    batch_end_time = time.time()
                    print(
                        f"< current epoch -> {epoch} || {step} of {batch_num} done || time elapsed {batch_end_time-batch_start_time} || current lr -> {self.lr_scheduler.get_lr()} || batch loss -> {batch_loss} >"
                    )
                    self.logger.info(
                        f"< current epoch -> {epoch} || {step} of {batch_num} done || time elapsed {batch_end_time-batch_start_time} || current lr -> {self.lr_scheduler.get_lr()} || current loss -> {batch_loss} >"
                    )
            if self.rank == 0:
                print(
                    f"------------------------------------ {epoch+1} of {self.config.max_epochs} done, total loss {total_loss/batch_num} ------------------------------------"
                )
                self.logger.info(
                    f"------------------------------------ {epoch+1} of {self.config.max_epochs} done, total loss {total_loss/batch_num} ------------------------------------"
                )
        end_time = time.time()
        if self.rank == 0:
            print(
                f"------------------------------- training is done, time elapsed {end_time-start_time}, saving model to {self.config.output_path} now -------------------------------"
            )
            self.logger.info(
                f"------------------------------- training is done, time elapsed {end_time-start_time}, saving model to {self.config.output_path} now -------------------------------"
            )
            torch.save(
                self.model.module.state_dict(),
                f"{self.config.output_path}/pytorch_model.bin",
            )

    def load_chk_point(self):
        chkpoint = torch.load(f"{self.chk_path}/checkpoint.pt")
        self.model.load_state_dict(chkpoint["MODEL_STATE"])
        self.optimizer.load_state_dict(chkpoint["OPTIMIZER_STATE"])
        self.lr_scheduler.load_state_dict(chkpoint["SCHEDULER_STATE"])
        self.epoch_run = chkpoint["EPOCH_RUN"]
        print(
            f"---------------------restoring training process from {self.chk_path}/checkpoint.pt at epoch {self.epoch_run}---------------------"
        )


def main(ids, config, device_lst, world_size, model_config, tokenizer):
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        filename=f"{config.log_path}/training-{ids}.log",
        filemode="w",
        encoding="utf-8",
        level=logging.DEBUG,
    )

    device_id = int(device_lst[ids])
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    # torch.cuda.set_device(device_id)
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)

    def collate_fn(batch_samples):
        x = tokenizer(
            [o["question"] for o in batch_samples],
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        y = torch.tensor([o["tagger_label"] for o in batch_samples])
        return x, y

    print(f"process {ids} using cuda:{device_id}")
    try:
        init_process_group(backend="nccl", rank=ids, world_size=world_size)

        training_data = TagGeneratorDataset(config.input_path)
        dist_dataloader = DataLoader(
            training_data,
            collate_fn=collate_fn,
            sampler=DistributedSampler(training_data),
            batch_size=config.batch_size,
            shuffle=False,
        )

        # model = AutoModelForCausalLM.from_pretrained(config.model_path).to("cuda")
        model = DDP(
            TagGenerator(model_config, freeze_encoder=True).to(f"cuda:{device_id}"),
            device_ids=[device_id],
        )
        print("------------- loading done -------------")

        optimizer = AdamW(model.parameters(), lr=config.learning_rate)
        # lr_scheduler = get_scheduler(
        #     config.scheduler,
        #     optimizer=optimizer,
        #     num_warmup_steps=config.warmup_steps,
        #     num_training_steps=config.max_epochs * len(dist_dataloader),
        # )
        lr_scheduler = Dummy_scheduler(lr=config.learning_rate)

        trainer = Trainer(
            config=config,
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            rank=ids,
            logger=logger,
        )
        trainer.train(dataloader=dist_dataloader)
    except Exception as e:
        logger.error(e)
        destroy_process_group()
        raise e

    destroy_process_group()


if __name__ == "__main__":
    argp = argparse.ArgumentParser()
    argp.add_argument("--input_path", default=None, type=str)
    argp.add_argument("--output_path", default=None, type=str)
    argp.add_argument("--base_model_path", default=None, type=str)
    argp.add_argument("--log_path", default="", type=str)
    argp.add_argument("--learning_rate", default=5e-4, type=float, help="learning rate")
    argp.add_argument("--max_epochs", default=600, type=int)
    argp.add_argument("--batch_size", default=16, type=int)
    argp.add_argument("--warmup_steps", default=50, type=int)
    argp.add_argument("--scheduler", default="linear", type=str)
    argp.add_argument("--devices", default="0", type=str)
    args = argp.parse_args()

    model_config = TagGeneratorConfig.from_file("./configs/model_configs/tag_generator_config.json")
    model_config["embedding_model"] = args.base_model_path
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)
    tokenizer.padding_side = "left"

    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)
    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)

    device_lst = args.devices.split(",")
    world_size = len(device_lst)

    config = Config(**{k: v for k, v in args._get_kwargs()})

    mp.spawn(
        main, args=(config, device_lst, world_size, model_config, tokenizer), nprocs=world_size
    )

    with open(f"{config.output_path}/config.json", "w") as f:
        json.dump(model_config, f)
    tokenizer.save_pretrained(config.output_path)


"""
python ./sources/train_tag_generator.py \
    --input_path ./datasets/training/train-dedup-7005.jsonl \
    --output_path ./models/tag_generator/50-epochs \
    --base_model_path ./models/bge/bge-large-en-v1.5\
    --log_path ./logs/tag_generator/50-epochs \
    --learning_rate 5e-4 \
    --max_epochs 50 \
    --batch_size 256 \
    --devices "0,1"
"""
