from transformers import AutoTokenizer
from AICoder import AICoder
from utils import MAPPER
import torch.multiprocessing as mp
import torch
import jsonlines
import os
import math
import time
import json
import logging
import argparse


def inference_proc(
    ids,
    inp,
    batch_size,
    devices_lst,
    model_path,
    resume_meta_path,
    num_sample,
    log_path,
    obj_fn,
    comu,
    progress,
    lock,
    args,
):
    proc_start_time = time.time()
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        filename=f"{log_path}inference-{ids}.log",
        filemode="w",
        encoding="utf-8",
        level=logging.DEBUG,
    )

    device = devices_lst[ids].replace("$", ",")
    # os.environ["CUDA_VISIBLE_DEVICES"] = device
    batch_num = math.ceil(len(inp) / batch_size)

    if os.path.exists(resume_meta_path):
        with open(f"{resume_meta_path}", "r") as f:
            resume_meta = json.load(f)

        num_finished = resume_meta["num_finished"]
        batch_num = math.ceil(len(inp[num_finished:]) / batch_size)

        msg = f"resuming from checkpoint where {num_finished} datapoints were finished"
        print(msg)
        logger.info(msg)
    else:
        num_finished = 0

    print(f"proc {ids} using cuda:{device}")

    try:
        model = AICoder.from_pretrained(model_path, torch.bfloat16).to(f"cuda:{device}")
        # model.generation_config.cache_implementation = "static"
        model = torch.compile(model, mode="reduce-overhead", fullgraph=True)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    except Exception as e:
        logger.error(e)
        raise e
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()

    while True:
        lock.acquire()
        curr_batch_num = progress.value
        comu[ids] = curr_batch_num + 1
        if curr_batch_num >= batch_num:
            lock.release()
            break
        else:
            progress.value += 1
            print(
                f"---------------------------- proc {ids} is processing batch {curr_batch_num+1}, {batch_num-curr_batch_num-1} left ----------------------------"
            )
            lock.release()

        try:
            batch_start_time = time.time()
            results = []
            obj_batch = inp[
                curr_batch_num * batch_size
                + num_finished : (curr_batch_num + 1) * batch_size
                + num_finished
            ]
            question_lst = [obj_fn(o) for o in obj_batch]
            tags = [o["first_tire_tags"] for o in obj_batch]

            done = False
            sub_batch_size = len(obj_batch)
            while not done:
                sub_batch_num = math.ceil(len(obj_batch) / sub_batch_size)
                try:
                    for i in range(sub_batch_num):
                        sub_question_lst = question_lst[
                            i * sub_batch_size : (i + 1) * sub_batch_size
                        ]
                        sub_tags = tags[i * sub_batch_size : (i + 1) * sub_batch_size]
                        tokenized = tokenizer(
                            sub_question_lst,
                            padding=True,
                            truncation=True,
                            max_length=4096,
                            return_tensors="pt",
                        ).to(model.device)
                        output = model.generate(
                            input_ids=tokenized.input_ids,
                            attention_mask=tokenized.attention_mask,
                            tags=sub_tags,
                            temperature=args.temperature,
                            num_return_sequences=num_sample,
                            do_sample=True,
                            top_k=35,
                            max_new_tokens=1024,
                            pad_token_id=tokenizer.eos_token_id,
                        )
                        output_decoded = tokenizer.batch_decode(
                            output, skip_special_tokens=True
                        )
                        tmp = []
                        for j in range(len(output_decoded)):
                            tmp.append(output_decoded[j])
                            if (j + 1) % num_sample == 0:
                                results.append([tmp, question_lst[j // num_sample]])
                                tmp = []

                        for o, r in zip(
                            obj_batch[i * sub_batch_size : (i + 1) * sub_batch_size],
                            results,
                        ):
                            o["model_solutions"] = r[0]
                            o["model_input"] = r[1]
                    done = True
                except torch.cuda.OutOfMemoryError as e:
                    sub_batch_size -= 1
                    if sub_batch_size == 0:
                        print(
                            "input text is too large, a single gpu cannot even fit with batch size 1!"
                        )
                        raise e

            batch_end_time = time.time()
            msg = f"------------------------------- batch {curr_batch_num} done, time elapsed {batch_end_time-batch_start_time} -------------------------------"
            logger.info(msg)
            # if ids == 0:
            #     msg = f"------------------------------- {batch_finished} of {batch_num} done, {batch_num-batch_finished} batch left, time elapsed {batch_end_time-batch_start_time} -------------------------------"
            #     print(msg)
        except Exception as e:
            logger.error(e)
            raise e

    proc_end_time = time.time()
    msg = f"------------------------------- inference at proc{ids} is done, time elapsed {proc_end_time-proc_start_time} -------------------------------"
    logger.info(msg)
    print(msg)


"""
python ./sources/AICoder_inference.py \
    --model_type "qwen/AICoder-67-ratio-codelen-prefix" \
    --metric "CodeContest" \
    --split "test" \
    --temperature 0.3 \
    --devices 3,4,5,6,7 \
    --batch_size 8 \
    --num_samples 5
"""

if __name__ == "__main__":
    argp = argparse.ArgumentParser()
    argp.add_argument("--model_type", default="", type=str)
    argp.add_argument("--metric", default="code_contest", type=str)
    argp.add_argument("--split", default="test", type=str)
    argp.add_argument("--temperature", default=0.3, type=float)
    argp.add_argument("--devices", default="0,1,2,3,4,5,6,7", type=str)
    argp.add_argument("--batch_size", default=6, type=int)
    argp.add_argument("--num_samples", default=10, type=int)
    args = argp.parse_args()

    model_path = f"./models/{args.model_type}"
    input_path = f"./datasets/{args.metric}/{args.split}.jsonl"
    output_path = f"./inference_results/{args.model_type}/{args.metric}_{args.split}set_inference_results_{args.temperature}.jsonl"
    log_path = f"./logs/{args.model_type}/"
    resume_meta_path = f"./inference_results/{args.model_type}/{args.metric}_{args.split}set_inference_results-resume-meta_{args.temperature}.json"

    devices = args.devices
    batch_size = args.batch_size
    num_sample = args.num_samples
    print(f"{args.model_type} || {args.metric} || {args.split} || {args.temperature}")

    try:
        get_prompt_fn = MAPPER[args.metric]
    except:
        raise Exception(f"no available get_prompt_fn for metric {args.metric}!")

    out_lst = output_path.split("/")[:-1]
    output_dir = "/".join(out_lst)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if not os.path.exists(log_path):
        os.makedirs(log_path)

    devices_lst = devices.split(",")
    world_size = len(devices_lst)

    start_time = time.time()
    try:
        manager = mp.Manager()
        comu = manager.dict()
        progress = manager.Value("i", 0)
        lock = manager.Lock()
        exception = None

        if os.path.exists(resume_meta_path):
            with open(resume_meta_path, "r") as f:
                inp = json.load(f)["data"]
                inp = [manager.dict(o) for o in inp]
        else:
            with jsonlines.open(input_path, mode="r") as reader:
                inp = [manager.dict(o) for o in reader]

        shared_inp = manager.list(inp)

        mp.spawn(
            fn=inference_proc,
            args=(
                inp,
                batch_size,
                devices_lst,
                model_path,
                resume_meta_path,
                num_sample,
                log_path,
                get_prompt_fn,
                comu,
                progress,
                lock,
                args,
            ),
            nprocs=world_size,
        )
    except KeyboardInterrupt as e:
        exception = e
    except Exception as e:
        exception = e

    end_time = time.time()

    if exception is not None:
        print(exception)
        batch_finished = min([v for k, v in comu.items()]) - 1
        num_finished = batch_size * batch_finished
        try:
            with open(f"{resume_meta_path}", "r") as f:
                resume_meta = json.load(f)
            num_finished += resume_meta["num_finished"]
        except:
            pass
        with open(
            f"./inference_results/{args.model_type}/{args.metric}_{args.split}set_inference_results-resume-meta_{args.temperature}.json",
            "w",
        ) as f:
            partly_finished = list(shared_inp)
            partly_finished = [dict(o) for o in partly_finished]
            json.dump({"num_finished": num_finished, "data": partly_finished}, f)

        print(
            f"encountered {exception}, checkpoint has been stored for resuming inference."
        )
        raise exception

    print(
        f"------------------------------- inference is done, time elapsed {end_time-start_time}, merging files now -------------------------------"
    )

    finished = list(shared_inp)
    finished = [dict(o) for o in finished]
    with jsonlines.open(output_path, mode="w") as writer:
        writer.write_all(finished)

    print("all done!")
