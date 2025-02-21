import argparse
from tag_generator import TagGenerator
from transformers import AutoTokenizer
from tqdm import trange
import jsonlines
import math
import os

"""
python ./sources/tag_generator_inference.py \
    --model_type tag_generator/100-epochs \
    --output_path ./inference_results/tag-generator \
    --device 7 \
    --batch_size 256 \
    --threshold 0.97
"""
if __name__ == "__main__":
    argp = argparse.ArgumentParser()
    argp.add_argument("--model_type", default="", type=str)
    argp.add_argument("--device", default=0, type=int)
    argp.add_argument("--batch_size", default=6, type=int)
    argp.add_argument("--threshold", default=0.95, type=float)
    argp.add_argument("--output_path", default="", type=str)
    args = argp.parse_args()

    device = args.device
    batch_size = args.batch_size
    threshold = args.threshold

    with jsonlines.open(
        "./datasets/training/test.jsonl",
        mode="r",
    ) as reader:
        data = [o for o in reader]
    tokenizer = AutoTokenizer.from_pretrained(f"./models/{args.model_type}")
    model = TagGenerator.from_pretrained(f"./models/{args.model_type}").to(
        f"cuda:{device}"
    )

    batch_num = math.ceil(len(data) / batch_size)
    results = []
    for i in trange(batch_num):
        batch = data[i * batch_size : (i + 1) * batch_size]
        text = [o["question"] for o in batch]
        tokenized = tokenizer(
            text, padding=True, truncation=True, return_tensors="pt"
        ).to(f"cuda:{device}")
        tags = model.generate(tokenized, threshold=threshold)
        for o, t in zip(batch, tags):
            results.append({"label": o["first_tire_tags"], "prediction": t})

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    with jsonlines.open(
        f"{args.output_path}/tag-generator-{threshold}.jsonl",
        mode="w",
    ) as writer:
        writer.write_all(results)
