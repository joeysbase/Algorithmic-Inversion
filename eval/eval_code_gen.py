import argparse
from evaluate import load
import code_contest_metric
import jsonlines
import json
import os
import time
import re


def post_process(generations):
    gens = []
    for gen in generations:
        tmp = []
        for i, c in enumerate(gen):
            try:
                code = c.split("\nANSWER:\n")[1]
            except IndexError:
                # tmp.append(c)
                # continue
                code = c
            locs = re.findall("```[P,p]ython3*\n|```\n", code)
            if len(locs) == 0:
                tmp.append(code)
            else:
                idx = code.find(locs[0])
                shift = len(locs[0])
                code = code[idx + shift :]
                end = code.find("```")
                end = len(code) if end == -1 else end
                tmp.append(code[:end])
        gens.append(tmp)
    return gens


def main(input_path, output_path, split, metric_type, suffix):
    with jsonlines.open(input_path, mode="r") as reader:
        r = [o for o in reader]

    generations = [o["model_solutions"] for o in r]
    generations = post_process(generations)
    # print(len(generations[0]))

    if metric_type == "APPS":
        apps_metric = load("./eval/apps_metric")
        metric, results = apps_metric.compute(
            predictions=generations,
            k_list=[1, 5, 10],
            split="test",
            level="all",
            debug=False,
        )
    elif metric_type == "CodeContest":
        metric, results = code_contest_metric.compute_metric(
            generations=generations,
            k_list=[1, 5, 10],
            split=split,
            num_process=10,
            debug=False,
            num_tasks="all",
        )
    else:
        print("metric not supported")
        return

    with open(f"{output_path}{metric_type}-{split}set-stat-{suffix}.json", "w") as f:
        json.dump(results, f)

    with open(f"{output_path}{metric_type}-{split}set-metric-{suffix}.json", "w") as f:
        json.dump(metric, f)



"""
python ./eval/eval_code_gen.py \
    --model_type "qwen/Qwen2.5-Coder-1.5B" \
    --metric "CodeContest" \
    --split "test" \
    --suffix 0.3
"""
if __name__ == "__main__":
    argp = argparse.ArgumentParser()
    argp.add_argument("--model_type", default="", type=str)
    argp.add_argument("--metric", default="code_contest", type=str)
    argp.add_argument("--split", default="test", type=str)
    argp.add_argument("--suffix", default="0.3", type=str)
    args = argp.parse_args()

    input_path = f"./inference_results/{args.model_type}/{args.metric}_{args.split}set_inference_results_{args.suffix}.jsonl"
    output_path = f"./inference_results/{args.model_type}/"

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    code_contest_metric.BATCH_SIZE = 256
    start_time = time.time()
    main(input_path, output_path, args.split, metric_type=args.metric, suffix=args.suffix)
    end_time = time.time()
    print(f"eval done!, time elapsed -> {end_time-start_time}")
