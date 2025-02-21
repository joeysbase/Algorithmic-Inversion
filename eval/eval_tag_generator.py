import argparse
import json
import jsonlines
from collections import defaultdict

"""
python ./eval/eval_tag_generator.py --threshold 0.97
"""
if __name__ == "__main__":
    argp = argparse.ArgumentParser()
    argp.add_argument("--threshold", default=0.97, type=float)
    args = argp.parse_args()

    with jsonlines.open(
        f"./inference_results/tag-generator/tag-generator-{args.threshold}.jsonl",
        mode="r",
    ) as reader:
        data = [o for o in reader]
    fine = defaultdict(list)
    result = defaultdict(float)
    for o in data:
        for t in o["label"]:
            fine[t].append(o)
    for o in data:
        if len(o["prediction"]) == 0:
            result["empty_rate"] += 1
        for t in o["prediction"]:
            if t not in o["label"]:
                result["wrong_tags"] += 1
                break
        else:
            result["overall"] += 1
    result["overall"] /= len(data)
    result["empty_rate"] /= len(data)
    result["wrong_tags"] /= len(data)

    for k, v in fine.items():
        for o in v:
            if k in o["prediction"]:
                result[k] += 1
    for k, v in fine.items():
        result[k] /= len(v)
    for k,v in result.items():
        result[k] = float(str(v)[:4])
    with open(
        f"./inference_results/tag-generator/accuracy-{args.threshold}.json",
        mode="w",
    ) as f:
        json.dump(result, f)
    print(result)
    # for k, v in result.items():
    #     print(f"{k} -> {v}")
