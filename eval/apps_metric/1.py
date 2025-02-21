import itertools
import json
import multiprocessing as mp
import numpy as np
import jsonlines
import math
import time
from typing import Dict
from copy import deepcopy
from .testing_util import run_test
from tqdm import trange

DATASET = "/home/shizhongyi/paper_code/benchmarks/APPS/datasets/test.jsonl"
TIMEOUT = 10
BATCH_SZIE = 1024


def check_correctness(sample, generations, num_procs, start_idx, timeout, debug=True):
    """Check correctness of code generation with a global timeout.
    The global timeout is to catch some extreme/rare cases not handled by the timeouts
    inside `run_test`"""

    def _temp_run(
        sample,
        generation,
        result,
        p_stack,
        shift,
        idx,
        debug,
    ):
        curr_res = run_test(sample, generation, debug)
        fixed = []
        for e in curr_res:
            if isinstance(e, np.ndarray):
                e = e.item(0)
            if isinstance(e, np.bool_):
                e = bool(e)
            fixed.append(e)
        curr_res = fixed
        if not np.all(curr_res):
            if debug:
                print(f"Results were not True for all test cases")
        result[f"{idx}-{shift}"] = curr_res
        p_stack.pop()

    manager = mp.Manager()
    result = manager.dict()
    p_stack = manager.list()
    p_dict = {}
    local_result = {}

    all_gens = []
    gen_size = len(generations[0])
    for gen in generations:
        all_gens.extend(gen)

    for i in trange(len(all_gens)):
        while len(p_stack) >= num_procs:
            time.sleep(0.2)
        # print(f"init proc {i}")

        p_stack.append("$")
        gen_idx = i // gen_size
        shift = i % gen_size
        s = sample[gen_idx]
        gen = all_gens[i]
        if local_result.get(gen_idx + start_idx) is None:
            local_result[gen_idx + start_idx] = [[-2]] * gen_size

        p = mp.Process(
            target=_temp_run,
            args=(
                deepcopy(s),
                gen,
                result,
                p_stack,
                shift,
                gen_idx + start_idx,
                debug,
            ),
        )
        p.start()
        p_dict[f"{gen_idx + start_idx}-{shift}"] = p

    for k, p in p_dict.items():
        p.join(timeout + 6)
        if p.is_alive():
            p.kill()

    for k, v in result.items():
        idx, shift = k.split("-")
        local_result[int(idx)][int(shift)] = v

    return local_result


def evaluate_generations(
    generations: list,
    level: str = "all",
    split: str = "test",
    debug: bool = False,
    num_procs=8,
):
    """We take the list of code generations and try to compile them
    and the run their corresponding unit tests which are retrieved from the APPS dataset.

    Args:
        generations: list of code generations (same order as samples in APPS dataset)
        level: difficulty level used in the generation, can be "all", "introductory", "interview" or "competition"

    Returns:
        results: dictionary of results, key is the problem index, value is a list of results for each generation
        [-2] = compile error, [-1] = runtime error [False] = failed test case [True] = passed test case
    """

    # generations are code generations in the same order of the dataset
    with jsonlines.open(DATASET, mode="r") as reader:
        apps_eval = [o for o in reader]

    batch_num = math.ceil(len(apps_eval) / BATCH_SZIE)
    results = {}
    for index in range(batch_num):
        curr_batch = generations[index * BATCH_SZIE : (index + 1) * BATCH_SZIE]
        curr_samples = apps_eval[index * BATCH_SZIE : (index + 1) * BATCH_SZIE]
        try:
            curr_res = check_correctness(
                curr_samples,
                curr_batch,
                num_procs,
                index * BATCH_SZIE,
                timeout=TIMEOUT,
                debug=debug,
            )
        except Exception as e:
            if debug:
                print(f"Compilation failed, test framework exception = {repr(e)}{e}\n")
            raise e
        results.update(curr_res)
    return results


def estimate_pass_at_k(num_samples, num_correct, k):
    """Estimates pass@k of each problem and returns them in an array."""

    def estimator(n: int, c: int, k: int) -> float:
        """Calculates 1 - comb(n - c, k) / comb(n, k)."""
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    if isinstance(num_samples, int):
        num_samples_it = itertools.repeat(num_samples, len(num_correct))
    else:
        assert len(num_samples) == len(num_correct)
        num_samples_it = iter(num_samples)

    return np.array(
        [estimator(int(n), int(c), k) for n, c in zip(num_samples_it, num_correct)]
    )


def get_results(
    results: Dict[int, list], count_errors: bool = False, k_list: list = [1, 10, 100]
):
    """
    Given the results evaluated against the testcases we output some statistics.
    For single generations:
    >>> example_results = {0: [[-2]], 1: [[False,False]], 2: [[True,True]], 3: [[False,True,False,True]], 4: [[-1,-1]]}
    >>> get_results(example_results, count_errors=True)
    Computing accuracy metrics...
    number of compile errors = 1 avg = 0.2
    number of runtime errors = 1 avg = 0.2
    number of problems evaluated = 5
    Average Accuracy : 0.3
    Strict Accuracy : 0.2
    {'avg_accuracy': 0.3, 'strict_accuracy': 0.2, 'pass_at_k': None}

    For multiple generations:
    >>> example_results = {0: [[-2], [True, True, True]], 1: [[-1,-1, -1], [True, False, True]]}
    >>> get_results(example_results, k_list=[1, 2])
    Computing pass@k metric for multiple generations...
    {'pass@1': 0.25, 'pass@2': 0.5}
    {'avg_accuracy': None, 'strict_accuracy': None, 'pass_at_k': {'pass@1': 0.25, 'pass@2': 0.5}}
    """

    metrics = {"avg_accuracy": None, "strict_accuracy": None, "pass_at_k": None}

    if len(results[0]) == 1:
        # for single generations we compute average accuracy and stric accuracy: original APPS metrics
        print("Computing accuracy metrics...")
        res = []
        per_prob_res = []
        all_correct = []
        for index in results:
            problem_results = np.asarray(results[index])
            res.extend(problem_results)
            per_prob_res.append(np.mean(problem_results > 0))
            all_correct.append(np.all(problem_results > 0))
        # we count campilation and runtime errors once per pronlem
        compile_errors = len([e for e in res if -2 in e])
        runtime_errors = len([e for e in res if -1 in e])
        total_testcases = len(res)
        if count_errors:
            print(
                f"number of compile errors = {compile_errors} avg = {compile_errors / total_testcases}"
            )
            print(
                f"number of runtime errors = {runtime_errors} avg = {runtime_errors / total_testcases}"
            )
            print(f"number of problems evaluated = {total_testcases}")

        print(f"Average Accuracy : {np.mean(per_prob_res)}")
        print(f"Strict Accuracy : {np.mean(all_correct)}")
        metrics["avg_accuracy"] = np.mean(per_prob_res)
        metrics["strict_accuracy"] = np.mean(all_correct)

    else:
        # for multiple generations we use pass@k metric used in the HumanEval benchmark
        # we use strict accuracy, a generation is valid if it has to pass all the tests
        print("Computing pass@k metric for multiple generations...")
        # total is list with nb generations per task (task=index)
        # correct is number of generations that passed all tests per task
        total = []
        correct = []
        for index in results:
            all_correct = []
            for generation in results[index]:
                gen = np.array(generation)
                all_correct.append(np.all(gen > 0))
            total.append(len(all_correct))
            correct.append(sum(all_correct))
        total = np.array(total)
        correct = np.array(correct)
        ks = k_list
        pass_at_k = {
            f"pass@{k}": estimate_pass_at_k(total, correct, k).mean()
            for k in ks
            if (total >= k).all()
        }
        print(pass_at_k)
        metrics["pass_at_k"] = pass_at_k
    return metrics


def compute_metrics(
    generations,
    level="all",
    k_list=[1, 10, 100],
    split="test",
    count_errors=True,
    debug=False,
):
    """Return metrics for the given generations.
    Args:
        generations: list of code generations for each problem (each generation is a list of generations)
        k_list: list of k values to compute pass@k when using multiple generations
        count_errors: whether to count compilation and runtime errors when using single generations
        level: difficulty level in APPS dataset that was used for the given generations (from: "all", "introductory", "interview", "competition")
    Returns:
        metrics: dict of metrics

    Examples:

    >>> import json
    >>> # lists of solutions to the two first APPS problems (note not all solutions pass all tests)
    >>> solution_sample1 = json.load(open("test_examples/solutions_problem_1.json", "r"))
    >>> solution_sample2 = json.load(open("test_examples/solutions_problem_2.json", "r"))
    >>> single_solutions = [solution_sample1[:1], solution_sample2[:1]]
    >>> compute_metrics(single_solutions, level="all")
    Computing accuracy metrics...
    number of compile errors = 0 avg = 0.0
    number of runtime errors = 0 avg = 0.0
    number of problems evaluated = 2
    Average Accuracy : 1.0
    Strict Accuracy : 1.0
    {'avg_accuracy': 1.0, 'strict_accuracy': 1.0, 'pass_at_k': None}
    >>> multiple_solutions = [solution_sample1[:3], solution_sample2[:3]]
    >>> compute_metrics(multiple_solutions, level="all", k_list=[1, 2, 3])
    Computing pass@k metric for multiple generations...
    {'pass@1': 1.0, 'pass@2': 1.0, 'pass@3': 1.0}
    {'avg_accuracy': None, 'strict_accuracy': None, 'pass_at_k': {'pass@1': 1.0, 'pass@2': 1.0, 'pass@3': 1.0}}
    """
    results = evaluate_generations(generations, split=split, level=level, debug=debug)
    metrics = get_results(results, count_errors=count_errors, k_list=k_list)
    return metrics, results


# import doctest
# doctest.testmod()

if __name__ == "__main__":
    with open(
        "/home/shizhongyi/paper_code/inference_results/vanilla_CodeQwen/APPS-testset-stat.json",
        "r",
    ) as f:
        results = json.load(f)
    results = {int(k): v for k, v in results.items()}
    metric = get_results(results, count_errors=True, k_list=[1, 5])
    print(metric)
