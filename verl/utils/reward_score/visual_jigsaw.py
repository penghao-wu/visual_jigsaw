import re
from mathruler.grader import extract_boxed_content
from latex2sympy2_extended import NormalizationConfig
import math_verify

def levenshtein_distance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2 + 1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]

def anls(
    references,
    pred,
    thresh_hold=0.5,
):
    """https://github.com/QwenLM/Qwen-VL/blob/master/eval_mm/infographicsvqa_eval.py"""
    values = []
    for answer in references:
        # preprocess both the answers - gt and prediction
        gt_answer = " ".join(answer.strip().lower().split())
        det_answer = " ".join(pred.strip().lower().split())

        dist = levenshtein_distance(gt_answer, det_answer)
        length = max(len(answer.upper()), len(pred.upper()))
        values.append(0.0 if length == 0 else float(dist) / float(length))

    question_result = 1 - min(values)

    if question_result < thresh_hold:
        question_result = 0
    return question_result

def relaxed_correctness(prediction, target, max_relative_change: float = 0.05) -> bool:
    """Calculates relaxed correctness.

    The correctness tolerates certain error ratio defined by max_relative_change.
    See https://arxiv.org/pdf/2203.10244.pdf, end of section 5.1:
    “Following Methani et al. (2020), we use a relaxed accuracy measure for the
    numeric answers to allow a minor inaccuracy that may result from the automatic
    data extraction process. We consider an answer to be correct if it is within
    5% of the gold answer. For non-numeric answers, we still need an exact match
    to consider an answer to be correct.”

    This funcion is taken from https://github.com/QwenLM/Qwen-VL/blob/34b4c0ee7b07726371b960911f249fe61b362ca3/eval_mm/evaluate_vqa.py#L113
    Args:
      target: List of target string.
      prediction: List of predicted string.
      max_relative_change: Maximum relative change.

    Returns:
      Whether the prediction was correct given the specified tolerance.
    """

    def _to_float(text: str):
        try:
            if text.endswith("%"):
                # Convert percentages to floats.
                return float(text.rstrip("%")) / 100.0
            else:
                return float(text)
        except ValueError:
            return None

    prediction_float = _to_float(prediction)
    target_float = _to_float(target)
    if prediction_float is not None and target_float:
        relative_change = abs(prediction_float - target_float) / abs(target_float)
        return relative_change <= max_relative_change
    else:
        return prediction.lower() == target.lower()

def check_format(predict: str) -> bool:
    pattern = re.compile(r"<think>.*</think>.*\\boxed\{.*\}.*", re.DOTALL)
    format_match = re.fullmatch(pattern, predict)
    return True if format_match else False

def has_excessive_repetition(
    text: str,
    min_len: int = 8,
    max_run: int = 1
) -> int:
    """
    Detect consecutive repetition of long lines.

    Returns
    -------
    int
        1  → at least one long line appears in a run longer than `max_run`
        0  → no run exceeds `max_run`
    """
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    prev_key, run_len = None, 0

    for line in lines:
        is_long = len(line.split()) >= min_len
        if not is_long:
            prev_key, run_len = None, 0
            continue

        key = line.lower()
        if key == prev_key:
            run_len += 1
        else:
            prev_key, run_len = key, 1

        if run_len > max_run:           # repetition exceeds threshold
            return 1

    return 0

def cal_math_reward(predict_str: str, solution: str) -> float:
    """@yema | Reward function that checks if the completion is the same as the ground truth.
    referenced this part of the code from 
    https://github.com/huggingface/open-r1/blob/main/src/open_r1/rewards.py"""
    predict_str, solution = predict_str.lower(), solution.lower()
    gold_parsed = math_verify.parse(
        solution,
        extraction_mode="first_match",
    )
    if len(gold_parsed) != 0:
        # We require the answer to be provided in correct latex (no malformed operators)
        answer_parsed = math_verify.parse(
            predict_str,
            extraction_config=[
                math_verify.LatexExtractionConfig(
                    normalization_config=NormalizationConfig(
                        nits=False,
                        malformed_operators=False,
                        basic_latex=True,
                        boxed="all",
                        units=True,
                    ),
                    # Ensures that boxed is tried first
                    boxed_match_priority=0,
                    try_extract_without_anchor=False,
                )
            ],
            extraction_mode="first_match",
        )
        # Compute binary rewards if verifiable, `None` otherwise to skip this example
        try:
            reward = float(math_verify.verify(gold_parsed, answer_parsed))
        except Exception as e:
            print(f"verify failed: {e}, answer: {answer_parsed}, gold: {gold_parsed}")
            #TODO: @yema the return should be none, and we need mask these "None" samples
            reward = 0.
    else:
        # If the gold solution is not parseable, we assign `None` to skip this example
        #TODO: @yema the return should be none, and we need mask these "None" samples
        reward = 0.
        print("Failed to parse gold solution: ", solution)

    return reward


def compute_score_math(solution_str, ground_truth, format_score=0.2, score=1.0, nothink=False):
    if nothink:
        answer_text = solution_str.strip()
        format_score = 0
    else:
        correct_format = check_format(solution_str)
        if not correct_format:
            return {'score': 0, 'acc_reward': 0, 'format_reward': 0, 'acc': 0}
        answer_text = extract_boxed_content(solution_str)
    try:
        score = cal_math_reward(answer_text, ground_truth)
    except:
        score = 0
    return {'score': format_score+score, 'acc_reward': score, 'format_reward': format_score, 'acc': score}
    
def compute_score_exact_match(solution_str, ground_truth, format_score=0.2, score=1.0, nothink=False):
    if nothink:
        answer_text = solution_str.strip()
        format_score = 0
    else:
        correct_format = check_format(solution_str)
        if not correct_format:
            return {'score': 0, 'acc_reward': 0, 'format_reward': 0, 'acc': 0}
        answer_text = extract_boxed_content(solution_str)
    if answer_text.lower() != ground_truth.lower():
        score = 0
    return {'score': format_score+score, 'acc_reward': score, 'format_reward': format_score, 'acc': score}

def compute_score_anls(solution_str, ground_truth, format_score=0.2, score=1.0, nothink=False):
    if nothink:
        answer_text = solution_str.strip()
        format_score = 0
    else:
        correct_format = check_format(solution_str)
        if not correct_format:
            return {'score': 0, 'acc_reward': 0, 'format_reward': 0, 'acc': 0}
        answer_text = extract_boxed_content(solution_str)
    try:
        score = anls([ground_truth], answer_text)
    except:
        score = 0
    return {'score': format_score+score, 'acc_reward': score, 'format_reward': format_score, 'acc': score}

def compute_score_relaxed_acc(solution_str, ground_truth, format_score=0.2, score=1.0, nothink=False):
    if nothink:
        answer_text = solution_str.strip()
        format_score = 0
    else:
        correct_format = check_format(solution_str)
        if not correct_format:
            return {'score': 0, 'acc_reward': 0, 'format_reward': 0, 'acc': 0}
        answer_text = extract_boxed_content(solution_str)
    try:
        score = int(relaxed_correctness(answer_text, ground_truth))
    except:
        score = 0
    return {'score': format_score+score, 'acc_reward': score, 'format_reward': format_score, 'acc': score}


def compute_score(solution_str, ground_truth, format_score=0.2, score=1.0, nothink=False):
    rep_penalty = 0
    if has_excessive_repetition(solution_str, min_len=7, max_run=3):
        rep_penalty=-0.5
    if nothink:
        answer_indices = solution_str.split(',')
        format_score = 0
    else:
        correct_format = check_format(solution_str)
        if not correct_format:
            return {'score': 0, 'acc_reward': 0, 'format_reward': 0, 'acc': 0}
        # answer_text = solution_str.split("<answer>")[-1].split("</answer>")[0].strip()
        answer_text = extract_boxed_content(solution_str)
        answer_indices = answer_text.split(',')
    try:
        answer_indices = [int(answer_index.strip()) for answer_index in answer_indices]
    except:
        answer_indices = []
    if answer_indices == ground_truth:
        return {'score': format_score+score+rep_penalty, 'acc_reward': score, 'format_reward': format_score, 'acc': 1}
    else:
        acc_reward = 0
        # partial correct
        if len(set(answer_indices)) == len(ground_truth):
            partial_correct_num = 0
            for i in range(len(ground_truth)):
                if answer_indices[i] == ground_truth[i]:
                    partial_correct_num += 1
            if len(ground_truth) > 4:
                acc_reward = partial_correct_num/len(ground_truth) * 0.2
        return {'score': format_score+acc_reward+rep_penalty, 'acc_reward': acc_reward, 'format_reward': format_score, 'acc': 0}
    
def compute_score_singleint(solution_str, ground_truth, format_score=0.2, score=1.0):
    correct_format = check_format(solution_str)
    if not correct_format:
        return {'score': 0, 'acc_reward': 0, 'format_reward': 0, 'acc': 0}
    answer_text = solution_str.split("<answer>")[-1].split("</answer>")[0].strip()
    if answer_text.strip() == str(ground_truth):
        return {'score': format_score+score, 'acc_reward': score, 'format_reward': format_score, 'acc': 1}
    else:
        return {'score': format_score, 'acc_reward': 0, 'format_reward': format_score, 'acc': 0}