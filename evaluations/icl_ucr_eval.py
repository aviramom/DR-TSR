import json
import os
import argparse
import re  # <-- Add this import for pattern matching
from tqdm import tqdm
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, recall_score

def _extract_predicted_label(response: str, options: list) -> str:
    """Return the option that best matches the response, or INVALID_PREDICTION.

    For thinking-mode models the answer section can contain long reasoning text before
    the final label.  We therefore collect the LAST position at which each option is
    found under any pattern and return the option whose last match is latest in the
    string — that is the model's final answer, not a label it mentioned in passing.
    """
    if response == "":
        return "INVALID_PREDICTION"

    # Exact match (short, clean response) — fast path
    for opt in options:
        if response == opt:
            return opt

    patterns = [
        lambda o: [(m.start(), o) for m in re.finditer(
            r'The class is\s+' + re.escape(o) + r'(?!\w)', response, re.IGNORECASE)],
        lambda o: [(m.start(), o) for m in re.finditer(
            r'The class is\s+<' + re.escape(o) + r'>', response, re.IGNORECASE)],
        lambda o: [(m.start(), o) for m in re.finditer(
            r'Predicted\s*Label\s*:\s*["\'<\[]?\s*' + re.escape(o) + r'(?!\d)', response, re.IGNORECASE)],
        lambda o: [(m.start(), o) for m in re.finditer(
            r'Predicted\s*:\s*["\'<\[]?\s*' + re.escape(o) + r'(?!\d)', response, re.IGNORECASE)],
        lambda o: [(m.start(), o) for m in re.finditer(
            r'(?<!\w)label\s*:\s*["\'<\[]?\s*' + re.escape(o) + r'(?!\d)', response, re.IGNORECASE)],
        lambda o: [(m.start(), o) for m in re.finditer(
            r'(?:correct\s+)?label\s+is\s+["\'<\[]?\s*' + re.escape(o) + r'(?!\d)', response, re.IGNORECASE)],
        # Official TSE MCQ format: "A) Linear" — match option letter followed by closing paren
        lambda o: [(m.start(), o) for m in re.finditer(
            r'(?<!\w)' + re.escape(o) + r'\)', response)],
    ]

    all_hits = []  # (position, option)
    for pat in patterns:
        for opt in options:
            all_hits.extend(pat(opt))

    if all_hits:
        # Return the option whose last match appears latest — the final answer
        _, best_opt = max(all_hits, key=lambda x: x[0])
        return best_opt

    # Fallback: look for a bare option token in the last 300 chars
    tail = response[-300:]
    for opt in options:
        if re.search(r'(?<!\w)' + re.escape(opt) + r'(?!\w)', tail):
            return opt

    return "INVALID_PREDICTION"


def _parse_options(prompt: str) -> list:
    """Extract label options from 'Return ONLY the label as one of: [a, b, ...]'."""
    match = re.search(r'Return ONLY the label as one of:\s*\[([^\]]+)\]', prompt)
    if not match:
        return []
    return [opt.strip() for opt in match.group(1).split(',')]


def run_evaluation_icl_ucr(model, dataloader=None, args=None):

    if "ucr" not in args.task_id.lower() and "tse" not in args.task_id.lower():
        raise ValueError("task_id must contain 'ucr' or 'tse'")
    
    accuracy_scores = []
    gold_answers = []
    predicted_answers = []
    input_ts = []
    generated_texts = []
    questions = []

    for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Evaluating {args.task_id}")):
        batch_prompts = batch["input_text"]
        gen_out = model.generate(batch)

        for i in range(len(batch_prompts)):
            answer = str(batch["output_text"][i]).strip()
            response = str(gen_out[i]).strip()

            # Prefer the batch's own options list (works for tse_official which has no "Return ONLY" line).
            batch_options = batch.get("options")
            if batch_options and batch_options[i]:
                options = list(batch_options[i])
            else:
                options = _parse_options(batch_prompts[i])
            predicted = _extract_predicted_label(response, options)

            accuracy_scores.append(1 if predicted == answer else 0)
            predicted_answers.append(predicted)
            gold_answers.append(answer)
            questions.append(batch_prompts[i])
            generated_texts.append(response)
            input_ts.append(batch["input_ts"][i])

    results = {}

    results['accuracy_scores'] = accuracy_scores
    results['num_of_classes'] = len(set(gold_answers))
    results['total_test_size'] = len(accuracy_scores)
    results['num_test_samples'] = len(dataloader.dataset) if hasattr(dataloader, 'dataset') else len(accuracy_scores)
    results['balanced_accuracy'] = balanced_accuracy_score(gold_answers, predicted_answers)
    results['f1_macro'] = f1_score(gold_answers, predicted_answers, average='macro')
    results['f1_weighted'] = f1_score(gold_answers, predicted_answers, average='weighted')
    results['precision_macro'] = precision_score(gold_answers, predicted_answers, average='macro', zero_division=0)
    
    # Recall: How many of the actual positives did we find?
    # Note: Macro Recall is mathematically identical to Balanced Accuracy in multiclass
    results['recall_macro'] = recall_score(gold_answers, predicted_answers, average='macro', zero_division=0)

    # Weighted versions (scaled by class frequency)
    results['precision_weighted'] = precision_score(gold_answers, predicted_answers, average='weighted', zero_division=0)
    results['recall_weighted'] = recall_score(gold_answers, predicted_answers, average='weighted', zero_division=0)



    input_output = {
        "questions": questions,
        "generated_texts": generated_texts,
        "predicted_answers": predicted_answers,
        "gold_answers": gold_answers,
        "input_ts": input_ts,
    }

    return results, input_output