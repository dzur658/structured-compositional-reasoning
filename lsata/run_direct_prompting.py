
from pathlib import Path
import json
import math
import os
import random
import re
import shutil
import itertools

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from sklearn.metrics import precision_recall_fscore_support
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed,
)

from datasets import load_dataset

HF_DATASET = "ojayy/logical-sata"
SPLIT = "validation"

LABELS = ["A", "B", "C", "D"]
LABEL_MAP = {0: "A", 1: "B", 2: "C", 3: "D"}


def load_lsata(split: str = SPLIT) -> pd.DataFrame:
    ds = load_dataset(HF_DATASET, split=split)
    rows = []

    for item in ds:
        choices = item["choices"]
        label = item["label"]
        label_letter = LABEL_MAP[int(label)] if str(label).isdigit() else str(label).upper()

        rows.append({
            "question": item["question"],
            "paragraph": item["paragraph"],
            "A": choices[0],
            "B": choices[1],
            "C": choices[2],
            "D": choices[3],
            "correct_label": label_letter,
            "correct_answer_text": choices[LABELS.index(label_letter)],
            "qa_type": item["qa_type"],
        })

    dataframe = pd.DataFrame(rows)
    dataframe["qa_type"] = dataframe["qa_type"].replace({
        "MIXED": "Mixed",
        "NNOR": "NEITHER",
        "NEITHER/NOR": "NEITHER",
    })

    print(f"Loaded {len(dataframe)} LOGICAL-SATA {split} examples from {HF_DATASET}")
    print(dataframe["qa_type"].value_counts(dropna=False))
    return dataframe


qa_df = load_lsata()
qa_df.head()


import gc

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"

NUM_RUNS = 5
RUN_SEEDS = [0, 1, 2, 3, 4]

DO_SAMPLE = True
TEMPERATURE = 0.7

BATCH_SIZE = 8
MAX_INPUT_TOKENS = 4096

REPAIR_UNPARSED = True
RESUME_COMPLETED_RUNS = True

gc.collect()
torch.cuda.empty_cache()

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map={"": 0},
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    attn_implementation="sdpa",
)

model.eval()
MODEL_DEVICE = next(model.parameters()).device

print(f"Loaded {MODEL_NAME} in bf16 on {MODEL_DEVICE}.")
print(
    f"Runs={NUM_RUNS}, seeds={RUN_SEEDS}, batch_size={BATCH_SIZE}, "
    f"temperature={TEMPERATURE}, sampling={DO_SAMPLE}"
)

LABELS = ["A", "B", "C", "D"]
INVALID_LABEL = "__INVALID__"

def normalize_label(value):
    if pd.isna(value):
        return None
    value = str(value).strip().upper()
    return value if value in LABELS else None


def parse_mcq_answer(text):
    """
    Parse an A/B/C/D answer conservatively, prioritizing explicit final-answer
    markers and only using line-level fallbacks near the end of the generation.

    Returns:
        (label, parse_method)
    """
    if text is None or not str(text).strip():
        return None, "empty"

    text = str(text).strip()

    patterns = [
        (
            "final_answer",
            r"FINAL\s*(?:ANSWER|CHOICE)\s*(?:IS\s*)?[:=\-]?\s*"
            r"[\(\[\{`*_]*([ABCD])(?=[\)\]\}`*_.:,\s]|$)",
        ),
        (
            "boxed",
            r"\\boxed\s*\{\s*([ABCD])\s*\}",
        ),
        (
            "explicit_answer",
            r"(?:THE\s+)?(?:CORRECT\s+)?(?:ANSWER|CHOICE)\s*(?:IS\s*)?"
            r"[:=\-]?\s*[\(\[\{`*_]*([ABCD])(?=[\)\]\}`*_.:,\s]|$)",
        ),
        (
            "option_phrase",
            r"(?:SELECT|CHOOSE|OPTION)\s*[:=\-]?\s*"
            r"[\(\[\{`*_]*([ABCD])(?=[\)\]\}`*_.:,\s]|$)",
        ),
    ]

    for method, pattern in patterns:
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        if matches:
            return matches[-1].upper(), method

    # Strong fallback for answer-only generations such as "A", "(B)", or "C".
    cleaned_whole = re.sub(r"[\s\(\)\[\]\{\}`*_.:,\-]+", "", text).upper()
    if cleaned_whole in LABELS:
        return cleaned_whole, "answer_only"

    # Examine only the final few non-empty lines to avoid extracting option
    # letters mentioned inside chain-of-thought reasoning.
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in reversed(lines[-4:]):
        cleaned_line = re.sub(
            r"^(?:FINAL\s*)?(?:ANSWER|CHOICE|OPTION)?\s*(?:IS\s*)?[:=\-]?\s*",
            "",
            line,
            flags=re.IGNORECASE,
        )
        cleaned_line = re.sub(r"[\s\(\)\[\]\{\}`*_.:,\-]+", "", cleaned_line).upper()
        if cleaned_line in LABELS:
            return cleaned_line, "last_line"

    return None, "unparsed"


def build_repair_prompt(original_prompt, raw_generation):
    return f"""
The response below was produced for a multiple-choice question, but its final
choice was not machine-parseable.

Original question and instructions:
{original_prompt}

Previous response:
{raw_generation}

Recover the intended final choice. Return ONLY one capital letter: A, B, C, or D.
""".strip()


def compute_metrics(results_df):
    """
    Strict metrics over every benchmark item.

    Unparsed predictions count as incorrect for accuracy and macro/micro metrics.
    answered_accuracy is also reported as a parser diagnostic.
    """
    if len(results_df) == 0:
        raise ValueError("Cannot evaluate an empty results DataFrame.")

    y_true = results_df["correct_label"].map(normalize_label)
    if y_true.isna().any():
        bad = results_df.loc[y_true.isna(), "correct_label"].head().tolist()
        raise ValueError(f"Invalid correct labels found, e.g. {bad}")

    y_pred_raw = results_df["predicted_label"].map(normalize_label)
    answered_mask = y_pred_raw.notna()
    y_pred_eval = y_pred_raw.fillna(INVALID_LABEL)

    total = len(results_df)
    answered = int(answered_mask.sum())
    correct = int((y_true == y_pred_eval).sum())

    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred_eval,
        labels=LABELS,
        average="macro",
        zero_division=0,
    )
    micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred_eval,
        labels=LABELS,
        average="micro",
        zero_division=0,
    )

    answered_accuracy = (
        float((y_true[answered_mask] == y_pred_raw[answered_mask]).mean())
        if answered
        else np.nan
    )

    initial_methods = results_df["initial_parse_method"].fillna("empty").astype(str)
    initial_parsed = (
        initial_methods.ne("unparsed")
        & initial_methods.ne("empty")
    )
    repaired = results_df["parse_method"].fillna("").astype(str).str.startswith("repair:")
    initial_failures = int((~initial_parsed).sum())
    repair_recovery_rate = (
        float(repaired.sum() / initial_failures)
        if initial_failures
        else 0.0
    )

    return {
        "n": total,
        "answered": answered,
        "unanswered": total - answered,
        "answer_rate": answered / total,
        "initial_parse_rate": float(initial_parsed.mean()),
        "repair_recovery_rate": repair_recovery_rate,
        "accuracy": correct / total,
        "answered_accuracy": answered_accuracy,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f1,
        "micro_precision": micro_p,
        "micro_recall": micro_r,
        "micro_f1": micro_f1,
    }


def evaluate_by_operator(results_df, setting, run_id, seed):
    rows = []
    operators = ["AND", "OR", "NEITHER", "Mixed", "ALL"]

    for operator in operators:
        subset = (
            results_df
            if operator == "ALL"
            else results_df[results_df["qa_type"] == operator]
        )
        if len(subset) == 0:
            continue

        row = {
            "setting": setting,
            "run": run_id,
            "seed": seed,
            "operator": operator,
        }
        row.update(compute_metrics(subset))
        rows.append(row)

    return pd.DataFrame(rows)


def aggregate_runs(metrics_per_run):
    metric_columns = [
        "answer_rate",
        "initial_parse_rate",
        "repair_recovery_rate",
        "accuracy",
        "answered_accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "micro_precision",
        "micro_recall",
        "micro_f1",
    ]

    grouped = (
        metrics_per_run
        .groupby(["setting", "operator"], sort=False)[metric_columns]
        .agg(["mean", "std"])
        .reset_index()
    )
    grouped.columns = [
        "_".join([str(part) for part in col if part]).rstrip("_")
        if isinstance(col, tuple)
        else col
        for col in grouped.columns
    ]

    fixed = (
        metrics_per_run
        .groupby(["setting", "operator"], sort=False)
        .agg(n=("n", "first"), runs=("run", "nunique"))
        .reset_index()
    )

    return fixed.merge(grouped, on=["setting", "operator"], how="left")


def print_mean_std_summary(summary_df):
    print("\nFIVE-RUN MEAN ± SAMPLE STANDARD DEVIATION")
    print("=" * 100)
    print(
        f"{'Setting':<14} {'Operator':<10} {'N':>6} "
        f"{'Accuracy':>19} {'Macro-F1':>19} {'Answer rate':>19}"
    )
    print("-" * 100)

    for _, row in summary_df.iterrows():
        accuracy = f"{row['accuracy_mean']:.4f} ± {row['accuracy_std']:.4f}"
        macro_f1 = f"{row['macro_f1_mean']:.4f} ± {row['macro_f1_std']:.4f}"
        answer_rate = f"{row['answer_rate_mean']:.4f} ± {row['answer_rate_std']:.4f}"
        print(
            f"{row['setting']:<14} {row['operator']:<10} {int(row['n']):>6} "
            f"{accuracy:>19} {macro_f1:>19} {answer_rate:>19}"
        )


def prepare_prompt_table(qa_dataframe, prompt_builder):
    prepared = qa_dataframe.copy().reset_index(drop=True)
    prepared["_row_id"] = np.arange(len(prepared))
    prepared["_prompt"] = [
        prompt_builder(row) for _, row in prepared.iterrows()
    ]

    prepared["_prompt_tokens"] = [
        len(tokenizer.encode(prompt, add_special_tokens=False))
        for prompt in tqdm(prepared["_prompt"], desc="Measuring prompt lengths")
    ]
    return prepared.sort_values("_prompt_tokens").reset_index(drop=True)


def _generate_single_batch(
    prompts,
    *,
    max_new_tokens,
    do_sample,
    temperature,
):
    encoded = None
    generated = None

    try:
        encoded = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_INPUT_TOKENS,
        )

        encoded = {
            key: value.to(MODEL_DEVICE)
            for key, value in encoded.items()
        }

        generation_kwargs = {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": tokenizer.eos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "use_cache": True,
        }

        if do_sample:
            generation_kwargs["temperature"] = temperature

        input_length = encoded["input_ids"].shape[1]

        with torch.inference_mode():
            generated = model.generate(**generation_kwargs)

        continuation = generated[:, input_length:]

        texts = tokenizer.batch_decode(
            continuation,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        return texts

    finally:
        if generated is not None:
            del generated

        if encoded is not None:
            del encoded


def _generate_texts_with_backoff(
      prompts,
      *,
      max_new_tokens,
      do_sample,
      temperature,
  ):
      """
      Generate a batch and split it after an OOM.

      The recursive calls happen only after leaving the exception block, preventing
      failed CUDA tracebacks from retaining tensors while smaller batches run.
      """
      try:
          return _generate_single_batch(
              prompts,
              max_new_tokens=max_new_tokens,
              do_sample=do_sample,
              temperature=temperature,
          )

      except torch.cuda.OutOfMemoryError:
          # Leave the failed generation frame before attempting smaller batches.
          pass

      gc.collect()
      torch.cuda.empty_cache()

      if len(prompts) == 1:
          prompt_tokens = len(
              tokenizer.encode(
                  prompts[0],
                  add_special_tokens=False,
              )
          )

          raise RuntimeError(
              "CUDA OOM even for one prompt. "
              f"Prompt length before truncation: {prompt_tokens} tokens. "
              "Check nvidia-smi for another process or duplicate model instance."
          )

      midpoint = len(prompts) // 2

      print(
          f"CUDA OOM for batch of {len(prompts)}; "
          f"retrying as {midpoint} + {len(prompts) - midpoint}."
      )

      left = _generate_texts_with_backoff(
          prompts[:midpoint],
          max_new_tokens=max_new_tokens,
          do_sample=do_sample,
          temperature=temperature,
      )

      right = _generate_texts_with_backoff(
          prompts[midpoint:],
          max_new_tokens=max_new_tokens,
          do_sample=do_sample,
          temperature=temperature,
      )

      return left + right


def run_batched_inference(
    prepared,
    *,
    run_id,
    seed,
    setting,
    max_new_tokens,
    batch_size=BATCH_SIZE,
    repair_unparsed=True,
):
    """
    Batched stochastic inference for one repeat.

    A different deterministic seed is set for each length-bucket batch, so a
    resumed/repeated run remains stable even if an earlier batch had an OOM
    split.
    """
    records = []
    total_batches = math.ceil(len(prepared) / batch_size)

    for batch_id, start in enumerate(
        tqdm(
            range(0, len(prepared), batch_size),
            total=total_batches,
            desc=f"{setting} | run {run_id}",
        )
    ):
        batch = prepared.iloc[start:start + batch_size].copy()
        prompts = batch["_prompt"].tolist()

        batch_seed = seed * 100_000 + batch_id
        set_seed(batch_seed)
        torch.cuda.manual_seed_all(batch_seed)

        raw_texts = _generate_texts_with_backoff(
            prompts,
            max_new_tokens=max_new_tokens,
            do_sample=DO_SAMPLE,
            temperature=TEMPERATURE,
        )

        initial = [parse_mcq_answer(text) for text in raw_texts]
        predictions = [item[0] for item in initial]
        initial_methods = [item[1] for item in initial]
        parse_methods = initial_methods.copy()
        repair_texts = [None] * len(batch)

        failed_positions = [
            i for i, prediction in enumerate(predictions) if prediction is None
        ]

        if repair_unparsed and failed_positions:
            repair_prompts = [
                build_repair_prompt(prompts[i], raw_texts[i])
                for i in failed_positions
            ]

            # Repair is deterministic extraction, not another stochastic repeat.
            repair_outputs = _generate_texts_with_backoff(
                repair_prompts,
                max_new_tokens=8,
                do_sample=False,
                temperature=TEMPERATURE,
            )

            for position, repair_text in zip(failed_positions, repair_outputs):
                repaired_label, repaired_method = parse_mcq_answer(repair_text)
                repair_texts[position] = repair_text
                if repaired_label is not None:
                    predictions[position] = repaired_label
                    parse_methods[position] = f"repair:{repaired_method}"

        keep_columns = [
            col for col in batch.columns
            if not col.startswith("_")
        ]
        batch_output = batch[keep_columns].copy()
        batch_output["_row_id"] = batch["_row_id"].to_numpy()
        batch_output["run"] = run_id
        batch_output["seed"] = seed
        batch_output["setting"] = setting
        batch_output["predicted_label"] = predictions
        batch_output["raw_generation"] = raw_texts
        batch_output["raw_repair_generation"] = repair_texts
        batch_output["initial_parse_method"] = initial_methods
        batch_output["parse_method"] = parse_methods
        records.append(batch_output)

    results = (
        pd.concat(records, ignore_index=True)
        .sort_values("_row_id")
        .reset_index(drop=True)
    )

    parsed = results["predicted_label"].notna().sum()
    repaired = results["parse_method"].astype(str).str.startswith("repair:").sum()
    print(
        f"{setting} run {run_id}: parsed {parsed}/{len(results)} "
        f"({parsed / len(results):.2%}); repaired {repaired}."
    )
    return results


RAW_FEWSHOT = [
    {
        "question": "How was Pilar Gonzalez Ferreira killed?",
        "paragraph": (
            "Madrid, Spain (CNN) -- Relatives of a woman killed in a Spanish "
            "airline crash were erroneously given the remains of another victim, "
            "and then were asked by authorities to return them. A Madrid judge "
            "has opened an investigation into the error. The family Wednesday "
            "received an urn numbered 104, and were told it contained the ashes "
            "of their loved one, Pilar Gonzalez Ferreira, who died in the crash. "
            "The Spanair MD82 jet crashed last week at Madrid's airport as the "
            "plane was trying to take off, killing 154 people. The aircraft "
            "managed to rise only slightly before coming down quickly to the "
            "right of the runway, its tail section hitting the ground first. "
            "Then the out-of-control plane skidded and bounced at least three "
            "times as it careered 1,200 meters across uneven terrain and exploded."
        ),
        "correct": [
            "Spanish airline crash",
            (
                "In a plane crash during takeoff at Madrid Airport. The plane "
                "exploded after skidding for 1,200m over uneven terrain"
            ),
            "Plane crash",
        ],
        "incorrect": ["Helicopter crash", "Road crash", "Officials"],
    },
    {
        "question": "What is left behind in a flood plain after the water recedes?",
        "paragraph": (
            "A flood occurs when a river overflows its banks. This might happen "
            "because of heavy rains. In very flat regions, flood water may spread "
            "out on the surface of the land. It then slows down and drops its "
            "sediment. If a river floods often, a floodplain develops. A "
            "floodplain is an area where a thick layer of rich soil is left "
            "behind as the floodwater recedes. That is why floodplains are "
            "usually good places for growing plants. They are very flat areas "
            "and they have very rich soils. The Nile River valley is a great "
            "example of a floodplain. Each year, the Nile River rises over its "
            "banks. This floodwater carries a lot of sediment. What is left "
            "behind is a very rich soil. That is why crops can be raised in the "
            "middle of a sandy desert."
        ),
        "correct": ["Very rich soil", "Sediment", "Thick layer of rich soil"],
        "incorrect": [
            "Crops",
            "Treasure and beans are left after water recedes",
            "Flood water",
            "Natural levees",
            "Plants",
        ],
    },
    {
        "question": "Where was the original address of the Legal Aid Society headquarters?",
        "paragraph": (
            "Nearly a year after Sept. 11, the Legal Aid Society — the lawyers "
            "for New York's poor and homeless — remains, well, homeless. The "
            "nonprofit has been barred from returning to its 90 Church St. "
            "headquarters, across from the World Trade Center site, because of "
            "environmental concerns. Legal Aid's 450 displaced attorneys and "
            "staffers have spent the past 12 months spread among previously "
            "unused spaces in the nonprofit's other offices. It could be another "
            "year and a half before they return to their old desks. Their papers "
            "and documents, some 20,000 boxes worth, are stuck in a storage "
            "facility in Linden, N.J."
        ),
        "correct": [
            "90 Church St, across from the World Trade Center site",
            "90 Church Street",
        ],
        "incorrect": [
            "New York",
            "123 World Trade Center Blvd",
            "Storage facility in Linden, N.J",
        ],
    },
]


def generate_and_combinations(correct, incorrect):
    valid = [
        f"{left} AND {right}"
        for left, right in itertools.combinations(correct, 2)
    ]
    invalid = [
        f"{correct_item} AND {incorrect_item}"
        for correct_item in correct
        for incorrect_item in incorrect
    ]
    invalid += [
        f"{left} AND {right}"
        for left, right in itertools.combinations(incorrect, 2)
    ]
    return {"correct": valid, "incorrect": invalid}


def generate_or_combinations(correct, incorrect):
    valid = [
        f"{left} OR {right}"
        for left, right in itertools.combinations(correct, 2)
    ]
    valid += [
        f"{correct_item} OR {incorrect_item}"
        for correct_item in correct
        for incorrect_item in incorrect
    ]
    invalid = [
        f"{left} OR {right}"
        for left, right in itertools.combinations(incorrect, 2)
    ]
    return {"correct": valid, "incorrect": invalid}


def generate_neither_combinations(correct, incorrect):
    valid = [
        f"NEITHER {left} NOR {right}"
        for left, right in itertools.combinations(incorrect, 2)
    ]
    invalid = [
        f"NEITHER {left} NOR {right}"
        for left, right in itertools.combinations(correct, 2)
    ]
    invalid += [
        f"NEITHER {correct_item} NOR {incorrect_item}"
        for correct_item in correct
        for incorrect_item in incorrect
    ]
    return {"correct": valid, "incorrect": invalid}


def build_fewshot_mcq(example, operator, rng):
    correct = example["correct"]
    incorrect = example["incorrect"]

    if operator == "AND":
        combinations = generate_and_combinations(correct, incorrect)
    elif operator == "OR":
        combinations = generate_or_combinations(correct, incorrect)
    elif operator == "NEITHER":
        combinations = generate_neither_combinations(correct, incorrect)
    elif operator == "Mixed":
        combinations = {
            "correct": (
                generate_and_combinations(correct, incorrect)["correct"]
                + generate_or_combinations(correct, incorrect)["correct"]
                + generate_neither_combinations(correct, incorrect)["correct"]
            ),
            "incorrect": (
                generate_and_combinations(correct, incorrect)["incorrect"]
                + generate_or_combinations(correct, incorrect)["incorrect"]
                + generate_neither_combinations(correct, incorrect)["incorrect"]
            ),
        }
    else:
        raise ValueError(f"Unsupported operator: {operator}")

    correct_pair = rng.choice(combinations["correct"])
    distractors = rng.sample(combinations["incorrect"], 3)
    choices = [correct_pair] + distractors
    rng.shuffle(choices)

    return {
        "question": example["question"],
        "paragraph": example["paragraph"],
        "choices": choices,
        "label": choices.index(correct_pair),
    }


def format_fewshot_block(mcq):
    lines = [
        f"Passage: {mcq['paragraph']}",
        f"Question: {mcq['question']}",
    ]
    for option_index, choice in enumerate(mcq["choices"]):
        lines.append(f"({LABEL_MAP[option_index]}) {choice}")
    lines.append(f"Answer: {LABEL_MAP[mcq['label']]}")
    return "\n".join(lines)


fewshot_rng = random.Random(42)
FEWSHOT_POOL = {}
for operator in ["AND", "OR", "NEITHER", "Mixed"]:
    FEWSHOT_POOL[operator] = [
        format_fewshot_block(
            build_fewshot_mcq(example, operator, fewshot_rng)
        )
        for example in RAW_FEWSHOT
    ]


def build_lsata_direct_prompt(row, n_shot):
    examples = FEWSHOT_POOL[row["qa_type"]][:n_shot]
    fewshot_text = "\n\n".join(examples)
    if fewshot_text:
        fewshot_text += "\n\n"

    current_question = (
        f"Passage: {row['paragraph']}\n"
        f"Question: {row['question']}\n"
        f"(A) {row['A']}\n"
        f"(B) {row['B']}\n"
        f"(C) {row['C']}\n"
        f"(D) {row['D']}\n"
        "Answer:"
    )

    instruction = (
        "Answer the following reading-comprehension question by selecting the "
        "correct option.\n"
        "Respond with ONLY the single capital letter A, B, C, or D. Do not "
        "include explanation or punctuation.\n\n"
    )
    return instruction + fewshot_text + current_question

# Run five repeats and aggregate mean ± standard deviation

EXPERIMENT_NAME = "lsata_0to3shot_5runs"
OUTPUT_DIR = Path(EXPERIMENT_NAME)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SHOT_SETTINGS = [0, 1, 2, 3]
MAX_NEW_TOKENS = 16

RUN_CONFIG = {
    "model_name": MODEL_NAME,
    "num_runs": NUM_RUNS,
    "run_seeds": RUN_SEEDS,
    "do_sample": DO_SAMPLE,
    "temperature": TEMPERATURE,
    "batch_size": BATCH_SIZE,
    "max_input_tokens": MAX_INPUT_TOKENS,
    "max_new_tokens": MAX_NEW_TOKENS,
    "repair_unparsed": REPAIR_UNPARSED,
    "shot_settings": SHOT_SETTINGS,
}
CONFIG_PATH = OUTPUT_DIR / "run_config.json"

if CONFIG_PATH.exists():
    existing_config = json.loads(CONFIG_PATH.read_text())
    if existing_config != RUN_CONFIG:
        raise RuntimeError(
            f"{OUTPUT_DIR} contains results from a different configuration. "
            "Rename/delete that folder or restore its run_config.json settings."
        )
else:
    CONFIG_PATH.write_text(json.dumps(RUN_CONFIG, indent=2))

all_metric_frames = []

for n_shot in SHOT_SETTINGS:
    setting = f"{n_shot}shot"
    print(f"\nPreparing prompts for {setting}...")
    prepared = prepare_prompt_table(
        qa_df,
        lambda row, n=n_shot: build_lsata_direct_prompt(row, n),
    )

    for run_id, seed in enumerate(RUN_SEEDS):
        run_path = OUTPUT_DIR / f"{EXPERIMENT_NAME}_{setting}_run{run_id}.csv"

        can_resume = False
        if RESUME_COMPLETED_RUNS and run_path.exists():
            cached = pd.read_csv(run_path)
            required_cached_columns = {
                "predicted_label",
                "correct_label",
                "qa_type",
                "initial_parse_method",
                "parse_method",
            }
            can_resume = (
                len(cached) == len(qa_df)
                and required_cached_columns.issubset(cached.columns)
            )

        if can_resume:
            print(f"Loading completed {setting} run {run_id} from {run_path}")
            run_results = cached
        else:
            run_results = run_batched_inference(
                prepared,
                run_id=run_id,
                seed=seed,
                setting=setting,
                max_new_tokens=MAX_NEW_TOKENS,
                repair_unparsed=REPAIR_UNPARSED,
            )
            run_results.to_csv(run_path, index=False)
            print(f"Saved {run_path}")

        all_metric_frames.append(
            evaluate_by_operator(
                run_results,
                setting=setting,
                run_id=run_id,
                seed=seed,
            )
        )

metrics_per_run = pd.concat(all_metric_frames, ignore_index=True)
metrics_summary = aggregate_runs(metrics_per_run)

metrics_per_run.to_csv(
    OUTPUT_DIR / f"{EXPERIMENT_NAME}_metrics_per_run.csv",
    index=False,
)
metrics_summary.to_csv(
    OUTPUT_DIR / f"{EXPERIMENT_NAME}_metrics_mean_std.csv",
    index=False,
)

print_mean_std_summary(metrics_summary)
metrics_summary

