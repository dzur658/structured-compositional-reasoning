# LOGICAL-SATA zero-shot chain-of-thought baseline
# five stochastic runs, passage grounded


# LSAT-SATA Zero-Shot CoT — Five-Run A100 Evaluation
# Passage-grounded, batched zero-shot chain-of-thought evaluation over five stochastic runs, with strict all-item metrics and robust answer recovery.

from pathlib import Path
import json
import math
import re
import shutil

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

print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")

if not torch.cuda.is_available():
    raise RuntimeError("This notebook requires a CUDA GPU. Select an A100 runtime.")

for device_id in range(torch.cuda.device_count()):
    properties = torch.cuda.get_device_properties(device_id)
    print(
        f"GPU {device_id}: {torch.cuda.get_device_name(device_id)} "
        f"({properties.total_memory / 2**30:.1f} GiB)"
    )

if "A100" not in torch.cuda.get_device_name(0).upper():
    print("Warning: this notebook is tuned for an A100; adaptive OOM backoff remains enabled.")

# Hugging Face access
# meta-llama/Llama-3.1-8B-Instruct is gated. Make sure HF_TOKEN is set in
# the environment before running the model-loading cell.

# Load benchmark data

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

# A100 model and repeat configuration
# The model is loaded in full bf16 on GPU 0, matching the A100 setup used in the LSAT-SATA CoT notebook. Five runs use seeds 0..4 with temperature 0.7; greedy decoding can be restored by setting DO_SAMPLE=False.

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"

NUM_RUNS = 5
RUN_SEEDS = [0, 1, 2, 3, 4]

# Five stochastic repeats match the existing repeat-based evaluation setup.
# Set DO_SAMPLE=False for greedy decoding; repeated runs will then be identical.
DO_SAMPLE = True
TEMPERATURE = 0.7

BATCH_SIZE = 48
MAX_INPUT_TOKENS = 4096
REPAIR_UNPARSED = True
RESUME_COMPLETED_RUNS = True

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

# Robust parsing, strict metrics, and batched generation
# Unparsed outputs are counted as incorrect in accuracy/F1. The notebook first applies a conservative multi-pattern parser and then sends only unresolved generations through a short, deterministic repair pass. Raw generations and parser methods are saved for auditing.

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

    # Length bucketing substantially reduces padding waste while preserving
    # output order through _row_id.
    prepared["_prompt_tokens"] = [
        len(tokenizer.encode(prompt, add_special_tokens=False))
        for prompt in tqdm(prepared["_prompt"], desc="Measuring prompt lengths")
    ]
    return prepared.sort_values("_prompt_tokens").reset_index(drop=True)


def _generate_texts_with_backoff(
    prompts,
    *,
    max_new_tokens,
    do_sample,
    temperature,
):
    """
    Generate one batch. If an A100 still runs out of memory because a batch
    contains unusually long passages, split only that batch and retry.
    """
    try:
        encoded = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_INPUT_TOKENS,
        )
        encoded = {key: value.to(MODEL_DEVICE) for key, value in encoded.items()}

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

        with torch.inference_mode():
            generated = model.generate(**generation_kwargs)

        continuation = generated[:, encoded["input_ids"].shape[1]:]
        return tokenizer.batch_decode(continuation, skip_special_tokens=True)

    except RuntimeError as error:
        if "out of memory" not in str(error).lower():
            raise
        if "generated" in locals():
            del generated
        if "encoded" in locals():
            del encoded
        torch.cuda.empty_cache()
        if len(prompts) == 1:
            raise
        midpoint = len(prompts) // 2
        print(
            f"CUDA OOM for batch of {len(prompts)}; retrying as "
            f"{midpoint} + {len(prompts) - midpoint}."
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

# Passage-grounded zero-shot CoT prompt

def build_lsata_cot_prompt(row):
    return f"""
You are a reasoning assistant.

You MUST follow this exact format:
FINAL ANSWER: X

Where X is ONE of: A, B, C, or D.
First provide concise step-by-step reasoning grounded only in the passage.
Then, on a new line, provide the final answer exactly as:

FINAL ANSWER: X

Reading-comprehension multiple-choice question:

Passage: {row['paragraph']}

Question: {row['question']}

A. {row['A']}
B. {row['B']}
C. {row['C']}
D. {row['D']}

Reason from the passage, then output the required FINAL ANSWER line.
""".strip()


prepared = prepare_prompt_table(qa_df, build_lsata_cot_prompt)
prepared[["qa_type", "_prompt_tokens"]].groupby("qa_type").describe()

# Run five repeats and aggregate mean ± standard deviation

EXPERIMENT_NAME = "lsata_zeroshot_cot_5runs"
SETTING = "0shot_cot"
MAX_NEW_TOKENS = 300

OUTPUT_DIR = Path(EXPERIMENT_NAME)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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
    "setting": SETTING,
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

metrics_frames = []

for run_id, seed in enumerate(RUN_SEEDS):
    run_path = OUTPUT_DIR / f"{EXPERIMENT_NAME}_{SETTING}_run{run_id}.csv"

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
        print(f"Loading completed run {run_id} from {run_path}")
        run_results = cached
    else:
        run_results = run_batched_inference(
            prepared,
            run_id=run_id,
            seed=seed,
            setting=SETTING,
            max_new_tokens=MAX_NEW_TOKENS,
            repair_unparsed=REPAIR_UNPARSED,
        )
        run_results.to_csv(run_path, index=False)
        print(f"Saved {run_path}")

    metrics_frames.append(
        evaluate_by_operator(
            run_results,
            setting=SETTING,
            run_id=run_id,
            seed=seed,
        )
    )

metrics_per_run = pd.concat(metrics_frames, ignore_index=True)
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

# Package all CSV results

archive_path = shutil.make_archive(
    EXPERIMENT_NAME,
    "zip",
    root_dir=OUTPUT_DIR,
)
print(f"Created {archive_path}")
