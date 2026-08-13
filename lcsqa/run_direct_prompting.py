
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from pathlib import Path
import gc
import json
import math
import re
import shutil

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from sklearn.metrics import precision_recall_fscore_support
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")

if not torch.cuda.is_available():
    raise RuntimeError("This notebook requires a CUDA GPU. Select an A100 runtime.")

# Load LCSQA data

from datasets import load_dataset

HF_DATASET = "ojayy/logical-csqa"
DATA_FILES_BY_SPLIT = {
    "train": "train_all_hf.json",
    "validation": "dev_all_hf.json",
}
SPLIT = "validation"

LABELS = ["A", "B", "C", "D"]
LABEL_MAP = {0: "A", 1: "B", 2: "C", 3: "D"}
OPERATORS = ["AND", "OR", "NEITHER", "Mixed"]


def load_lcsqa(split: str = SPLIT) -> pd.DataFrame:
    filename = DATA_FILES_BY_SPLIT[split]
    ds = load_dataset(HF_DATASET, data_files={split: filename}, split=split)
    rows = []

    for item in ds:
        choices = item["choices"]
        raw_label = item["label"]

        if isinstance(raw_label, str) and raw_label.strip().upper() in LABELS:
            label_letter = raw_label.strip().upper()
        else:
            label_letter = LABEL_MAP[int(raw_label)]

        qa_type = str(item["qa_type"]).strip()
        qa_type = {
            "MIXED": "Mixed",
            "mixed": "Mixed",
            "NNOR": "NEITHER",
            "NEITHER/NOR": "NEITHER",
        }.get(qa_type, qa_type)

        rows.append({
            "question": item["question"],
            "A": choices[0],
            "B": choices[1],
            "C": choices[2],
            "D": choices[3],
            "correct_label": label_letter,
            "correct_answer_text": choices[LABELS.index(label_letter)],
            "qa_type": qa_type,
        })

    dataframe = pd.DataFrame(rows)

    unexpected_operators = sorted(set(dataframe["qa_type"]) - set(OPERATORS))
    if unexpected_operators:
        print(f"Warning: unexpected qa_type values: {unexpected_operators}")

    print(f"Loaded {len(dataframe)} LCSQA {split} examples from {HF_DATASET}")
    print(dataframe["qa_type"].value_counts(dropna=False))
    return dataframe


qa_df = load_lcsqa()
qa_df.head()


SLICE_SPECS = {
    "first_250": (0, 250),
    "second_250": (250, 500),
}
EVALUATION_SLICES = list(SLICE_SPECS)

parts = []

for operator in OPERATORS:
    operator_df = (
        qa_df[qa_df["qa_type"] == operator]
        .reset_index(drop=True)
    )

    required_examples = max(stop for _, stop in SLICE_SPECS.values())
    if len(operator_df) < required_examples:
        raise ValueError(
            f"{operator} has only {len(operator_df)} examples; "
            f"at least {required_examples} are required."
        )

    for slice_name, (start, stop) in SLICE_SPECS.items():
        slice_df = operator_df.iloc[start:stop].copy()
        slice_df["evaluation_slice"] = slice_name
        slice_df["operator_row_index"] = np.arange(start, stop)
        parts.append(slice_df)
        print(
            f"{slice_name:<12} | {operator:<10}: "
            f"rows {start}:{stop} -> {len(slice_df)} examples"
        )


eval_df = pd.concat(parts, ignore_index=True)

print(f"\nTotal batched evaluation examples: {len(eval_df)}")
print("Examples in each independent slice:")
print(
    eval_df.groupby(["evaluation_slice", "qa_type"])
    .size()
    .rename("n")
    .reset_index()
)

# A100 model and five-run configuration

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"

NUM_RUNS = 5
RUN_SEEDS = [0, 1, 2, 3, 4]


DO_SAMPLE = True
TEMPERATURE = 0.7

BATCH_SIZE = 32
MAX_INPUT_TOKENS = 2048
MAX_NEW_TOKENS = 8

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

free_bytes, total_bytes = torch.cuda.mem_get_info()
print(f"Loaded {MODEL_NAME} in bf16 on {MODEL_DEVICE}.")
print(
    f"Runs={NUM_RUNS}, seeds={RUN_SEEDS}, batch_size={BATCH_SIZE}, "
    f"temperature={TEMPERATURE}, sampling={DO_SAMPLE}"
)
print(
    f"GPU memory after model load: "
    f"{free_bytes / 2**30:.2f} GiB free / {total_bytes / 2**30:.2f} GiB total"
)

FEWSHOT_POOL = {
    "AND": [
        """Question: Sammy wanted to go to where the people were. Where might he go?
A. social venues AND quiet retreats
B. local events AND social venues
C. sports arenas AND quiet retreats
D. sports arenas AND train platforms

Answer: B""",
        """Question: The fox walked from the city into the forest, what was it looking for?
A. suitable prey AND shelter from disturbances
B. urban garden AND farm fields
C. natural habitat AND farm fields
D. suitable prey AND neighborhood pets

Answer: A""",
        """Question: What home entertainment equipment requires cable?
A. wireless speaker AND portable projector
B. television AND home theater system
C. wireless speaker AND gaming console
D. digital frame AND portable projector

Answer: B""",
    ],
    "OR": [
        """Question: Sammy wanted to go to where the people were. Where might he go?
A. quiet retreats OR empty parks
B. local events OR empty parks
C. sports arenas OR empty parks
D. train platforms OR empty parks

Answer: B""",
        """Question: The forgotten leftovers had gotten quite old, he found it covered in mold in the back of his what?
A. living room OR dining area
B. utility drawer OR kitchen counter
C. living room OR utility drawer
D. cold storage OR utility drawer

Answer: D""",
        """Question: The fox walked from the city into the forest, what was it looking for?
A. natural habitat OR urban garden
B. farm fields OR city park
C. suitable prey OR neighborhood pets
D. urban garden OR farm fields

Answer: C""",
    ],
    "NEITHER": [
        """Question: Sammy wanted to go to where the people were. Where might he go?
A. NEITHER local events NOR train platforms
B. NEITHER sports arenas NOR train platforms
C. NEITHER social venues NOR sports arenas
D. NEITHER social venues NOR quiet retreats

Answer: B""",
        """Question: The forgotten leftovers had gotten quite old, he found it covered in mold in the back of his what?
A. NEITHER cold storage NOR dining area
B. NEITHER sealed container NOR living room
C. NEITHER food cabinet NOR utility drawer
D. NEITHER living room NOR utility drawer

Answer: D""",
        """Question: The only baggage the woman checked was a drawstring bag, where was she heading with it?
A. NEITHER family gathering NOR local gathering
B. NEITHER airport travel NOR local gathering
C. NEITHER local gathering NOR short trip
D. NEITHER business engagement NOR local gathering

Answer: C""",
    ],
    "Mixed": [
        """Question: The fox walked from the city into the forest, what was it looking for?
A. natural habitat AND shelter from disturbances
B. farm fields OR neighborhood pets
C. city park AND neighborhood pets
D. farm fields AND neighborhood pets

Answer: A""",
        """Question: Google Maps and other highway and street GPS services have replaced what?
A. historical navigation logs AND tourist information centers
B. NEITHER printed road maps NOR route planning software
C. manual navigation techniques AND tourist information centers
D. manual navigation techniques AND route planning software

Answer: D""",
        """Question: Sammy wanted to go to where the people were. Where might he go?
A. sports arenas AND train platforms
B. NEITHER local events NOR social venues
C. local events AND social venues
D. quiet retreats OR empty parks

Answer: C""",
    ],
}

for operator, examples in FEWSHOT_POOL.items():
    print(f"{operator:<10}: {len(examples)} demonstrations")

# Direct-answer prompt builder

def build_lcsqa_direct_prompt(row, n_shot):
    if n_shot not in (0, 1, 2, 3):
        raise ValueError("n_shot must be one of 0, 1, 2, or 3")

    operator = row["qa_type"]
    examples = FEWSHOT_POOL[operator][:n_shot]
    fewshot_text = "\n\n".join(examples)
    if fewshot_text:
        fewshot_text += "\n\nNow answer the following question:\n\n"

    current_question = (
        f"Question: {row['question']}\n"
        f"A. {row['A']}\n"
        f"B. {row['B']}\n"
        f"C. {row['C']}\n"
        f"D. {row['D']}\n\n"
        "Answer:"
    )

    instruction = (
        "Answer the following commonsense question by selecting the correct option.\n"
        "Respond with ONLY the single capital letter A, B, C, or D. "
        "Do not include explanation or punctuation.\n\n"
    )

    return instruction + fewshot_text + current_question


print(build_lcsqa_direct_prompt(eval_df.iloc[0], n_shot=1))

INVALID_LABEL = "__INVALID__"


def normalize_label(value):
    if pd.isna(value):
        return None
    value = str(value).strip().upper()
    return value if value in LABELS else None


def parse_mcq_answer(text):
    """
    Parse an A/B/C/D answer, prioritizing explicit final-answer markers and
    safe answer-only/last-line fallbacks.

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
        ("boxed", r"\\boxed\s*\{\s*([ABCD])\s*\}"),
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

    cleaned_whole = re.sub(r"[\s\(\)\[\]\{\}`*_.:,\-]+", "", text).upper()
    if cleaned_whole in LABELS:
        return cleaned_whole, "answer_only"


    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in reversed(lines[-4:]):
        cleaned_line = re.sub(
            r"^(?:FINAL\s*)?(?:ANSWER|CHOICE|OPTION)?\s*(?:IS\s*)?[:=\-]?\s*",
            "",
            line,
            flags=re.IGNORECASE,
        )
        cleaned_line = re.sub(
            r"[\s\(\)\[\]\{\}`*_.:,\-]+", "", cleaned_line
        ).upper()
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
    """Strict metrics over every benchmark item."""
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
    initial_parsed = initial_methods.ne("unparsed") & initial_methods.ne("empty")
    repaired = results_df["parse_method"].fillna("").astype(str).str.startswith("repair:")
    initial_failures = int((~initial_parsed).sum())
    repair_recovery_rate = (
        float(repaired.sum() / initial_failures) if initial_failures else 0.0
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


def evaluate_by_slice_and_operator(results_df, setting, run_id, seed):
    """Compute independent metrics for each 250-example slice."""
    rows = []

    for slice_name in EVALUATION_SLICES:
        slice_results = results_df[
            results_df["evaluation_slice"] == slice_name
        ]

        for operator in OPERATORS + ["ALL"]:
            subset = (
                slice_results
                if operator == "ALL"
                else slice_results[slice_results["qa_type"] == operator]
            )
            if len(subset) == 0:
                continue

            row = {
                "evaluation_slice": slice_name,
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

    group_columns = ["evaluation_slice", "setting", "operator"]

    grouped = (
        metrics_per_run
        .groupby(group_columns, sort=False)[metric_columns]
        .agg(["mean", "std"])
        .reset_index()
    )
    grouped.columns = [
        "_".join(str(part) for part in column if part).rstrip("_")
        if isinstance(column, tuple)
        else column
        for column in grouped.columns
    ]

    fixed = (
        metrics_per_run
        .groupby(group_columns, sort=False)
        .agg(n=("n", "first"), runs=("run", "nunique"))
        .reset_index()
    )

    return fixed.merge(grouped, on=group_columns, how="left")


def print_mean_std_summary(summary_df):
    for slice_name in EVALUATION_SLICES:
        slice_summary = summary_df[
            summary_df["evaluation_slice"] == slice_name
        ]

        print(f"\n{slice_name.upper()} — FIVE-RUN MEAN ± SAMPLE STANDARD DEVIATION")
        print("=" * 100)
        print(
            f"{'Setting':<12} {'Operator':<10} {'N':>6} "
            f"{'Accuracy':>19} {'Macro-F1':>19} {'Answer rate':>19}"
        )
        print("-" * 100)

        for _, row in slice_summary.iterrows():
            accuracy = f"{row['accuracy_mean']:.4f} ± {row['accuracy_std']:.4f}"
            macro_f1 = f"{row['macro_f1_mean']:.4f} ± {row['macro_f1_std']:.4f}"
            answer_rate = (
                f"{row['answer_rate_mean']:.4f} ± "
                f"{row['answer_rate_std']:.4f}"
            )
            print(
                f"{row['setting']:<12} {row['operator']:<10} "
                f"{int(row['n']):>6} {accuracy:>19} "
                f"{macro_f1:>19} {answer_rate:>19}"
            )


def prepare_prompt_table(qa_dataframe, prompt_builder):
    prepared = qa_dataframe.copy().reset_index(drop=True)
    prepared["_row_id"] = np.arange(len(prepared))
    prepared["_prompt"] = [
        prompt_builder(row) for _, row in prepared.iterrows()
    ]

    # Length bucketing reduces padding waste while _row_id restores benchmark order.
    prepared["_prompt_tokens"] = [
        len(tokenizer.encode(prompt, add_special_tokens=False))
        for prompt in tqdm(prepared["_prompt"], desc="Measuring prompt lengths")
    ]
    return prepared.sort_values("_prompt_tokens").reset_index(drop=True)


def _generate_one_batch(prompts, *, max_new_tokens, do_sample, temperature):
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

        input_length = encoded["input_ids"].shape[1]
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

        continuation = generated[:, input_length:]
        return tokenizer.batch_decode(
            continuation,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

    finally:
        if generated is not None:
            del generated
        if encoded is not None:
            del encoded


def generate_texts_with_backoff(
    prompts,
    *,
    max_new_tokens,
    do_sample,
    temperature,
):
    """Generate a batch, iteratively splitting only batches that OOM."""
    pending = [(0, list(prompts))]
    completed = {}

    while pending:
        start_index, current_prompts = pending.pop(0)

        try:
            outputs = _generate_one_batch(
                current_prompts,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
            )
            for offset, output in enumerate(outputs):
                completed[start_index + offset] = output

        except (torch.cuda.OutOfMemoryError, RuntimeError) as error:
            if not isinstance(error, torch.cuda.OutOfMemoryError) and (
                "out of memory" not in str(error).lower()
            ):
                raise

            gc.collect()
            torch.cuda.empty_cache()

            if len(current_prompts) == 1:
                prompt_tokens = len(
                    tokenizer.encode(current_prompts[0], add_special_tokens=False)
                )
                raise RuntimeError(
                    "CUDA OOM for one prompt. Restart the runtime and check for "
                    f"another GPU process or duplicate model. Prompt tokens={prompt_tokens}."
                ) from error

            midpoint = len(current_prompts) // 2
            left = current_prompts[:midpoint]
            right = current_prompts[midpoint:]

            print(
                f"CUDA OOM for batch of {len(current_prompts)}; "
                f"splitting into {len(left)} + {len(right)}."
            )
            pending.insert(0, (start_index + midpoint, right))
            pending.insert(0, (start_index, left))

        finally:
            gc.collect()
            torch.cuda.empty_cache()

    return [completed[index] for index in range(len(prompts))]


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
    """Run one stochastic repeat using direct generated-letter predictions."""
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

        raw_texts = generate_texts_with_backoff(
            prompts,
            max_new_tokens=max_new_tokens,
            do_sample=DO_SAMPLE,
            temperature=TEMPERATURE,
        )

        initial = [parse_mcq_answer(text) for text in raw_texts]
        predictions = [parsed[0] for parsed in initial]
        initial_methods = [parsed[1] for parsed in initial]
        parse_methods = initial_methods.copy()
        repair_texts = [None] * len(batch)

        failed_positions = [
            position
            for position, prediction in enumerate(predictions)
            if prediction is None
        ]

        if repair_unparsed and failed_positions:
            repair_prompts = [
                build_repair_prompt(prompts[position], raw_texts[position])
                for position in failed_positions
            ]
            repair_outputs = generate_texts_with_backoff(
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
            column for column in batch.columns
            if not column.startswith("_")
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

    parsed_count = int(results["predicted_label"].notna().sum())
    repaired_count = int(
        results["parse_method"].astype(str).str.startswith("repair:").sum()
    )
    print(
        f"{setting} run {run_id}: parsed {parsed_count}/{len(results)} "
        f"({parsed_count / len(results):.2%}); repaired {repaired_count}."
    )
    return results


EXPERIMENT_NAME = "lcsqa_0to3shot_direct_letter_5runs_two_slices"
OUTPUT_DIR = Path(EXPERIMENT_NAME)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SHOT_SETTINGS = [0, 1, 2, 3]

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
    "slice_specs": {
        name: [start, stop]
        for name, (start, stop) in SLICE_SPECS.items()
    },
    "examples_per_operator_per_slice": 250,
    "prediction_method": "direct_generated_letter",
    "batched_inference": True,
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
        eval_df,
        lambda row, n=n_shot: build_lcsqa_direct_prompt(row, n),
    )

    for run_id, seed in enumerate(RUN_SEEDS):
        combined_run_path = (
            OUTPUT_DIR
            / f"{EXPERIMENT_NAME}_{setting}_run{run_id}_all_slices.csv"
        )

        can_resume = False
        if RESUME_COMPLETED_RUNS and combined_run_path.exists():
            cached = pd.read_csv(combined_run_path)
            required_cached_columns = {
                "evaluation_slice",
                "predicted_label",
                "correct_label",
                "qa_type",
                "raw_generation",
                "initial_parse_method",
                "parse_method",
            }
            can_resume = (
                len(cached) == len(eval_df)
                and required_cached_columns.issubset(cached.columns)
                and set(cached["evaluation_slice"].dropna().unique())
                    == set(EVALUATION_SLICES)
            )

        if can_resume:
            print(
                f"Loading completed {setting} run {run_id} "
                f"from {combined_run_path}"
            )
            run_results = cached
        else:
            run_results = run_batched_inference(
                prepared,
                run_id=run_id,
                seed=seed,
                setting=setting,
                max_new_tokens=MAX_NEW_TOKENS,
                batch_size=BATCH_SIZE,
                repair_unparsed=REPAIR_UNPARSED,
            )
            run_results.to_csv(combined_run_path, index=False)
            print(f"Saved combined batched outputs: {combined_run_path}")

        # Save fully separate raw-result files for the two requested slices.
        for slice_name in EVALUATION_SLICES:
            slice_run_results = run_results[
                run_results["evaluation_slice"] == slice_name
            ].copy()

            expected_slice_size = 250 * len(OPERATORS)
            if len(slice_run_results) != expected_slice_size:
                raise RuntimeError(
                    f"{slice_name} contains {len(slice_run_results)} rows; "
                    f"expected {expected_slice_size}."
                )

            slice_run_path = (
                OUTPUT_DIR
                / f"{EXPERIMENT_NAME}_{slice_name}_{setting}_run{run_id}.csv"
            )
            slice_run_results.to_csv(slice_run_path, index=False)
            print(f"Saved separate {slice_name} outputs: {slice_run_path}")

        all_metric_frames.append(
            evaluate_by_slice_and_operator(
                run_results,
                setting=setting,
                run_id=run_id,
                seed=seed,
            )
        )

metrics_per_run = pd.concat(all_metric_frames, ignore_index=True)
metrics_summary = aggregate_runs(metrics_per_run)

# Combined metric tables retain evaluation_slice as an explicit column.
combined_per_run_path = OUTPUT_DIR / f"{EXPERIMENT_NAME}_metrics_per_run.csv"
combined_summary_path = OUTPUT_DIR / f"{EXPERIMENT_NAME}_metrics_mean_std.csv"
metrics_per_run.to_csv(combined_per_run_path, index=False)
metrics_summary.to_csv(combined_summary_path, index=False)

print(f"Saved {combined_per_run_path}")
print(f"Saved {combined_summary_path}")


for slice_name in EVALUATION_SLICES:
    slice_metrics_per_run = metrics_per_run[
        metrics_per_run["evaluation_slice"] == slice_name
    ].copy()
    slice_metrics_summary = metrics_summary[
        metrics_summary["evaluation_slice"] == slice_name
    ].copy()

    slice_per_run_path = (
        OUTPUT_DIR / f"{slice_name}_metrics_per_run.csv"
    )
    slice_summary_path = (
        OUTPUT_DIR / f"{slice_name}_metrics_mean_std.csv"
    )

    slice_metrics_per_run.to_csv(slice_per_run_path, index=False)
    slice_metrics_summary.to_csv(slice_summary_path, index=False)

    print(f"Saved {slice_per_run_path}")
    print(f"Saved {slice_summary_path}")

print_mean_std_summary(metrics_summary)
metrics_summary
