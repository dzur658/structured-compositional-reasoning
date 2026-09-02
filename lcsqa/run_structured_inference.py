import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from datasets import load_dataset
from vllm import LLM, SamplingParams

from utils.prompting import render_prompt, extract_json_object
from utils.scoring import (
    true_prob_from_logprobs,
    ab_probs_from_logprobs,
    parse_verbalized_confidence,
    extract_atomics,
)
from utils.vllm_utils import llm_engine_kwargs, cleanup_llm
from utils.ilp import select_answer
from utils.logging_utils import RunLogger
from utils.metrics import compute_metrics, print_metrics

MODEL = "meta-llama/Llama-3.1-8B-Instruct"
TEMPERATURE = 0.0
MAX_TOKENS = 512
MAX_MODEL_LEN = 4096

HF_DATASET = "ojayy/logical-csqa"
DATA_FILES_BY_SPLIT = {
    "train": "train_all_hf.json",
    "validation": "dev_all_hf.json",
}
SPLIT = "validation"

VALID_QA_TYPES = ["AND", "OR", "NEITHER", "Mixed"]
SAMPLE_QA_TYPES = ["AND", "OR", "NEITHER", "Mixed"]
SAMPLES_PER_OPERATOR = 250
SAMPLE_OFFSET = 0

RESULTS_DIR = Path("./results-lcsqa-or-fix")
RESULTS_DIR.mkdir(exist_ok=True)

_OPTION_KEYS = ["A", "B", "C", "D"]


def _canonical_qa_type(raw: str, idx: int) -> str:
    key = str(raw).strip().casefold()
    aliases = {
        "and": "AND",
        "or": "OR",
        "neither": "NEITHER",
        "neither/nor": "NEITHER",
        "neither nor": "NEITHER",
        "mixed": "Mixed",
        "mix": "Mixed",
    }
    canonical = aliases.get(key)
    if canonical is None:
        s = str(raw).strip()
        if s in VALID_QA_TYPES:
            return s
        raise ValueError(f"record {idx} has invalid qa_type: {raw!r} (expected one of {VALID_QA_TYPES})")
    return canonical


@dataclass(frozen=True)
class LcsqaExample:
    question_id: str
    question: str
    choices: list[str]
    label: int
    qa_type: str


def _normalize_record(record: dict[str, Any], idx: int) -> LcsqaExample:
    question = str(record["question"]).strip()
    choices = list(record["choices"])
    if len(choices) != 4:
        raise ValueError(f"record {idx} does not have 4 choices.")
    choices = [str(choice).strip() for choice in choices]

    label = int(record["label"])
    if label < 0 or label > 3:
        raise ValueError(f"record {idx} has invalid label: {label}")

    qa_type = _canonical_qa_type(record["qa_type"], idx)

    return LcsqaExample(
        question_id=str(record.get("id", f"{SPLIT}-{idx}")),
        question=question,
        choices=choices,
        label=label,
        qa_type=qa_type,
    )


def load_validation_examples(split: str = SPLIT) -> list[LcsqaExample]:
    filename = DATA_FILES_BY_SPLIT[split]
    ds = load_dataset(HF_DATASET, data_files={split: filename}, split=split)
    return [_normalize_record(row, idx) for idx, row in enumerate(ds)]


def _bucket_by_qa_type(examples: list[LcsqaExample]) -> dict[str, list[LcsqaExample]]:
    by_type: dict[str, list[LcsqaExample]] = {qa_type: [] for qa_type in SAMPLE_QA_TYPES}
    for example in examples:
        if example.qa_type in SAMPLE_QA_TYPES:
            by_type[example.qa_type].append(example)
    return by_type


def sample_stratified(
    examples: list[LcsqaExample],
    *,
    samples_per_operator: int | None = None,
    sample_offset: int | None = None,
) -> list[LcsqaExample]:
    n = samples_per_operator if samples_per_operator is not None else SAMPLES_PER_OPERATOR
    offset = sample_offset if sample_offset is not None else SAMPLE_OFFSET
    if n < 0 or offset < 0:
        raise ValueError(f"samples_per_operator and sample_offset must be non-negative, got N={n}, offset={offset}")

    by_type = _bucket_by_qa_type(examples)
    sampled: list[LcsqaExample] = []
    for qa_type in SAMPLE_QA_TYPES:
        bucket = by_type[qa_type]
        end = offset + n
        if len(bucket) < end:
            raise ValueError(
                f"not enough {qa_type} examples: need indices [{offset}:{end}] "
                f"({n} after offset {offset}), found {len(bucket)} in split."
            )
        sampled.extend(bucket[offset:end])

    expected = len(SAMPLE_QA_TYPES) * n
    if len(sampled) != expected:
        raise ValueError(f"expected {expected} sampled examples, found {len(sampled)}")
    return sampled


def _parse_choice(choice: str) -> dict:
    m = re.match(r"^NEITHER\s+(.+?)\s+NOR\s+(.+)$", choice, re.IGNORECASE)
    if m:
        return {"operator": "NEITHER", "a1": m.group(1).strip(), "a2": m.group(2).strip()}
    m = re.match(r"^(.+?)\s+OR\s+(.+)$", choice)
    if m:
        return {"operator": "OR", "a1": m.group(1).strip(), "a2": m.group(2).strip()}
    m = re.match(r"^(.+?)\s+AND\s+(.+)$", choice)
    if m:
        return {"operator": "AND", "a1": m.group(1).strip(), "a2": m.group(2).strip()}
    raise ValueError(f"cannot parse choice: {choice!r}")


def example_to_record(ex: LcsqaExample) -> dict[str, Any]:
    options = {key: _parse_choice(c) for key, c in zip(_OPTION_KEYS, ex.choices)}
    return {
        "question_id": ex.question_id,
        "question": ex.question,
        "options": options,
        "answer": _OPTION_KEYS[ex.label],
        "qa_type": ex.qa_type,
    }

# constraint decomposition
_PROMPT = """
Decompose the question into ALL of its individual constraints. Identify
each atomic constraint separately and state that ALL must hold
simultaneously.

All listed constraints MUST hold simultaneously.

A constraint is a required property of a valid commonsense answer.

Rules:
1. Break the question down into its distinct atomic conditions -- do NOT
   treat it as one combined idea if it genuinely contains separable
   requirements.
2. Do NOT simply restate or paraphrase the question as a whole. Each
   constraint must isolate ONE specific required condition.
3. Phrase every constraint as a requirement on the answer: "the answer
   must be / must have / must satisfy ...".
4. Where the question implies a strict, emphatic, or superlative
   requirement (e.g. "worst", "often", "always", "competing", "first"),
   encode that emphasis directly in the relevant constraint -- do not
   soften it into a generic restatement.
5. Output valid JSON only.

Return exactly this JSON format:

For one constraint:
{"constraints": "One constraint must hold: (1) the answer must ..."}

For multiple constraints:
{"constraints": "Two constraints must hold simultaneously: (1) the answer must ..., and (2) the answer must ..."}

If there are three or more constraints:
{"constraints": "Three constraints must hold simultaneously: (1) the answer must ..., (2) the answer must ..., and (3) the answer must ..."}

Examples:

Example 1:
Question:
Sammy wanted to go to where the people were. Where might he go?

Output:
{"constraints": "Two constraints must hold simultaneously: (1) the answer must be a location Sammy can go to, and (2) that location must ALWAYS have people present."}

Example 2:
Question:
Where would you sit in a chair to watch four-legged animals compete?

Output:
{"constraints": "Four constraints must hold simultaneously: (1) the answer must be a location with a chair to sit in, (2) the location must allow watching, (3) four-legged animals must be present there, and (4) those animals must be COMPETING there, not merely present."}

Example 3:
Question:
What is the worst part of playing games?

Output:
{"constraints": "Two constraints must hold simultaneously: (1) the answer must be something that happens during playing games, and (2) it must be the WORST such thing -- not just a negative or challenging part, but the most negative."}

Example 4:
Question:
He was a very sharp engineer, but when it came to the details his calculations could were often what?

Output:
{"constraints": "Two constraints must hold simultaneously: (1) the answer must describe a property of the engineer's calculations, and (2) it must be that the calculations were OFTEN INCORRECT -- not just sometimes wrong, but wrong with regularity."}

Now generate constraints.

Question:
{{question}}
""".strip()


_MARKER_RE = re.compile(r"\((\d+)\)")
_TRAILING_RE = re.compile(r"[,\s]*(?:\band\b)?\s*$", re.IGNORECASE)


def _split_constraints(text: str) -> list[str]:
    markers = list(_MARKER_RE.finditer(text))
    if not markers:
        return [text.strip()]
    result = []
    for idx, m in enumerate(markers):
        start = m.end()
        end = markers[idx + 1].start() if idx + 1 < len(markers) else len(text)
        chunk = text[start:end].lstrip(" ,")
        chunk = _TRAILING_RE.sub("", chunk).strip().rstrip(".")
        if chunk:
            result.append(chunk)
    return result


class ConstraintGenerator:

    def __init__(self, llm: LLM) -> None:
        self.llm = llm
        self.sampling_params = SamplingParams(temperature=TEMPERATURE, max_tokens=MAX_TOKENS)
        self._template = _PROMPT

    def _build_prompt(self, question: str) -> str:
        return render_prompt(self._template, question=question) + "\nOutput:"

    def _parse(self, text: str) -> list[str]:
        return _split_constraints(str(extract_json_object(text)["constraints"]))

    def generate(self, question: str) -> list[str]:
        outputs = self.llm.generate([self._build_prompt(question)], self.sampling_params)
        return self._parse(outputs[0].outputs[0].text)

    def generate_batch(self, questions: list[str]) -> list[list[str]]:
        prompts = [self._build_prompt(q) for q in questions]
        outputs = self.llm.generate(prompts, self.sampling_params)
        results = []
        for out in outputs:
            try:
                results.append(self._parse(out.outputs[0].text))
            except Exception:
                results.append([])
        return results

# hypothesis generation
HYP_PROMPT = """
You create two hypothesis statements for an atomic answer against one constraint.

Given:
- Question
- Atomic statement
- Constraint

Create:
H+ = a positive hypothesis saying the atomic satisfies the constraint.
H- = a negative hypothesis saying the atomic does not satisfy the constraint.

Rules:
1. Use the full atomic statement exactly as written.
2. Use the full constraint exactly as written.
3. Do not simplify or replace the atomic statement.
4. H+ and H- must be logical opposites.
5. H+ must say the atomic satisfies the constraint.
6. H- must say the atomic does not satisfy the constraint.
7. Do not judge which hypothesis is correct.
8. Output valid JSON only.

Example 1:
Question:
Where is homemade food often stored?

Atomic statement:
food container

Constraint:
The answer must be somewhere homemade food is often kept after preparation.

Output:
{
  "H+": "Food container satisfies the requirement of being a typical place or container where homemade food is normally kept after preparation.",
  "H-": "Food container does not satisfy the requirement of being a typical place or container where homemade food is normally kept after preparation."
}

Example 2:
Question:
A squirrel ran up to its home, where is the small dog likely to be barking up?

Atomic statement:
backyard patio

Constraint:
The answer must be the location of the squirrel's home.

Output:
{
  "H+": "Backyard patio satisfies the requirement of being the location of the squirrel's home.",
  "H-": "Backyard patio does not satisfy the requirement of being the location of the squirrel's home."
}

Now generate hypotheses.

Question:
{{question}}

Atomic statement:
{{atomic}}

Constraint:
{{constraint}}"""


class HypothesisGenerator:

    def __init__(self, llm: LLM) -> None:
        self.llm = llm
        self.sampling_params = SamplingParams(temperature=TEMPERATURE, max_tokens=MAX_TOKENS)
        self._template = HYP_PROMPT

    def _build_prompt(self, question: str, atomic: str, constraint: str) -> str:
        return render_prompt(self._template, question=question, atomic=atomic, constraint=constraint) + "\nOutput:"

    def generate_batch(self, examples: list[tuple[str, list[str], list[str]]]) -> list[dict]:
        """One llm.generate call for every (question, atomic, constraint) triplet
        across all examples. Returns one result dict per example."""
        index: list[tuple[int, str, str]] = []
        prompts: list[str] = []
        for ex_idx, (q, atomics, constraints) in enumerate(examples):
            for a in atomics:
                for c in constraints:
                    index.append((ex_idx, a, c))
                    prompts.append(self._build_prompt(q, a, c))

        if not prompts:
            return [{} for _ in examples]

        outputs = self.llm.generate(prompts, self.sampling_params)
        results: list[dict] = [{} for _ in examples]
        for (ex_idx, a, c), out in zip(index, outputs):
            try:
                data = extract_json_object(out.outputs[0].text)
                h_plus = data["H+"]
                h_minus = data["H-"]
            except Exception:
                h_plus = h_minus = ""
            results[ex_idx].setdefault(a, {})[c] = {"H+": h_plus, "H-": h_minus}
        return results

PLAUSIBILITY_PROMPT_CONFIDENCE_MC = """
Decide which of two competing hypotheses about an atomic answer statement
is correct, with respect to a required commonsense constraint.

You are given:
1. A commonsense question.
2. A constraint describing what a valid answer must do.
3. One atomic answer statement.
4. Two hypotheses about that atomic statement:
   Option A: the atomic SATISFIES the constraint.
   Option B: the atomic FAILS the constraint.


Important rules:
- Judge the atomic against the constraint, not against other answer
  options.
- Do not reward surface word overlap or topic similarity alone.
- Do not penalize an atomic merely because it is not the most obvious or
  stereotypical answer -- judge whether THIS atomic fulfills the required
  relation under a reasonable reading of the constraint.
- Preserve important roles in the constraint, such as "first", "worst",
  "best", "typical", "where", "who", "why", or "what".
- This is a forced choice: you must commit to exactly one letter, A or B,
  even in borderline cases.
- Output ONLY the single letter A or B. Do not output any other words,
  punctuation, explanation, or reasoning of any kind.

Examples:

Example 1:
Question:
Where can you see a mountain in your own home?

Constraint:
The answer must be something in or visible from inside a home that lets a
person see an actual mountain or a representation of one.

Atomic statement:
a window facing the mountains

Option A:
The atomic statement satisfies the constraint.

Option B:
The atomic statement fails the constraint.

Answer: A

Example 2:
Question:
Where can you see a mountain in your own home?

Constraint:
The answer must be something in or visible from inside a home that lets a
person see an actual mountain or a representation of one.

Atomic statement:
a mountain-themed restaurant

Option A:
The atomic statement satisfies the constraint.

Option B:
The atomic statement fails the constraint.

Answer: B

Example 3:
Question:
What is the worst part of getting lost in an unfamiliar city?

Constraint:
The answer must be a significant negative consequence or experience of
being lost in an unfamiliar city.

Atomic statement:
discovering new restaurants

Option A:
The atomic statement satisfies the constraint.

Option B:
The atomic statement fails the constraint.

Answer: B

Example 4:
Question:
Why would someone choose to work from home?

Constraint:
The answer must be a genuine personal or practical benefit of working from
home.

Atomic statement:
fewer interruptions from coworkers walking by

Option A:
The atomic statement satisfies the constraint.

Option B:
The atomic statement fails the constraint.

Answer: A


Now judge the given case. Output ONLY a single letter, A or B -- nothing else.

Question:
{{question}}

Constraint:
{{constraint}}

Atomic statement:
{{atomic_statement}}

Option A:
The atomic statement satisfies the constraint.

Option B:
The atomic statement fails the constraint.

Answer:"""


PLAUSIBILITY_PROMPT_TRUEFALSE_PLUS = """
Decide whether a hypothesis about an atomic answer statement is TRUE, with
respect to a required commonsense constraint.

You are given:
1. A commonsense question.
2. A constraint describing what a valid answer must do.
3. One atomic answer statement.
4. A hypothesis: the atomic statement SATISFIES the constraint.


Important rules:
- Judge the atomic against the constraint on its own terms.
- Do not reward surface word overlap or topic similarity alone.
- Do not penalize an atomic merely because it is not the most obvious or
  stereotypical answer -- judge whether THIS atomic fulfills the required
  relation under a reasonable reading of the constraint.
- Preserve important roles in the constraint, such as "first", "worst",
  "best", "typical", "where", "who", "why", or "what".
- Output ONLY the single word True or False. Do not output any other
  words, punctuation, explanation, or reasoning of any kind.

Examples:

Example 1:
Question:
Where can you see a mountain in your own home?

Constraint:
The answer must be something in or visible from inside a home that lets a
person see an actual mountain or a representation of one.

Atomic statement:
a window facing the mountains

Hypothesis: the atomic statement satisfies the constraint.

Answer: True

Example 2:
Question:
Where can you see a mountain in your own home?

Constraint:
The answer must be something in or visible from inside a home that lets a
person see an actual mountain or a representation of one.

Atomic statement:
a mountain-themed restaurant

Hypothesis: the atomic statement satisfies the constraint.

Answer: False

Example 3:
Question:
What is the worst part of getting lost in an unfamiliar city?

Constraint:
The answer must be a significant negative consequence or experience of
being lost in an unfamiliar city.

Atomic statement:
discovering new restaurants

Hypothesis: the atomic statement satisfies the constraint.

Answer: False

Example 4:
Question:
Why would someone choose to work from home?

Constraint:
The answer must be a genuine personal or practical benefit of working from
home.

Atomic statement:
fewer interruptions from coworkers walking by

Hypothesis: the atomic statement satisfies the constraint.

Answer: True

Now judge the given case. Output ONLY True or False -- nothing else.

Question:
{{question}}

Constraint:
{{constraint}}

Atomic statement:
{{atomic_statement}}

Hypothesis: the atomic statement satisfies the constraint.
{{H_plus}}

Answer:"""


PLAUSIBILITY_PROMPT_TRUEFALSE_MINUS = """
Decide whether a hypothesis about an atomic answer statement is TRUE, with
respect to a required commonsense constraint.

You are given:
1. A commonsense question.
2. A constraint describing what a valid answer must do.
3. One atomic answer statement.
4. A hypothesis: the atomic statement FAILS the constraint.

Important rules:
- Judge the atomic against the constraint on its own terms.
- Do not answer True (i.e. "fails") merely because the atomic is not the
  most obvious or stereotypical answer.
- Do not answer False (i.e. "does not fail") merely because the atomic is
  factually true or topically related -- a true, related atomic can still
  genuinely fail the required relation, role, or framing.
- Preserve important roles in the constraint, such as "first", "worst",
  "best", "typical", "where", "who", "why", or "what".
- Output ONLY the single word True or False. Do not output any other
  words, punctuation, explanation, or reasoning of any kind.

Examples:

Example 1:
Question:
Where can you see a mountain in your own home?

Constraint:
The answer must be something in or visible from inside a home that lets a
person see an actual mountain or a representation of one.

Atomic statement:
a window facing the mountains

Hypothesis: the atomic statement fails the constraint.

Answer: False

Example 2:
Question:
Where can you see a mountain in your own home?

Constraint:
The answer must be something in or visible from inside a home that lets a
person see an actual mountain or a representation of one.

Atomic statement:
a mountain-themed restaurant

Hypothesis: the atomic statement fails the constraint.

Answer: True

Example 3:
Question:
What is the worst part of getting lost in an unfamiliar city?

Constraint:
The answer must be a significant negative consequence or experience of
being lost in an unfamiliar city.

Atomic statement:
discovering new restaurants

Hypothesis: the atomic statement fails the constraint.

Answer: True

Example 4:
Question:
Why would someone choose to work from home?

Constraint:
The answer must be a genuine personal or practical benefit of working from
home.

Atomic statement:
fewer interruptions from coworkers walking by

Hypothesis: the atomic statement fails the constraint.

Answer: False

Now judge the given case. Output ONLY True or False -- nothing else.

Question:
{{question}}

Constraint:
{{constraint}}

Atomic statement:
{{atomic_statement}}

Hypothesis: the atomic statement fails the constraint.
{{H_minus}}

Answer:"""


PLAUSIBILITY_PROMPT_TRUEFALSE = (
    PLAUSIBILITY_PROMPT_TRUEFALSE_PLUS,
    PLAUSIBILITY_PROMPT_TRUEFALSE_MINUS,
)

PLAUSIBILITY_PROMPTS: dict[str, object] = {
    "confidence_mc": PLAUSIBILITY_PROMPT_CONFIDENCE_MC,
    "independent": PLAUSIBILITY_PROMPT_TRUEFALSE,
    "generation_sampling": PLAUSIBILITY_PROMPT_CONFIDENCE_MC,
}

# verbalized confidence prompts
VERBALIZED_PROMPT_PLUS = """
State your confidence that a hypothesis about an atomic answer statement is
TRUE, with respect to a required commonsense constraint.

You are given:
1. A commonsense question.
2. A constraint describing what a valid answer must do.
3. One atomic answer statement.
4. A hypothesis: the atomic statement SATISFIES the constraint.

Important rules:
- Judge the atomic against the constraint on its own terms.
- Do not reward surface word overlap or topic similarity alone.
- Do not penalize an atomic merely because it is not the most obvious or
  stereotypical answer -- judge whether THIS atomic fulfills the required
  relation under a reasonable reading of the constraint.
- Preserve important roles in the constraint, such as "first", "worst",
  "best", "typical", "where", "who", "why", or "what".
- Output ONLY a single integer from 0 to 10, where 0 means you are
  completely confident the hypothesis is FALSE and 10 means you are
  completely confident the hypothesis is TRUE. Do not output any other
  words, punctuation, explanation, or reasoning of any kind.

Examples:

Example 1:
Question:
Where can you see a mountain in your own home?

Constraint:
The answer must be something in or visible from inside a home that lets a
person see an actual mountain or a representation of one.

Atomic statement:
a window facing the mountains

Hypothesis: the atomic statement satisfies the constraint.

Confidence: 9

Example 2:
Question:
Where can you see a mountain in your own home?

Constraint:
The answer must be something in or visible from inside a home that lets a
person see an actual mountain or a representation of one.

Atomic statement:
a mountain-themed restaurant

Hypothesis: the atomic statement satisfies the constraint.

Confidence: 1

Example 3:
Question:
What is the worst part of getting lost in an unfamiliar city?

Constraint:
The answer must be a significant negative consequence or experience of
being lost in an unfamiliar city.

Atomic statement:
discovering new restaurants

Hypothesis: the atomic statement satisfies the constraint.

Confidence: 1

Example 4:
Question:
Why would someone choose to work from home?

Constraint:
The answer must be a genuine personal or practical benefit of working from
home.

Atomic statement:
fewer interruptions from coworkers walking by

Hypothesis: the atomic statement satisfies the constraint.

Confidence: 9

Example 5 (gradient set, 1 of 3):
Question:
What sort of life event might draw a large crowd of people together to
celebrate?

Constraint:
The answer must be a significant, public, or widely recognized life event
that people typically gather to celebrate.

Atomic statement:
community festival

Hypothesis: the atomic statement satisfies the constraint.

Confidence: 8

Example 6 (gradient set, 2 of 3):
Question:
What sort of life event might draw a large crowd of people together to
celebrate?

Constraint:
The answer must be a significant, public, or widely recognized life event
that people typically gather to celebrate.

Atomic statement:
charity fundraiser

Hypothesis: the atomic statement satisfies the constraint.

Confidence: 8

Example 7 (gradient set, 3 of 3):
Question:
What sort of life event might draw a large crowd of people together to
celebrate?

Constraint:
The answer must be a significant, public, or widely recognized life event
that people typically gather to celebrate.

Atomic statement:
funeral procession

Hypothesis: the atomic statement satisfies the constraint.

Confidence: 1

Now judge the given case. Output ONLY a single integer from 0 to 10 --
nothing else.

Question:
{{question}}

Constraint:
{{constraint}}

Atomic statement:
{{atomic_statement}}

Hypothesis: the atomic statement satisfies the constraint.
{{H_plus}}

Confidence:"""


VERBALIZED_PROMPT_MINUS = """
State your confidence that a hypothesis about an atomic answer statement is
TRUE, with respect to a required commonsense constraint.

You are given:
1. A commonsense question.
2. A constraint describing what a valid answer must do.
3. One atomic answer statement.
4. A hypothesis: the atomic statement FAILS the constraint.


Important rules:
- Judge the atomic against the constraint on its own terms.
- Do not give a high confidence (i.e. "fails") merely because the atomic is
  not the most obvious or stereotypical answer.
- Do not give a low confidence (i.e. "does not fail") merely because the
  atomic is factually true or topically related -- a true, related atomic
  can still genuinely fail the required relation, role, or framing.
- Preserve important roles in the constraint, such as "first", "worst",
  "best", "typical", "where", "who", "why", or "what".
- Output ONLY a single integer from 0 to 10, where 0 means you are
  completely confident the hypothesis is FALSE (the atomic does NOT fail
  the constraint) and 10 means you are completely confident the hypothesis
  is TRUE (the atomic DOES fail the constraint). Do not output any other
  words, punctuation, explanation, or reasoning of any kind.

Examples:

Example 1:
Question:
Where can you see a mountain in your own home?

Constraint:
The answer must be something in or visible from inside a home that lets a
person see an actual mountain or a representation of one.

Atomic statement:
a window facing the mountains

Hypothesis: the atomic statement fails the constraint.

Confidence: 1

Example 2:
Question:
Where can you see a mountain in your own home?

Constraint:
The answer must be something in or visible from inside a home that lets a
person see an actual mountain or a representation of one.

Atomic statement:
a mountain-themed restaurant

Hypothesis: the atomic statement fails the constraint.

Confidence: 9

Example 3:
Question:
What is the worst part of getting lost in an unfamiliar city?

Constraint:
The answer must be a significant negative consequence or experience of
being lost in an unfamiliar city.

Atomic statement:
discovering new restaurants

Hypothesis: the atomic statement fails the constraint.

Confidence: 9

Example 4:
Question:
Why would someone choose to work from home?

Constraint:
The answer must be a genuine personal or practical benefit of working from
home.

Atomic statement:
fewer interruptions from coworkers walking by

Hypothesis: the atomic statement fails the constraint.

Confidence: 1

Example 5 (gradient set, 1 of 3):
Question:
What sort of life event might draw a large crowd of people together to
celebrate?

Constraint:
The answer must be a significant, public, or widely recognized life event
that people typically gather to celebrate.

Atomic statement:
community festival

Hypothesis: the atomic statement fails the constraint.

Confidence: 2

Example 6 (gradient set, 2 of 3):
Question:
What sort of life event might draw a large crowd of people together to
celebrate?

Constraint:
The answer must be a significant, public, or widely recognized life event
that people typically gather to celebrate.

Atomic statement:
charity fundraiser

Hypothesis: the atomic statement fails the constraint.

Confidence: 2

Example 7 (gradient set, 3 of 3):
Question:
What sort of life event might draw a large crowd of people together to
celebrate?

Constraint:
The answer must be a significant, public, or widely recognized life event
that people typically gather to celebrate.

Atomic statement:
funeral procession

Hypothesis: the atomic statement fails the constraint.

Confidence: 9

Now judge the given case. Output ONLY a single integer from 0 to 10 --
nothing else.

Question:
{{question}}

Constraint:
{{constraint}}

Atomic statement:
{{atomic_statement}}

Hypothesis: the atomic statement fails the constraint.
{{H_minus}}

Confidence:"""


VERBALIZED_PROMPT_TUPLE = (VERBALIZED_PROMPT_PLUS, VERBALIZED_PROMPT_MINUS)
PLAUSIBILITY_PROMPTS["verbalized"] = VERBALIZED_PROMPT_TUPLE


class HypothesisScorer:

    def __init__(self, llm: LLM, plausibility_mode: str = "confidence_mc") -> None:
        if plausibility_mode not in PLAUSIBILITY_PROMPTS:
            raise ValueError(
                f"unknown plausibility_mode={plausibility_mode!r}. "
                f"choose from {sorted(PLAUSIBILITY_PROMPTS)}."
            )
        self.llm = llm
        self.plausibility_mode = plausibility_mode
        self.sampling_params = SamplingParams(temperature=TEMPERATURE, max_tokens=MAX_TOKENS)
        # greedy, single token, top-20 logprobs -- reads p(True) or p(A)/p(B) directly
        self.confidence_sampling_params = SamplingParams(temperature=0, max_tokens=1, logprobs=20)
        # stochastic sampling for generation_sampling: sample n completions, count label frequency
        self.sampling_n_params = SamplingParams(temperature=0.7, max_tokens=1, n=5)

        self._logprob_modes = {"independent", "confidence_mc"}

    @classmethod
    def _parse_independent_side(cls, output) -> float:
        return true_prob_from_logprobs(output)

    @classmethod
    def _parse_mc_confidence(cls, output) -> dict:
        p_a, p_b = ab_probs_from_logprobs(output)
        denom = p_a + p_b
        if denom == 0.0:
            p_plus, p_minus = 0.0, 0.0
        else:
            p_plus = p_a / denom
            p_minus = 1.0 - p_plus
        return {
            "fit_label": "",
            "p_plus": p_plus,
            "p_minus": p_minus,
            "reason": f"p(A)={p_a:.4f}, p(B)={p_b:.4f}",
        }

    @staticmethod
    def _parse_mc_sampling(output) -> dict:
        n = len(output.outputs)
        count_a = sum(1 for o in output.outputs if o.text.strip().upper() == "A")
        count_b = sum(1 for o in output.outputs if o.text.strip().upper() == "B")
        p_plus = count_a / n
        p_minus = count_b / n
        return {
            "fit_label": "",
            "p_plus": p_plus,
            "p_minus": p_minus,
            "reason": f"sampled {n}x: A={count_a}, B={count_b}, other={n - count_a - count_b}",
        }

    def _build_plausibility_prompts(self, question: str, pairs: list[tuple[str, str, dict]]):
        mode = self.plausibility_mode
        jobs: list[tuple] = []
        prompts: list[str] = []

        if mode == "independent":
            plus_tpl, minus_tpl = PLAUSIBILITY_PROMPTS["independent"]
            for a, c, h in pairs:
                prompts.append(render_prompt(
                    plus_tpl, question=question, atomic_statement=a,
                    constraint=c, H_plus=h.get("H+", ""),
                ))
                jobs.append((a, c, "plus"))
                prompts.append(render_prompt(
                    minus_tpl, question=question, atomic_statement=a,
                    constraint=c, H_minus=h.get("H-", ""),
                ))
                jobs.append((a, c, "minus"))
        elif mode in ("confidence_mc", "generation_sampling"):
            template = PLAUSIBILITY_PROMPTS["confidence_mc"]
            for a, c, h in pairs:
                prompts.append(render_prompt(
                    template, question=question, atomic_statement=a,
                    constraint=c, H_plus=h.get("H+", ""), H_minus=h.get("H-", ""),
                ))
                jobs.append((a, c, "single"))
        else:
            raise ValueError(f"_build_plausibility_prompts: unhandled mode {mode!r}")

        return jobs, prompts

    def score(self, question: str, hypotheses: dict) -> dict:
        """hypotheses: {atomic: {constraint: {"H+": str, "H-": str}}}
        returns: {atomic: {constraint: <record>}}"""
        pairs = [
            (a, c, hypotheses[a][c])
            for a, constraint_map in hypotheses.items()
            for c in constraint_map
        ]

        mode = self.plausibility_mode
        result: dict = {a: {} for a in hypotheses}

        if mode == "independent":
            result.update(self._score_independent(question, pairs))
        else:
            result.update(self._score_dependent(question, pairs))

        return result

    def _score_dependent(self, question: str, pairs: list[tuple[str, str, dict]]) -> dict:
        mode = self.plausibility_mode
        jobs, prompts = self._build_plausibility_prompts(question, pairs)

        sp = (
            self.sampling_n_params if mode == "generation_sampling"
            else self.confidence_sampling_params if mode in self._logprob_modes
            else self.sampling_params
        )
        outputs = self.llm.generate(prompts, sp)

        parsed: dict[tuple[str, str], dict] = {}
        for (a, c, _kind), out in zip(jobs, outputs):
            if mode == "confidence_mc":
                parsed[(a, c)] = self._parse_mc_confidence(out)
            elif mode == "generation_sampling":
                parsed[(a, c)] = self._parse_mc_sampling(out)
            else:
                raise ValueError(f"_score_dependent: unhandled mode {mode!r}")

        result: dict = {}
        for a, c, h in pairs:
            p = parsed[(a, c)]
            p_plus, p_minus = p["p_plus"], p["p_minus"]
            active = "H+" if p_plus >= p_minus else "H-"
            result.setdefault(a, {})[c] = {
                "fit_label": p["fit_label"],
                "p_plus": p_plus,
                "p_minus": p_minus,
                "active": active,
                "plausibility_reason": p["reason"],
                "mode": mode,
                "T": p_plus * 5,
                "F": p_minus * 5,
            }

        return result

    def _score_independent(self, question: str, pairs: list[tuple[str, str, dict]]) -> dict:
        jobs, prompts = self._build_plausibility_prompts(question, pairs)
        outputs = self.llm.generate(prompts, self.confidence_sampling_params)

        raw_true_plus: dict[tuple[str, str], float] = {}
        raw_true_minus: dict[tuple[str, str], float] = {}
        for (a, c, kind), out in zip(jobs, outputs):
            try:
                p_true = self._parse_independent_side(out)
            except Exception:
                p_true = 0.0
            if kind == "plus":
                raw_true_plus[(a, c)] = p_true
            else:
                raw_true_minus[(a, c)] = p_true

        result: dict = {}
        for a, c, h in pairs:
            pt_plus = raw_true_plus.get((a, c), 0.0)
            pt_minus = raw_true_minus.get((a, c), 0.0)
            denom = pt_plus + pt_minus
            if denom == 0.0:
                p_plus, p_minus = 0.0, 0.0
            else:
                p_plus = pt_plus / denom
                p_minus = 1.0 - p_plus
            active = "H+" if p_plus >= p_minus else "H-"

            result.setdefault(a, {})[c] = {
                "fit_label": "",
                "p_plus": p_plus,
                "p_minus": p_minus,
                "active": active,
                "plausibility_reason": f"p_true(H+)={pt_plus:.4f}, p_true(H-)={pt_minus:.4f}",
                "mode": "independent",
                "T": p_plus * 5,
                "F": p_minus * 5,
            }

        return result


class HypothesisScorerV2(HypothesisScorer):
    """Adds a 'verbalized' scoring mode on top of HypothesisScorer."""

    def __init__(self, llm, plausibility_mode: str = "confidence_mc") -> None:
        super().__init__(llm, plausibility_mode=plausibility_mode)
        self.verbalized_sampling_params = SamplingParams(temperature=TEMPERATURE, max_tokens=8)

    def _build_plausibility_prompts(self, question: str, pairs: list[tuple[str, str, dict]]):
        if self.plausibility_mode != "verbalized":
            return super()._build_plausibility_prompts(question, pairs)

        plus_tpl, minus_tpl = PLAUSIBILITY_PROMPTS["verbalized"]
        jobs: list[tuple] = []
        prompts: list[str] = []
        for a, c, h in pairs:
            prompts.append(render_prompt(
                plus_tpl, question=question, atomic_statement=a,
                constraint=c, H_plus=h.get("H+", ""),
            ))
            jobs.append((a, c, "plus"))
            prompts.append(render_prompt(
                minus_tpl, question=question, atomic_statement=a,
                constraint=c, H_minus=h.get("H-", ""),
            ))
            jobs.append((a, c, "minus"))
        return jobs, prompts

    def score(self, question: str, hypotheses: dict) -> dict:
        if self.plausibility_mode != "verbalized":
            return super().score(question, hypotheses)
        pairs = [(a, c, hypotheses[a][c]) for a, cm in hypotheses.items() for c in cm]
        result: dict = {a: {} for a in hypotheses}
        result.update(self._score_verbalized(question, pairs))
        return result

    def _score_verbalized(self, question: str, pairs: list[tuple[str, str, dict]]) -> dict:
        jobs, prompts = self._build_plausibility_prompts(question, pairs)
        outputs = self.llm.generate(prompts, self.verbalized_sampling_params)

        raw_conf_plus: dict[tuple[str, str], float] = {}
        raw_conf_minus: dict[tuple[str, str], float] = {}
        for (a, c, kind), out in zip(jobs, outputs):
            try:
                conf = parse_verbalized_confidence(out.outputs[0].text)
            except Exception:
                conf = 0.0
            if kind == "plus":
                raw_conf_plus[(a, c)] = conf
            else:
                raw_conf_minus[(a, c)] = conf

        result: dict = {}
        for a, c, h in pairs:
            conf_plus = raw_conf_plus.get((a, c), 0.0)
            conf_minus = raw_conf_minus.get((a, c), 0.0)
            denom = conf_plus + conf_minus
            if denom == 0.0:
                p_plus, p_minus = 0.0, 0.0
            else:
                p_plus = conf_plus / denom
                p_minus = 1.0 - p_plus
            active = "H+" if p_plus >= p_minus else "H-"

            result.setdefault(a, {})[c] = {
                "fit_label": "",
                "p_plus": p_plus,
                "p_minus": p_minus,
                "active": active,
                "plausibility_reason": f"confidence(H+)={conf_plus:.1f}, confidence(H-)={conf_minus:.1f}",
                "mode": "verbalized",
                "T": p_plus * 5,
                "F": p_minus * 5,
            }
        return result


def score_batch_single_prompt(scorer, questions, hypotheses_list, constraints_list,
                               atomics_list, sp, parser_fn, mode_name):
    """confidence_mc / generation_sampling -- one prompt per (atomic, constraint) pair."""
    all_prompts, all_jobs = [], []
    for ex_idx, (q, h, cons, atoms) in enumerate(
        zip(questions, hypotheses_list, constraints_list, atomics_list)
    ):
        pairs = [(a, c, h[a][c]) for a in atoms for c in cons if a in h and c in h[a]]
        jobs, prompts = scorer._build_plausibility_prompts(q, pairs)
        for (a, c, _kind), p in zip(jobs, prompts):
            all_jobs.append((ex_idx, a, c))
            all_prompts.append(p)
    if not all_prompts:
        return [{} for _ in questions]

    outputs = scorer.llm.generate(all_prompts, sp)

    results = [dict() for _ in questions]
    for (ex_idx, a, c), out in zip(all_jobs, outputs):
        parsed = parser_fn(out)
        p_plus, p_minus = parsed["p_plus"], parsed["p_minus"]
        results[ex_idx].setdefault(a, {})[c] = {
            "fit_label": parsed.get("fit_label", ""),
            "p_plus": p_plus, "p_minus": p_minus,
            "active": "H+" if p_plus >= p_minus else "H-",
            "plausibility_reason": parsed.get("reason", ""),
            "mode": mode_name,
            "T": p_plus * 5, "F": p_minus * 5,
        }
    return results


def score_batch_independent(scorer, questions, hypotheses_list, constraints_list, atomics_list, sp):
    """independent -- two separate prompts per pair, logprob-read, normalized."""
    all_prompts, all_jobs = [], []
    for ex_idx, (q, h, cons, atoms) in enumerate(
        zip(questions, hypotheses_list, constraints_list, atomics_list)
    ):
        pairs = [(a, c, h[a][c]) for a in atoms for c in cons if a in h and c in h[a]]
        jobs, prompts = scorer._build_plausibility_prompts(q, pairs)
        for (a, c, kind), p in zip(jobs, prompts):
            all_jobs.append((ex_idx, a, c, kind))
            all_prompts.append(p)
    if not all_prompts:
        return [{} for _ in questions]

    outputs = scorer.llm.generate(all_prompts, sp)

    raw_true_plus, raw_true_minus = {}, {}
    for (ex_idx, a, c, kind), out in zip(all_jobs, outputs):
        try:
            p_true = scorer._parse_independent_side(out)
        except Exception:
            p_true = 0.0
        key = (ex_idx, a, c)
        (raw_true_plus if kind == "plus" else raw_true_minus)[key] = p_true

    results = [dict() for _ in questions]
    seen_keys = set(list(raw_true_plus.keys()) + list(raw_true_minus.keys()))
    for ex_idx, a, c in seen_keys:
        pt_plus = raw_true_plus.get((ex_idx, a, c), 0.0)
        pt_minus = raw_true_minus.get((ex_idx, a, c), 0.0)
        
        # old scoring method
        denom = pt_plus + pt_minus
        p_plus_old, p_minus_old = (pt_plus / denom, 1.0 - pt_plus / denom) if denom > 0 else (0.0, 0.0)
        
        # new scoring method
        p_plus = pt_plus
        p_minus = 1.0 - p_plus
        
        results[ex_idx].setdefault(a, {})[c] = {
            "fit_label": "", "p_plus": p_plus, "p_minus": p_minus,
            "active": "H+" if p_plus >= p_minus else "H-",
            "plausibility_reason": f"p_true(H+)={pt_plus:.4f}, p_true(H-)={pt_minus:.4f}",
            "mode": "independent",
            "T": p_plus * 5, "F": p_minus * 5,
        }
    return results


def score_batch_verbalized(scorer, questions, hypotheses_list, constraints_list, atomics_list, sp):
    """verbalized -- two separate prompts per pair, free-text confidence, normalized."""
    all_prompts, all_jobs = [], []
    for ex_idx, (q, h, cons, atoms) in enumerate(
        zip(questions, hypotheses_list, constraints_list, atomics_list)
    ):
        pairs = [(a, c, h[a][c]) for a in atoms for c in cons if a in h and c in h[a]]
        jobs, prompts = scorer._build_plausibility_prompts(q, pairs)
        for (a, c, kind), p in zip(jobs, prompts):
            all_jobs.append((ex_idx, a, c, kind))
            all_prompts.append(p)
    if not all_prompts:
        return [{} for _ in questions]

    outputs = scorer.llm.generate(all_prompts, sp)

    raw_conf_plus, raw_conf_minus = {}, {}
    for (ex_idx, a, c, kind), out in zip(all_jobs, outputs):
        try:
            conf = parse_verbalized_confidence(out.outputs[0].text)
        except Exception:
            conf = 0.0
        key = (ex_idx, a, c)
        (raw_conf_plus if kind == "plus" else raw_conf_minus)[key] = conf

    results = [dict() for _ in questions]
    seen_keys = set(list(raw_conf_plus.keys()) + list(raw_conf_minus.keys()))
    for ex_idx, a, c in seen_keys:
        conf_plus = raw_conf_plus.get((ex_idx, a, c), 0.0)
        conf_minus = raw_conf_minus.get((ex_idx, a, c), 0.0)
        denom = conf_plus + conf_minus
        p_plus, p_minus = (conf_plus / denom, 1.0 - conf_plus / denom) if denom > 0 else (0.0, 0.0)
        results[ex_idx].setdefault(a, {})[c] = {
            "fit_label": "", "p_plus": p_plus, "p_minus": p_minus,
            "active": "H+" if p_plus >= p_minus else "H-",
            "plausibility_reason": f"confidence(H+)={conf_plus:.1f}, confidence(H-)={conf_minus:.1f}",
            "mode": "verbalized",
            "T": p_plus * 5, "F": p_minus * 5,
        }
    return results


N_REPEATS = 5
SAMPLE_TEMPERATURE = 0.7
METHODS_TO_RUN = ["confidence_mc", "independent", "generation_sampling", "verbalized"]

sp_logprob = SamplingParams(temperature=SAMPLE_TEMPERATURE, max_tokens=1, logprobs=20)
sp_sampling_n = SamplingParams(temperature=SAMPLE_TEMPERATURE, max_tokens=1, n=5)
sp_verbalized = SamplingParams(temperature=SAMPLE_TEMPERATURE, max_tokens=8)


def run() -> None:
    examples = load_validation_examples()
    sampled = sample_stratified(examples)

    print(f"total examples: {len(examples)}, sampled: {len(sampled)}")
    for qt, n in Counter(ex.qa_type for ex in sampled).items():
        print(f"  {qt}: {n}")

    llm = LLM(**llm_engine_kwargs(MODEL, max_model_len=MAX_MODEL_LEN))

    records = [example_to_record(ex) for ex in sampled]
    atomics_list = [extract_atomics(d["options"]) for d in records]


    unique_questions = list(dict.fromkeys(d["question"] for d in records))
    constraints_cache = {q: [q] for q in unique_questions}

    hypothesis_gen = HypothesisGenerator(llm)
    all_hypotheses = hypothesis_gen.generate_batch(
        [(d["question"], atomics_list[i], constraints_cache[d["question"]]) for i, d in enumerate(records)]
    )

    questions = [d["question"] for d in records]
    constraints_list = [constraints_cache[d["question"]] for d in records]

    all_repeat_results: dict[str, list[dict]] = {}

    for method in METHODS_TO_RUN:
        print(f"\n{'='*70}\n  METHOD: {method}\n{'='*70}")

        if method == "verbalized":
            scorer = HypothesisScorerV2(llm, plausibility_mode="verbalized")
        else:
            scorer = HypothesisScorer(llm, plausibility_mode=method)

        repeat_reports = []

        for repeat in range(N_REPEATS):
            print(f"\n-- {method}  repeat {repeat+1}/{N_REPEATS} --")

            if method == "confidence_mc":
                all_scores = score_batch_single_prompt(
                    scorer, questions, all_hypotheses, constraints_list, atomics_list,
                    sp_logprob, scorer._parse_mc_confidence, "confidence_mc")
            elif method == "generation_sampling":
                all_scores = score_batch_single_prompt(
                    scorer, questions, all_hypotheses, constraints_list, atomics_list,
                    sp_sampling_n, scorer._parse_mc_sampling, "generation_sampling")
            elif method == "independent":
                all_scores = score_batch_independent(
                    scorer, questions, all_hypotheses, constraints_list, atomics_list, sp_logprob)
            elif method == "verbalized":
                all_scores = score_batch_verbalized(
                    scorer, questions, all_hypotheses, constraints_list, atomics_list, sp_verbalized)

            log_path = RESULTS_DIR / f"run_{method}_no_decomp_repeat{repeat}.jsonl"

            run_results: list[dict] = []
            with RunLogger(log_path, mode="w") as logger:
                for i, d in enumerate(records):
                    opts, gold = d["options"], d["answer"]
                    c, a = constraints_list[i], atomics_list[i]
                    sc = all_scores[i]
                    ilp_out = select_answer(opts, a, c, sc)
                    preds = {"hard_gate": ilp_out["prediction"]}
                    option_scores = {"hard_gate": ilp_out["option_scores"]}
                    record = {
                        "question_id": d["question_id"], "qa_type": d["qa_type"], "question": d["question"],
                        "gold": gold, "constraints": c, "atomics": a, "hypotheses": all_hypotheses[i],
                        "atomic_scores": sc, "options": opts, "preds": preds, "option_scores": option_scores,
                        "plausibility_mode": method,
                        "repeat": repeat, "sample_temperature": SAMPLE_TEMPERATURE,
                    }
                    logger.write(record)
                    run_results.append({"qa_type": d["qa_type"], "gold": gold, "preds": preds, "option_scores": option_scores})

            report = compute_metrics(run_results, "hard_gate")
            print_metrics(report, f"{method} repeat {repeat+1}")
            repeat_reports.append(report)

        all_repeat_results[method] = repeat_reports

    cleanup_llm(llm)

    print(f"\n\n{'='*80}\n  REPRODUCIBILITY SUMMARY (mean & std across {N_REPEATS} repeats)\n{'='*80}")
    print(f"{'Method':<22} {'Operator':<10} {'Mean':>8} {'Std':>8} {'Var':>8} {'Min':>8} {'Max':>8}")
    print("-" * 80)

    summary_rows = []
    for method, reports in all_repeat_results.items():
        operators = list(reports[0].keys())
        for op in operators:
            accs = np.array([r[op]["accuracy"] for r in reports])
            mean = accs.mean()
            std = accs.std(ddof=1) if len(accs) > 1 else 0.0
            var = accs.var(ddof=1) if len(accs) > 1 else 0.0
            print(f"{method:<22} {op:<10} {mean:>8.3f} {std:>8.3f} {var:>8.4f} {accs.min():>8.3f} {accs.max():>8.3f}")
            summary_rows.append({
                "method": method, "operator": op, "mean": float(mean), "std": float(std),
                "var": float(var), "min": float(accs.min()), "max": float(accs.max()),
                "all_runs": accs.tolist(),
            })

    with open(RESULTS_DIR / "reproducibility_summary.json", "w") as f:
        json.dump(summary_rows, f, indent=2)
    print(f"\nsaved {RESULTS_DIR / 'reproducibility_summary.json'}")


if __name__ == "__main__":
    run()
