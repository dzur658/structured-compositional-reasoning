import itertools
import json
import os
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
MAX_MODEL_LEN = 8192

HF_DATASET = "ojayy/logical-sata"
SPLIT = "validation"

VALID_QA_TYPES = ["AND", "OR", "NEITHER", "Mixed"]
SAMPLE_QA_TYPES = ["AND", "OR", "NEITHER", "Mixed"]
SAMPLES_PER_OPERATOR = 250
SAMPLE_OFFSET = 0

RESULTS_DIR = Path("./results")
RESULTS_DIR.mkdir(exist_ok=True)

PLAUSIBILITY_MODE = "confidence_mc"

USE_PARAGRAPH_STAGES: dict[str, bool] = {
    "constraint": True,
    "hypothesis": True,
    "scoring": True,
}

_OPTION_KEYS = ["A", "B", "C", "D"]
LABEL_MAP = {0: "A", 1: "B", 2: "C", 3: "D"}


@dataclass(frozen=True)
class LsataExample:
    question_id: str
    question: str
    paragraph: str
    choices: list[str]
    label: int
    qa_type: str


def clean_text(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s*Multi Label Question:\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def generate_and_combinations(correct, incorrect):
    valid = [f"{c1} AND {c2}" for c1, c2 in itertools.combinations(correct, 2)]
    invalid = [f"{c} AND {i}" for c in correct for i in incorrect] + \
              [f"{i1} AND {i2}" for i1, i2 in itertools.combinations(incorrect, 2)]
    return {"correct": valid, "incorrect": invalid}


def generate_or_combinations(correct, incorrect):
    valid = [f"{c1} OR {c2}" for c1, c2 in itertools.combinations(correct, 2)]
    valid += [f"{c} OR {i}" for c in correct for i in incorrect]
    invalid = [f"{i1} OR {i2}" for i1, i2 in itertools.combinations(incorrect, 2)]
    return {"correct": valid, "incorrect": invalid}


def generate_neither_combinations(correct, incorrect):
    valid = [f"NEITHER {i1} NOR {i2}" for i1, i2 in itertools.combinations(incorrect, 2)]
    invalid = [f"NEITHER {c1} NOR {c2}" for c1, c2 in itertools.combinations(correct, 2)]
    invalid += [f"NEITHER {c} NOR {i}" for c in correct for i in incorrect]
    return {"correct": valid, "incorrect": invalid}


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


def example_to_record_lsata(ex: LsataExample) -> dict[str, Any]:
    options = {key: _parse_choice(c) for key, c in zip(_OPTION_KEYS, ex.choices)}
    return {
        "question_id": ex.question_id,
        "question": ex.question,
        "paragraph": ex.paragraph,
        "options": options,
        "answer": _OPTION_KEYS[ex.label],
        "qa_type": ex.qa_type,
    }


def load_lsata_examples(split: str = SPLIT) -> list[LsataExample]:
    ds = load_dataset(HF_DATASET, split=split)
    examples: list[LsataExample] = []
    for idx, item in enumerate(ds):
        examples.append(
            LsataExample(
                question_id=str(item.get("idx", idx)),
                question=item["question"],
                paragraph=item["paragraph"],
                choices=item["choices"],
                label=int(item["label"]),
                qa_type=item["qa_type"],
            )
        )
    return examples


def _bucket_by_qa_type(examples: list[LsataExample]) -> dict[str, list[LsataExample]]:
    by_type: dict[str, list[LsataExample]] = {qa_type: [] for qa_type in SAMPLE_QA_TYPES}
    for example in examples:
        if example.qa_type in SAMPLE_QA_TYPES:
            by_type[example.qa_type].append(example)
    return by_type


def sample_stratified(
    examples: list[LsataExample],
    *,
    samples_per_operator: int | None = None,
    sample_offset: int | None = None,
) -> list[LsataExample]:
    n = samples_per_operator if samples_per_operator is not None else SAMPLES_PER_OPERATOR
    offset = sample_offset if sample_offset is not None else SAMPLE_OFFSET
    if n < 0 or offset < 0:
        raise ValueError(f"samples_per_operator and sample_offset must be non-negative, got N={n}, offset={offset}")

    by_type = _bucket_by_qa_type(examples)
    sampled: list[LsataExample] = []
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


def extract_atomics_from_record(options: dict) -> list[str]:
    return extract_atomics(options)


# constraint decomposition
_PROMPT_LSATA = """
Decompose the question into ALL of its individual constraints. Identify
each atomic constraint separately and state that ALL must hold
simultaneously.

All listed constraints MUST hold simultaneously.

A constraint is a required property of a valid answer, grounded in what
the passage explicitly states or clearly implies -- not in general world
knowledge.

Rules:
1. Break the question down into its distinct atomic conditions -- do NOT
   treat it as one combined idea if it genuinely contains separable
   requirements.
2. Do NOT simply restate or paraphrase the question as a whole. Each
   constraint must isolate ONE specific required condition.
3. Phrase every constraint as a requirement on the answer: "the answer
   must be / must have / must satisfy ...".
4. Where the question implies a strict, emphatic, or superlative
   requirement (e.g. "worst", "often", "always", "first", "original"),
   encode that emphasis directly in the relevant constraint -- do not
   soften it into a generic restatement.
5. Every constraint must be verifiable against the passage -- do NOT
   generate constraints that rely on general world knowledge or
   plausibility outside what the passage provides.
6. Do NOT generate constraints that can only be satisfied by already
   knowing the answer. The constraint must describe a PROPERTY a valid
   answer must have, not point to the answer itself.
7. Output valid JSON only.

Return exactly this JSON format:

For one constraint:
{"constraints": "One constraint must hold: (1) the answer must ..."}

For multiple constraints:
{"constraints": "Two constraints must hold simultaneously: (1) the answer must ..., and (2) the answer must ..."}

If there are three or more constraints:
{"constraints": "Three constraints must hold simultaneously: (1) the answer must ..., (2) the answer must ..., and (3) the answer must ..."}

Examples:

Example 1:
Passage:
A flood occurs when a river overflows its banks, often because of heavy
rain. In flat regions, the water spreads out, slows down, and drops the
sediment it was carrying. Over time this builds up a thick, dark layer
that is unusually good for growing crops, which is why farmers in places
like the Nile valley have relied on flooding rivers for centuries.

Question:
What quality makes floodplain land valuable for farming?

Output:
{"constraints": "Two constraints must hold simultaneously: (1) the answer must be a quality or property that the passage attributes to floodplain soil, and (2) that quality must be the specific reason the passage gives for why the land supports strong crop growth, not merely a description of the flooding process itself."}

Example 2:
Passage:
Nearly a year after Sept. 11, the Legal Aid Society remains homeless. The
nonprofit has been barred from returning to its 90 Church St. headquarters,
across from the World Trade Center site, because of environmental
concerns. Legal Aid's 450 displaced attorneys and staffers have spent the
past 12 months spread among previously unused spaces.

Question:
Why can't the Legal Aid Society return to its original headquarters?

Output:
{"constraints": "Two constraints must hold simultaneously: (1) the answer must identify the reason for the continued displacement as the passage explains it, and (2) that reason must connect to the specific circumstance the passage describes -- the environmental concern tied to the building's proximity to the World Trade Center site -- rather than a generic or unrelated obstacle."}

Example 3:
Passage:
The Spanair MD82 jet crashed last week at Madrid's airport as the plane
was trying to take off, killing 154 people. The aircraft managed to rise
only slightly before coming down quickly to the right of the runway, its
tail section hitting the ground first. Then the out-of-control plane
skidded and bounced at least three times as it careered 1,200 meters
across uneven terrain and exploded.

Question:
How was Pilar Gonzalez Ferreira killed?

Output:
{"constraints": "Two constraints must hold simultaneously: (1) the answer must describe the specific manner or event through which the death occurred, as explicitly stated or directly implied by the passage, and (2) the answer must reflect the actual cause described in the passage -- not a general category or inference beyond what the passage supports."}

Now generate constraints.

Passage:
{{paragraph}}

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
        self._template = _PROMPT_LSATA

    def _build_prompt(self, question: str, paragraph: str = "") -> str:
        p = paragraph if USE_PARAGRAPH_STAGES["constraint"] else ""
        return render_prompt(self._template, question=question, paragraph=p) + "\nOutput:"

    def _parse(self, text: str) -> list[str]:
        return _split_constraints(str(extract_json_object(text)["constraints"]))

    def generate(self, question: str, paragraph: str = "") -> list[str]:
        outputs = self.llm.generate([self._build_prompt(question, paragraph)], self.sampling_params)
        return self._parse(outputs[0].outputs[0].text)

    def generate_batch(
        self,
        questions: list[str],
        paragraphs: list[str] | None = None,
    ) -> list[list[str]]:
        if paragraphs is None:
            paragraphs = [""] * len(questions)
        prompts = [self._build_prompt(q, p) for q, p in zip(questions, paragraphs)]
        outputs = self.llm.generate(prompts, self.sampling_params)
        results = []
        for out in outputs:
            try:
                results.append(self._parse(out.outputs[0].text))
            except Exception:
                results.append([])
        return results

# hypothesis generation
HYP_PROMPT_LSATA = """
You create two hypothesis statements for an atomic answer against one
constraint, grounded in a reading comprehension passage.

Given:
- Passage
- Question
- Atomic statement
- Constraint

Create:
H+ = a positive hypothesis saying the atomic satisfies the constraint
H- = a negative hypothesis saying the atomic does not satisfy the constraint

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
Passage:
A flood occurs when a river overflows its banks, often because of heavy
rain. In flat regions, the water spreads out, slows down, and drops the
sediment it was carrying. Over time this builds up a thick, dark layer
that is unusually good for growing crops, which is why farmers in places
like the Nile valley have relied on flooding rivers for centuries.

Question:
What quality makes floodplain land valuable for farming?

Atomic statement:
fertility

Constraint:
The answer must be a quality or property attributed to floodplain soil,
and must be the specific reason the land supports strong crop growth.

Output:
{
  "H+": "Fertility satisfies the requirement of being a quality or property attributed to floodplain soil, and being the specific reason the land supports strong crop growth, because it is in the passage.",
  "H-": "Fertility does not satisfy the requirement of being a quality or property attributed to floodplain soil, and being the specific reason the land supports strong crop growth, because it is not in the passage."
}

Example 2:
Passage:
Nearly a year after Sept. 11, the Legal Aid Society remains homeless. The
nonprofit has been barred from returning to its 90 Church St. headquarters,
across from the World Trade Center site, because of environmental
concerns. Legal Aid's 450 displaced attorneys and staffers have spent the
past 12 months spread among previously unused spaces.

Question:
Why can't the Legal Aid Society return to its original headquarters?

Atomic statement:
contamination risk from the nearby World Trade Center site

Constraint:
The answer must identify the reason for the continued displacement,
connected to the specific circumstance described.

Output:
{
  "H+": "Contamination risk from the nearby World Trade Center site satisfies the requirement of identifying the reason for the continued displacement, connected to the specific circumstance described, because it is in the passage.",
  "H-": "Contamination risk from the nearby World Trade Center site does not satisfy the requirement of identifying the reason for the continued displacement, connected to the specific circumstance described, because it is not in the passage."
}

Now generate hypotheses.

Passage:
{{paragraph}}

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
        self._template = HYP_PROMPT_LSATA

    def _build_prompt(self, question: str, atomic: str, constraint: str, paragraph: str = "") -> str:
        p = paragraph if USE_PARAGRAPH_STAGES["hypothesis"] else ""
        return render_prompt(
            self._template, question=question, atomic=atomic, constraint=constraint, paragraph=p,
        ) + "\nOutput:"

    def generate_batch(
        self,
        examples: list[tuple[str, list[str], list[str]]],
        paragraphs: list[str] | None = None,
    ) -> list[dict]:
        """One llm.generate call for every (question, atomic, constraint) triplet
        across all examples. Returns one result dict per example."""
        if paragraphs is None:
            paragraphs = [""] * len(examples)
        index: list[tuple[int, str, str]] = []
        prompts: list[str] = []
        for ex_idx, ((q, atomics, constraints), para) in enumerate(zip(examples, paragraphs)):
            for a in atomics:
                for c in constraints:
                    index.append((ex_idx, a, c))
                    prompts.append(self._build_prompt(q, a, c, para))

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

PLAUSIBILITY_PROMPT_CONFIDENCE_MC_LSATA = """
Decide which of two competing hypotheses about an atomic answer statement
is correct, with respect to a required reading comprehension constraint.

You are given:
1. A passage.
2. A question about the passage.
3. A constraint describing what a valid answer must do, grounded in the
   passage.
4. One atomic answer statement.
5. Two hypotheses about that atomic statement:
   Option A: the atomic SATISFIES the constraint.
   Option B: the atomic FAILS the constraint.

Important rules:
- Judge the atomic against the constraint and the passage, not against
  other answer options.
- Do not reward surface word overlap between the atomic and the passage
  alone -- the semantic relationship must hold.
- Do not penalize an atomic merely because it is not the most obvious
  phrasing -- judge whether THIS atomic genuinely satisfies the required
  relation as described in the passage.
- Preserve important roles in the constraint, such as "original",
  "specific", "explicit", "first", "directly stated".
- This is a forced choice: you must commit to exactly one letter, A or B,
  even in borderline cases.
- Output ONLY the single letter A or B. Do not output any other words,
  punctuation, explanation, or reasoning of any kind.

Examples:

Example 1:
Passage:
A flood occurs when a river overflows its banks, often because of heavy
rain. In flat regions, the water spreads out, slows down, and drops the
sediment it was carrying. Over time this builds up a thick, dark layer
that is unusually good for growing crops, which is why farmers in places
like the Nile valley have relied on flooding rivers for centuries.

Question:
What quality makes floodplain land valuable for farming?

Constraint:
The answer must be a quality or property that the passage attributes to
floodplain soil, and must be the specific reason the passage gives for
why the land supports strong crop growth.

Atomic statement:
fertility

Option A:
The atomic statement satisfies the constraint.

Option B:
The atomic statement fails the constraint.

Answer: A

Example 2:
Passage:
A flood occurs when a river overflows its banks, often because of heavy
rain. In flat regions, the water spreads out, slows down, and drops the
sediment it was carrying. Over time this builds up a thick, dark layer
that is unusually good for growing crops, which is why farmers in places
like the Nile valley have relied on flooding rivers for centuries.

Question:
What quality makes floodplain land valuable for farming?

Constraint:
The answer must be a quality or property that the passage attributes to
floodplain soil, and must be the specific reason the passage gives for
why the land supports strong crop growth.

Atomic statement:
elevation

Option A:
The atomic statement satisfies the constraint.

Option B:
The atomic statement fails the constraint.

Answer: B

Example 3:
Passage:
Nearly a year after Sept. 11, the Legal Aid Society remains homeless. The
nonprofit has been barred from returning to its 90 Church St. headquarters,
across from the World Trade Center site, because of environmental
concerns.

Question:
Why can't the Legal Aid Society return to its original headquarters?

Constraint:
The answer must identify the reason for the continued displacement as the
passage explains it, connected to the specific circumstance the passage
describes.

Atomic statement:
contamination risk from the nearby World Trade Center site

Option A:
The atomic statement satisfies the constraint.

Option B:
The atomic statement fails the constraint.

Answer: A

Example 4:
Passage:
Nearly a year after Sept. 11, the Legal Aid Society remains homeless. The
nonprofit has been barred from returning to its 90 Church St. headquarters,
across from the World Trade Center site, because of environmental
concerns.

Question:
Why can't the Legal Aid Society return to its original headquarters?

Constraint:
The answer must identify the reason for the continued displacement as the
passage explains it, connected to the specific circumstance the passage
describes.

Atomic statement:
a lack of available office space

Option A:
The atomic statement satisfies the constraint.

Option B:
The atomic statement fails the constraint.

Answer: B

Now judge the given case. Output ONLY a single letter, A or B -- nothing
else.

Passage:
{{paragraph}}

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


PLAUSIBILITY_PROMPT_TRUEFALSE_PLUS_LSATA = """
Decide whether a hypothesis about an atomic answer statement is TRUE, with
respect to a required reading comprehension constraint.

You are given:
1. A passage.
2. A question about the passage.
3. A constraint describing what a valid answer must do, grounded in the
   passage.
4. One atomic answer statement.
5. A hypothesis: the atomic statement SATISFIES the constraint.

Important rules:
- Judge the atomic against the constraint and the passage on its own terms.
- Do not reward surface word overlap between the atomic and the passage
  alone -- the semantic relationship must hold.
- Do not penalize an atomic merely because it is not the most obvious
  phrasing -- judge whether THIS atomic genuinely satisfies the required
  relation as described in the passage.
- Preserve important roles in the constraint, such as "original",
  "specific", "explicit", "first", "directly stated".
- Output ONLY the single word True or False. Do not output any other
  words, punctuation, explanation, or reasoning of any kind.

Examples:

Example 1:
Passage:
A flood occurs when a river overflows its banks, often because of heavy
rain. In flat regions, the water spreads out, slows down, and drops the
sediment it was carrying. Over time this builds up a thick, dark layer
that is unusually good for growing crops, which is why farmers in places
like the Nile valley have relied on flooding rivers for centuries.

Question:
What quality makes floodplain land valuable for farming?

Constraint:
The answer must be a quality or property that the passage attributes to
floodplain soil, and must be the specific reason the passage gives for
why the land supports strong crop growth.

Atomic statement:
fertility

Hypothesis: the atomic statement satisfies the constraint.

Answer: True

Example 2:
Passage:
A flood occurs when a river overflows its banks, often because of heavy
rain. In flat regions, the water spreads out, slows down, and drops the
sediment it was carrying. Over time this builds up a thick, dark layer
that is unusually good for growing crops, which is why farmers in places
like the Nile valley have relied on flooding rivers for centuries.

Question:
What quality makes floodplain land valuable for farming?

Constraint:
The answer must be a quality or property that the passage attributes to
floodplain soil, and must be the specific reason the passage gives for
why the land supports strong crop growth.

Atomic statement:
elevation

Hypothesis: the atomic statement satisfies the constraint.

Answer: False

Example 3:
Passage:
Nearly a year after Sept. 11, the Legal Aid Society remains homeless. The
nonprofit has been barred from returning to its 90 Church St. headquarters,
across from the World Trade Center site, because of environmental
concerns.

Question:
Why can't the Legal Aid Society return to its original headquarters?

Constraint:
The answer must identify the reason for the continued displacement as the
passage explains it, connected to the specific circumstance the passage
describes.

Atomic statement:
contamination risk from the nearby World Trade Center site

Hypothesis: the atomic statement satisfies the constraint.

Answer: True

Example 4:
Passage:
Nearly a year after Sept. 11, the Legal Aid Society remains homeless. The
nonprofit has been barred from returning to its 90 Church St. headquarters,
across from the World Trade Center site, because of environmental
concerns.

Question:
Why can't the Legal Aid Society return to its original headquarters?

Constraint:
The answer must identify the reason for the continued displacement as the
passage explains it, connected to the specific circumstance the passage
describes.

Atomic statement:
a lack of available office space

Hypothesis: the atomic statement satisfies the constraint.

Answer: False


Now judge the given case. Output ONLY True or False -- nothing else.

Passage:
{{paragraph}}

Question:
{{question}}

Constraint:
{{constraint}}

Atomic statement:
{{atomic_statement}}

Hypothesis: the atomic statement satisfies the constraint.
{{H_plus}}

Answer:"""


PLAUSIBILITY_PROMPT_TRUEFALSE_MINUS_LSATA = """
Decide whether a hypothesis about an atomic answer statement is TRUE, with
respect to a required reading comprehension constraint.

You are given:
1. A passage.
2. A question about the passage.
3. A constraint describing what a valid answer must do, grounded in the
   passage.
4. One atomic answer statement.
5. A hypothesis: the atomic statement FAILS the constraint.

Important rules:
- Judge the atomic against the constraint and the passage on its own terms.
- Do not answer True (i.e. "fails") merely because the atomic is not the
  most obvious or canonical phrasing found in the passage.
- Do not answer False (i.e. "does not fail") merely because the atomic
  appears somewhere in the passage -- appearing in the passage does not
  mean the atomic satisfies the specific constraint.
- Preserve important roles in the constraint, such as "original",
  "specific", "explicit", "first", "directly stated".
- Output ONLY the single word True or False. Do not output any other
  words, punctuation, explanation, or reasoning of any kind.

Examples:

Example 1:
Passage:
A flood occurs when a river overflows its banks, often because of heavy
rain. In flat regions, the water spreads out, slows down, and drops the
sediment it was carrying. Over time this builds up a thick, dark layer
that is unusually good for growing crops, which is why farmers in places
like the Nile valley have relied on flooding rivers for centuries.

Question:
What quality makes floodplain land valuable for farming?

Constraint:
The answer must be a quality or property that the passage attributes to
floodplain soil, and must be the specific reason the passage gives for
why the land supports strong crop growth.

Atomic statement:
fertility

Hypothesis: the atomic statement fails the constraint.

Answer: False

Example 2:
Passage:
A flood occurs when a river overflows its banks, often because of heavy
rain. In flat regions, the water spreads out, slows down, and drops the
sediment it was carrying. Over time this builds up a thick, dark layer
that is unusually good for growing crops, which is why farmers in places
like the Nile valley have relied on flooding rivers for centuries.

Question:
What quality makes floodplain land valuable for farming?

Constraint:
The answer must be a quality or property that the passage attributes to
floodplain soil, and must be the specific reason the passage gives for
why the land supports strong crop growth.

Atomic statement:
elevation

Hypothesis: the atomic statement fails the constraint.

Answer: True

Example 3:
Passage:
Nearly a year after Sept. 11, the Legal Aid Society remains homeless. The
nonprofit has been barred from returning to its 90 Church St. headquarters,
across from the World Trade Center site, because of environmental
concerns.

Question:
Why can't the Legal Aid Society return to its original headquarters?

Constraint:
The answer must identify the reason for the continued displacement as the
passage explains it, connected to the specific circumstance the passage
describes.

Atomic statement:
contamination risk from the nearby World Trade Center site

Hypothesis: the atomic statement fails the constraint.

Answer: False

Example 4:
Passage:
Nearly a year after Sept. 11, the Legal Aid Society remains homeless. The
nonprofit has been barred from returning to its 90 Church St. headquarters,
across from the World Trade Center site, because of environmental
concerns.

Question:
Why can't the Legal Aid Society return to its original headquarters?

Constraint:
The answer must identify the reason for the continued displacement as the
passage explains it, connected to the specific circumstance the passage
describes.

Atomic statement:
a lack of available office space

Hypothesis: the atomic statement fails the constraint.

Answer: True

Now judge the given case. Output ONLY True or False -- nothing else.

Passage:
{{paragraph}}

Question:
{{question}}

Constraint:
{{constraint}}

Atomic statement:
{{atomic_statement}}

Hypothesis: the atomic statement fails the constraint.
{{H_minus}}

Answer:"""


PLAUSIBILITY_PROMPT_TRUEFALSE_LSATA = (
    PLAUSIBILITY_PROMPT_TRUEFALSE_PLUS_LSATA,
    PLAUSIBILITY_PROMPT_TRUEFALSE_MINUS_LSATA,
)

PLAUSIBILITY_PROMPTS: dict[str, object] = {
    "confidence_mc": PLAUSIBILITY_PROMPT_CONFIDENCE_MC_LSATA,
    "independent": PLAUSIBILITY_PROMPT_TRUEFALSE_LSATA,
    "generation_sampling": PLAUSIBILITY_PROMPT_CONFIDENCE_MC_LSATA,
}


# verbalized confidence prompts
VERBALIZED_PROMPT_PLUS_LSATA = """
State your confidence that a hypothesis about an atomic answer statement is
TRUE, with respect to a required reading comprehension constraint.

You are given:
1. A passage.
2. A question about the passage.
3. A constraint describing what a valid answer must do, grounded in the
   passage.
4. One atomic answer statement.
5. A hypothesis: the atomic statement SATISFIES the constraint.

Important rules:
- Judge the atomic against the constraint and the passage on its own terms.
- Do not reward surface word overlap between the atomic and the passage
  alone -- the semantic relationship must hold.
- Do not penalize an atomic merely because it is not the most obvious
  phrasing -- judge whether THIS atomic genuinely satisfies the required
  relation as described in the passage.
- Preserve important roles in the constraint, such as "original",
  "specific", "explicit", "first", "directly stated".
- Output ONLY a single integer from 0 to 10, where 0 means you are
  completely confident the hypothesis is FALSE and 10 means you are
  completely confident the hypothesis is TRUE. Do not output any other
  words, punctuation, explanation, or reasoning of any kind.

Examples:

Example 1:
Passage:
A flood occurs when a river overflows its banks, often because of heavy
rain. In flat regions, the water spreads out, slows down, and drops the
sediment it was carrying. Over time this builds up a thick, dark layer
that is unusually good for growing crops, which is why farmers in places
like the Nile valley have relied on flooding rivers for centuries.

Question:
What quality makes floodplain land valuable for farming?

Constraint:
The answer must be a quality or property that the passage attributes to
floodplain soil, and must be the specific reason the passage gives for
why the land supports strong crop growth.

Atomic statement:
fertility

Hypothesis: the atomic statement satisfies the constraint.

Confidence: 9

Example 2:
Passage:
A flood occurs when a river overflows its banks, often because of heavy
rain. In flat regions, the water spreads out, slows down, and drops the
sediment it was carrying. Over time this builds up a thick, dark layer
that is unusually good for growing crops, which is why farmers in places
like the Nile valley have relied on flooding rivers for centuries.

Question:
What quality makes floodplain land valuable for farming?

Constraint:
The answer must be a quality or property that the passage attributes to
floodplain soil, and must be the specific reason the passage gives for
why the land supports strong crop growth.

Atomic statement:
elevation

Hypothesis: the atomic statement satisfies the constraint.

Confidence: 1

Example 3:
Passage:
Nearly a year after Sept. 11, the Legal Aid Society remains homeless. The
nonprofit has been barred from returning to its 90 Church St. headquarters,
across from the World Trade Center site, because of environmental
concerns.

Question:
Why can't the Legal Aid Society return to its original headquarters?

Constraint:
The answer must identify the reason for the continued displacement as the
passage explains it, connected to the specific circumstance the passage
describes.

Atomic statement:
contamination risk from the nearby World Trade Center site

Hypothesis: the atomic statement satisfies the constraint.

Confidence: 9

Example 4:
Passage:
Nearly a year after Sept. 11, the Legal Aid Society remains homeless. The
nonprofit has been barred from returning to its 90 Church St. headquarters,
across from the World Trade Center site, because of environmental
concerns.

Question:
Why can't the Legal Aid Society return to its original headquarters?

Constraint:
The answer must identify the reason for the continued displacement as the
passage explains it, connected to the specific circumstance the passage
describes.

Atomic statement:
a lack of available office space

Hypothesis: the atomic statement satisfies the constraint.

Confidence: 1

Example 5:
Passage:
The Spanair MD82 jet crashed last week at Madrid's airport as the plane
was trying to take off, killing 154 people. The out-of-control plane
skidded and bounced at least three times as it careered 1,200 meters
across uneven terrain and exploded.

Question:
How was Pilar Gonzalez Ferreira killed?

Constraint:
The answer must describe the specific manner or event through which the
death occurred, as explicitly stated or directly implied by the passage.

Atomic statement:
plane crash

Hypothesis: the atomic statement satisfies the constraint.

Confidence: 9

Example 6:
Passage:
The Spanair MD82 jet crashed last week at Madrid's airport as the plane
was trying to take off, killing 154 people. The out-of-control plane
skidded and bounced at least three times as it careered 1,200 meters
across uneven terrain and exploded.

Question:
How was Pilar Gonzalez Ferreira killed?

Constraint:
The answer must describe the specific manner or event through which the
death occurred, as explicitly stated or directly implied by the passage.

Atomic statement:
road crash

Hypothesis: the atomic statement satisfies the constraint.

Confidence: 1

Now judge the given case. Output ONLY a single integer from 0 to 10 --
nothing else.

Passage:
{{paragraph}}

Question:
{{question}}

Constraint:
{{constraint}}

Atomic statement:
{{atomic_statement}}

Hypothesis: the atomic statement satisfies the constraint.
{{H_plus}}

Confidence:"""


VERBALIZED_PROMPT_MINUS_LSATA = """
State your confidence that a hypothesis about an atomic answer statement is
TRUE, with respect to a required reading comprehension constraint.

You are given:
1. A passage.
2. A question about the passage.
3. A constraint describing what a valid answer must do, grounded in the
   passage.
4. One atomic answer statement.
5. A hypothesis: the atomic statement FAILS the constraint.

Important rules:
- Judge the atomic against the constraint and the passage on its own terms.
- Do not give a high confidence (i.e. "fails") merely because the atomic is
  not the most obvious phrasing.
- Do not give a low confidence (i.e. "does not fail") merely because the
  atomic is mentioned somewhere in the passage -- a mentioned, related
  atomic can still genuinely fail the required relation.
- Preserve important roles in the constraint, such as "original",
  "specific", "explicit", "first", "directly stated".
- Output ONLY a single integer from 0 to 10, where 0 means you are
  completely confident the hypothesis is FALSE (the atomic does NOT fail
  the constraint) and 10 means you are completely confident the hypothesis
  is TRUE (the atomic DOES fail the constraint). Do not output any other
  words, punctuation, explanation, or reasoning of any kind.

Examples:

Example 1:
Passage:
A flood occurs when a river overflows its banks, often because of heavy
rain. In flat regions, the water spreads out, slows down, and drops the
sediment it was carrying. Over time this builds up a thick, dark layer
that is unusually good for growing crops, which is why farmers in places
like the Nile valley have relied on flooding rivers for centuries.

Question:
What quality makes floodplain land valuable for farming?

Constraint:
The answer must be a quality or property that the passage attributes to
floodplain soil, and must be the specific reason the passage gives for
why the land supports strong crop growth.

Atomic statement:
fertility

Hypothesis: the atomic statement fails the constraint.

Confidence: 1

Example 2:
Passage:
A flood occurs when a river overflows its banks, often because of heavy
rain. In flat regions, the water spreads out, slows down, and drops the
sediment it was carrying. Over time this builds up a thick, dark layer
that is unusually good for growing crops, which is why farmers in places
like the Nile valley have relied on flooding rivers for centuries.

Question:
What quality makes floodplain land valuable for farming?

Constraint:
The answer must be a quality or property that the passage attributes to
floodplain soil, and must be the specific reason the passage gives for
why the land supports strong crop growth.

Atomic statement:
elevation

Hypothesis: the atomic statement fails the constraint.

Confidence: 9

Example 3:
Passage:
Nearly a year after Sept. 11, the Legal Aid Society remains homeless. The
nonprofit has been barred from returning to its 90 Church St. headquarters,
across from the World Trade Center site, because of environmental
concerns.

Question:
Why can't the Legal Aid Society return to its original headquarters?

Constraint:
The answer must identify the reason for the continued displacement as the
passage explains it, connected to the specific circumstance the passage
describes.

Atomic statement:
contamination risk from the nearby World Trade Center site

Hypothesis: the atomic statement fails the constraint.

Confidence: 1

Example 4:
Passage:
Nearly a year after Sept. 11, the Legal Aid Society remains homeless. The
nonprofit has been barred from returning to its 90 Church St. headquarters,
across from the World Trade Center site, because of environmental
concerns.

Question:
Why can't the Legal Aid Society return to its original headquarters?

Constraint:
The answer must identify the reason for the continued displacement as the
passage explains it, connected to the specific circumstance the passage
describes.

Atomic statement:
a lack of available office space

Hypothesis: the atomic statement fails the constraint.

Confidence: 9

Example 5:
Passage:
The Spanair MD82 jet crashed last week at Madrid's airport as the plane
was trying to take off, killing 154 people. The out-of-control plane
skidded and bounced at least three times as it careered 1,200 meters
across uneven terrain and exploded.

Question:
How was Pilar Gonzalez Ferreira killed?

Constraint:
The answer must describe the specific manner or event through which the
death occurred, as explicitly stated or directly implied by the passage.

Atomic statement:
plane crash

Hypothesis: the atomic statement fails the constraint.

Confidence: 1

Example 6:
Passage:
The Spanair MD82 jet crashed last week at Madrid's airport as the plane
was trying to take off, killing 154 people. The out-of-control plane
skidded and bounced at least three times as it careered 1,200 meters
across uneven terrain and exploded.

Question:
How was Pilar Gonzalez Ferreira killed?

Constraint:
The answer must describe the specific manner or event through which the
death occurred, as explicitly stated or directly implied by the passage.

Atomic statement:
road crash

Hypothesis: the atomic statement fails the constraint.

Confidence: 9

Now judge the given case. Output ONLY a single integer from 0 to 10 --
nothing else.

Passage:
{{paragraph}}

Question:
{{question}}

Constraint:
{{constraint}}

Atomic statement:
{{atomic_statement}}

Hypothesis: the atomic statement fails the constraint.
{{H_minus}}

Confidence:"""


VERBALIZED_PROMPT_TUPLE_LSATA = (VERBALIZED_PROMPT_PLUS_LSATA, VERBALIZED_PROMPT_MINUS_LSATA)

PLAUSIBILITY_PROMPTS["verbalized"] = VERBALIZED_PROMPT_TUPLE_LSATA

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
        self.sampling_n_params = SamplingParams(temperature=0.7, max_tokens=1, n=10)

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

    def _build_plausibility_prompts(
        self, question: str, pairs: list[tuple[str, str, dict]], paragraph: str = ""
    ):
        mode = self.plausibility_mode
        para = paragraph if USE_PARAGRAPH_STAGES["scoring"] else ""
        jobs: list[tuple] = []
        prompts: list[str] = []

        if mode == "independent":
            plus_tpl, minus_tpl = PLAUSIBILITY_PROMPTS["independent"]
            for a, c, h in pairs:
                prompts.append(render_prompt(
                    plus_tpl, question=question, atomic_statement=a,
                    constraint=c, paragraph=para, H_plus=h.get("H+", ""),
                ))
                jobs.append((a, c, "plus"))
                prompts.append(render_prompt(
                    minus_tpl, question=question, atomic_statement=a,
                    constraint=c, paragraph=para, H_minus=h.get("H-", ""),
                ))
                jobs.append((a, c, "minus"))
        elif mode in ("confidence_mc", "generation_sampling"):
            template = PLAUSIBILITY_PROMPTS["confidence_mc"]
            for a, c, h in pairs:
                prompts.append(render_prompt(
                    template, question=question, atomic_statement=a,
                    constraint=c, paragraph=para,
                    H_plus=h.get("H+", ""), H_minus=h.get("H-", ""),
                ))
                jobs.append((a, c, "single"))
        else:
            raise ValueError(f"_build_plausibility_prompts: unhandled mode {mode!r}")

        return jobs, prompts

    def score(self, question: str, hypotheses: dict, paragraph: str = "") -> dict:
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
            result.update(self._score_independent(question, pairs, paragraph))
        else:
            result.update(self._score_dependent(question, pairs, paragraph))

        return result

    def _score_dependent(
        self, question: str, pairs: list[tuple[str, str, dict]], paragraph: str = ""
    ) -> dict:
        mode = self.plausibility_mode
        jobs, prompts = self._build_plausibility_prompts(question, pairs, paragraph)

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

    def _score_independent(
        self, question: str, pairs: list[tuple[str, str, dict]], paragraph: str = ""
    ) -> dict:
        jobs, prompts = self._build_plausibility_prompts(question, pairs, paragraph)
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

    def _build_plausibility_prompts(self, question: str, pairs: list[tuple[str, str, dict]],
                                     paragraph: str = ""):
        if self.plausibility_mode != "verbalized":
            return super()._build_plausibility_prompts(question, pairs, paragraph)

        plus_tpl, minus_tpl = PLAUSIBILITY_PROMPTS["verbalized"]
        para = paragraph if USE_PARAGRAPH_STAGES["scoring"] else ""
        jobs, prompts = [], []
        for a, c, h in pairs:
            prompts.append(render_prompt(plus_tpl, question=question, atomic_statement=a,
                                         constraint=c, paragraph=para, H_plus=h.get("H+", "")))
            jobs.append((a, c, "plus"))
            prompts.append(render_prompt(minus_tpl, question=question, atomic_statement=a,
                                         constraint=c, paragraph=para, H_minus=h.get("H-", "")))
            jobs.append((a, c, "minus"))
        return jobs, prompts

    def score(self, question: str, hypotheses: dict, paragraph: str = "") -> dict:
        if self.plausibility_mode != "verbalized":
            return super().score(question, hypotheses, paragraph=paragraph)
        pairs = [(a, c, hypotheses[a][c]) for a, cm in hypotheses.items() for c in cm]
        result: dict = {a: {} for a in hypotheses}
        result.update(self._score_verbalized(question, pairs, paragraph))
        return result

    def _score_verbalized(self, question: str, pairs: list[tuple[str, str, dict]], paragraph: str = "") -> dict:
        plus_tpl, minus_tpl = PLAUSIBILITY_PROMPTS["verbalized"]
        para = paragraph if USE_PARAGRAPH_STAGES["scoring"] else ""

        prompts, jobs = [], []
        for a, c, h in pairs:
            prompts.append(render_prompt(plus_tpl, question=question, atomic_statement=a,
                                         constraint=c, paragraph=para, H_plus=h.get("H+", "")))
            jobs.append((a, c, "plus"))
            prompts.append(render_prompt(minus_tpl, question=question, atomic_statement=a,
                                         constraint=c, paragraph=para, H_minus=h.get("H-", "")))
            jobs.append((a, c, "minus"))

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
                               atomics_list, paragraphs, sp, parser_fn, mode_name):
    """confidence_mc / generation_sampling -- one prompt per (atomic, constraint) pair."""
    all_prompts, all_jobs = [], []
    for ex_idx, (q, h, cons, atoms, para) in enumerate(
        zip(questions, hypotheses_list, constraints_list, atomics_list, paragraphs)
    ):
        score_para = para if USE_PARAGRAPH_STAGES["scoring"] else ""
        pairs = [(a, c, h[a][c]) for a in atoms for c in cons if a in h and c in h[a]]
        jobs, prompts = scorer._build_plausibility_prompts(q, pairs, score_para)
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


def score_batch_independent(scorer, questions, hypotheses_list, constraints_list,
                             atomics_list, paragraphs, sp):
    """independent -- two separate prompts per pair, logprob-read, normalized."""
    all_prompts, all_jobs = [], []
    for ex_idx, (q, h, cons, atoms, para) in enumerate(
        zip(questions, hypotheses_list, constraints_list, atomics_list, paragraphs)
    ):
        score_para = para if USE_PARAGRAPH_STAGES["scoring"] else ""
        pairs = [(a, c, h[a][c]) for a in atoms for c in cons if a in h and c in h[a]]
        jobs, prompts = scorer._build_plausibility_prompts(q, pairs, score_para)
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
        denom = pt_plus + pt_minus
        p_plus, p_minus = (pt_plus / denom, 1.0 - pt_plus / denom) if denom > 0 else (0.0, 0.0)
        results[ex_idx].setdefault(a, {})[c] = {
            "fit_label": "", "p_plus": p_plus, "p_minus": p_minus,
            "active": "H+" if p_plus >= p_minus else "H-",
            "plausibility_reason": f"p_true(H+)={pt_plus:.4f}, p_true(H-)={pt_minus:.4f}",
            "mode": "independent",
            "T": p_plus * 5, "F": p_minus * 5,
        }
    return results


def score_batch_verbalized(scorer, questions, hypotheses_list, constraints_list,
                            atomics_list, paragraphs, sp):
    """verbalized -- two separate prompts per pair, free-text confidence, normalized."""
    all_prompts, all_jobs = [], []
    for ex_idx, (q, h, cons, atoms, para) in enumerate(
        zip(questions, hypotheses_list, constraints_list, atomics_list, paragraphs)
    ):
        score_para = para if USE_PARAGRAPH_STAGES["scoring"] else ""
        pairs = [(a, c, h[a][c]) for a in atoms for c in cons if a in h and c in h[a]]
        jobs, prompts = scorer._build_plausibility_prompts(q, pairs, score_para)
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
sp_sampling_n = SamplingParams(temperature=SAMPLE_TEMPERATURE, max_tokens=1, n=10)
sp_verbalized = SamplingParams(temperature=SAMPLE_TEMPERATURE, max_tokens=8)


def run() -> None:
    os.environ.setdefault("HF_TOKEN", os.environ.get("HF_TOKEN", ""))

    examples = load_lsata_examples()
    sampled = sample_stratified(examples)

    print(f"total examples: {len(examples)}, sampled: {len(sampled)}")
    for qt, n in Counter(ex.qa_type for ex in sampled).items():
        print(f"  {qt}: {n}")

    llm = LLM(**llm_engine_kwargs(MODEL, max_model_len=MAX_MODEL_LEN))

    records = [example_to_record_lsata(ex) for ex in sampled]
    atomics_list = [extract_atomics_from_record(d["options"]) for d in records]

    # no decomposition: raw question text is the sole constraint
    unique_pairs = list(dict.fromkeys((d["question"], d["paragraph"]) for d in records))
    constraints_cache = {pair: [pair[0]] for pair in unique_pairs}

    hypothesis_gen = HypothesisGenerator(llm)
    hyp_paragraphs = [
        d["paragraph"] if USE_PARAGRAPH_STAGES["hypothesis"] else ""
        for d in records
    ]
    all_hypotheses = hypothesis_gen.generate_batch(
        [
            (d["question"], atomics_list[i], constraints_cache[(d["question"], d["paragraph"])])
            for i, d in enumerate(records)
        ],
        paragraphs=hyp_paragraphs,
    )

    questions = [d["question"] for d in records]
    paragraphs = [d["paragraph"] for d in records]
    constraints_list = [constraints_cache[(d["question"], d["paragraph"])] for d in records]

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
                    scorer, questions, all_hypotheses, constraints_list, atomics_list, paragraphs,
                    sp_logprob, scorer._parse_mc_confidence, "confidence_mc")
            elif method == "generation_sampling":
                all_scores = score_batch_single_prompt(
                    scorer, questions, all_hypotheses, constraints_list, atomics_list, paragraphs,
                    sp_sampling_n, scorer._parse_mc_sampling, "generation_sampling")
            elif method == "independent":
                all_scores = score_batch_independent(
                    scorer, questions, all_hypotheses, constraints_list, atomics_list, paragraphs, sp_logprob)
            elif method == "verbalized":
                all_scores = score_batch_verbalized(
                    scorer, questions, all_hypotheses, constraints_list, atomics_list, paragraphs, sp_verbalized)

            log_path = RESULTS_DIR / f"run_{method}_no_decomp_repeat{repeat}.jsonl"

            run_results: list[dict] = []
            with RunLogger(log_path, mode="w") as logger:
                for i, d in enumerate(records):
                    opts, gold = d["options"], d["answer"]
                    c, a = constraints_list[i], atomics_list[i]
                    sc = all_scores[i]
                    if not c or not a:
                        continue
                    ilp_out = select_answer(opts, a, c, sc)
                    preds = {"hard_gate": ilp_out["prediction"]}
                    option_scores = {"hard_gate": ilp_out["option_scores"]}
                    record = {
                        "question_id": d["question_id"], "qa_type": d["qa_type"], "question": d["question"],
                        "paragraph": d["paragraph"], "gold": gold, "constraints": c, "atomics": a,
                        "hypotheses": all_hypotheses[i], "atomic_scores": sc, "options": opts,
                        "preds": preds, "option_scores": option_scores,
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
