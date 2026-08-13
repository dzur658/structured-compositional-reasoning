## From Atomic Evidence to Logical Composition

Code for "From Atomic Evidence to Logical Composition: Structured Compositional Reasoning over Compound Answer Options".

This repo has the evaluation code for both benchmarks: direct-prompting baselines (0 to 3 shot, zero-shot CoT) and the structured inference framework (contrastive hypothesis scoring, calibration, operator-constrained ILP) for LOGICAL-COMMONSENSEQA and LOGICAL-SATA.

Model used throughout is Llama-3.1-8B-Instruct.

### Layout

```
lcsqa/
  run_direct_prompting.py       0/1/2/3-shot direct letter prediction, 5 runs each, two eval slices per operator
  run_structured_inference.py   decomposition, contrastive hypothesis scoring, calibration, ILP composition

lsata/
  run_direct_prompting.py       0/1/2/3-shot direct prediction, passage grounded, 5 runs each
  run_zeroshot_cot.py           zero-shot chain-of-thought baseline, 5 runs
  run_structured_inference.py   same pipeline as lcsqa, adapted for paragraph grounding

utils/
  prompting.py       prompt rendering, JSON extraction from model output
  scoring.py         logprob and confidence-parsing helpers, atomic extraction
  vllm_utils.py       vLLM engine setup/teardown
  ilp.py             operator-constrained hard-gate ILP that picks the valid compound option
  logging_utils.py    jsonl run logger
  metrics.py         macro-F1 / hits@2 by operator
```

The structured-inference scripts run four confidence-elicitation methods: paired multiple choice, independent True/False, generation sampling, and verbalized confidence. Selection is by ILP where each atomic answer is either fully valid or fully invalid, and a compound option is only eligible if its atoms' validity pattern actually satisfies the operator (AND/OR/NEITHER-NOR).

### Setup

```
pip install -r requirements.txt
```

You'll need access to `meta-llama/Llama-3.1-8B-Instruct` on Hugging Face

```
export HF_TOKEN=your_token_here
```

### Data

Both datasets are pulled directly from Hugging Face at run time:

- LOGICAL-COMMONSENSEQA: `ojayy/logical-csqa`
- LOGICAL-SATA: `ojayy/logical-sata`

Everything here runs against the **validation** split of each dataset. The test splits used for the numbers reported in the paper aren't public, so results here won't line up exactly with the paper's tables — use validation for development and comparison, and reach out if you need the held-out test set for a direct comparison.

LOGICAL-SATA is derived from SATA-Bench (Xu et al., 2025); the construction process — pairing correct/incorrect annotated answers into compound AND/OR/NEITHER options — is described in the paper, Section 4.

### Running things

```
python lcsqa/run_direct_prompting.py
python lcsqa/run_structured_inference.py
python lsata/run_direct_prompting.py
python lsata/run_zeroshot_cot.py
python lsata/run_structured_inference.py
```

Each script writes its results to `./results` as CSV/JSON/JSONL. The structured-inference scripts run all four confidence-elicitation methods, five repeats each by default (`N_REPEATS` in the script).

### Results

On the held-out test split used in the paper, structured inference improves Macro-F1 from 48.3 (best direct-prompting baseline) to 77.0 on LOGICAL-COMMONSENSEQA, and from 47.0 to 75.6 on LOGICAL-SATA. Largest gains are on NEITHER/NOR — full breakdown by operator is in Tables 1 and 2 of the paper.

### Citation


### Contact

obed.junias@colorado.edu
