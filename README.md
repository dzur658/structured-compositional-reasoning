## Fixes
This branch changes two constants in the run scripts:
- `SAMPLE_TEMPERATURE` (both `run_structured_inference.py` files):
  0.0 → 0.7
- `SAMPLES_PER_OPERATOR` (lsata only): 500 → 250
No other logic is modified.

### VLLM Greedy Sampling Error
VLLM version 0.15.0 (as requested by the original repository's `requirements.txt`) running on CUDA 13 will result in the following tracebacks:

**lcsqa Structured Inference**

```
Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "/home/dzur/ai-projects/structured-compositional-reasoning/lcsqa/run_structured_inference.py", line 1414, in <module>
    sp_sampling_n = SamplingParams(temperature=SAMPLE_TEMPERATURE, max_tokens=1, n=5)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/dzur/ai-projects/structured-compositional-reasoning/.venv/lib/python3.12/site-packages/vllm/sampling_params.py", line 372, in __post_init__
    self._verify_greedy_sampling()
  File "/home/dzur/ai-projects/structured-compositional-reasoning/.venv/lib/python3.12/site-packages/vllm/sampling_params.py", line 481, in _verify_greedy_sampling
    raise ValueError(f"n must be 1 when using greedy sampling, got {self.n}.")
ValueError: n must be 1 when using greedy sampling, got 5.
```

**lsata Structured Inference**

```
Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "/home/dzur/ai-projects/structured-compositional-reasoning/lsata/run_structured_inference.py", line 1674, in <module>
    sp_sampling_n = SamplingParams(temperature=SAMPLE_TEMPERATURE, max_tokens=1, n=10)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/dzur/ai-projects/structured-compositional-reasoning/.venv/lib/python3.12/site-packages/vllm/sampling_params.py", line 372, in __post_init__
    self._verify_greedy_sampling()
  File "/home/dzur/ai-projects/structured-compositional-reasoning/.venv/lib/python3.12/site-packages/vllm/sampling_params.py", line 481, in _verify_greedy_sampling
    raise ValueError(f"n must be 1 when using greedy sampling, got {self.n}.")
ValueError: n must be 1 when using greedy sampling, got 10.
```

The issue is caused by `SAMPLE_TEMPERATURE` in both respective `run_structured_inference.py` files being set to `0.0`. Setting `SAMPLE_TEMPERATURE` to a value of `0.7` as recommended by Appendix A resolves the issue.

### SAMPLES_PER_OPERATOR Index Error
By default `SAMPLES_PER_OPERATOR` is set to 500 in `lsata/run_structured_inference.py` which will result in the following traceback when ran on the [logical-sata](https://huggingface.co/datasets/ojayy/logical-sata) validation split:

```
Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "/home/dzur/ai-projects/structured-compositional-reasoning/lsata/run_structured_inference.py", line 1801, in <module>
    run()
  File "/home/dzur/ai-projects/structured-compositional-reasoning/lsata/run_structured_inference.py", line 1682, in run
    sampled = sample_stratified(examples)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/dzur/ai-projects/structured-compositional-reasoning/lsata/run_structured_inference.py", line 159, in sample_stratified
    raise ValueError(
ValueError: not enough AND examples: need indices [0:500] (500 after offset 0), found 250 in split.
```

The traceback occurs because the unreleased test set has a total of 2000 samples (500 for each operator) while the released validation set only has a total of 1000 samples (250 for each operator). Setting the default value for `SAMPLES_PER_OPERATOR` to 250 resolves the issue.

---

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

```
@misc{junias2026atomicevidencelogicalcomposition,
      title={From Atomic Evidence to Logical Composition: Structured Compositional Reasoning over Compound Answer Options}, 
      author={Obed Junias and Maria Leonor Pacheco},
      year={2026},
      eprint={2608.12836},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2608.12836}, 
}
```
### Contact

obed.junias@colorado.edu
