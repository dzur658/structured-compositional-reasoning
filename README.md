# Independent Reproduction Attempt: "From Atomic Evidence to
# Logical Composition" (Junias & Pacheco, 2026)

This fork documents an independent attempt to run the released code for the paper's structured inference framework on the validation splits of both benchmarks ([logical-sata](https://huggingface.co/datasets/ojayy/logical-sata) and [logical-csqa](https://huggingface.co/datasets/ojayy/logical-csqa)), prior to a planned (and ultimately blocked) calibration-feature ablation.

## Status

This reproduction remains in a partial state. The pipeline runs after the documented fixes; the main elicitation results reproduce qualitatively. The planned ablation could not be run; see Finding 1 below.

## Setup

- Model: Llama-3.1-8B-Instruct (official Meta weights)
- Hardware: MSI Edgexpert (GB10 box like NVIDIA DGX Spark)
- Library versions (see `pyproject.toml`; vllm pinning to `0.15.0` respected)
- CUDA version: 13.0
- Evaluation: validation split, 250 instances per operator
  (AND / OR / NEITHER / Mixed), 1000 total per benchmark, per the
  repo's guidance that validation is the sanctioned evaluation
  surface (test labels are not public)
- 5 repeats per configuration

## Reproduction Results

#### Direct Prompting LCSQA

Note: the direct-prompting script slices evaluation into two 250-instance blocks per operator (SLICE_SPECS: first_250 = (0, 250), second_250 = (250, 500)), hardcoded to positions 0–500.

```
FIRST_250 — FIVE-RUN MEAN ± SAMPLE STANDARD DEVIATION
====================================================================================================
Setting      Operator        N            Accuracy            Macro-F1         Answer rate
----------------------------------------------------------------------------------------------------
0shot        AND           250     0.6664 ± 0.0292     0.6665 ± 0.0292     1.0000 ± 0.0000
0shot        OR            250     0.5656 ± 0.0227     0.5664 ± 0.0229     1.0000 ± 0.0000
0shot        NEITHER       250     0.1712 ± 0.0118     0.1650 ± 0.0119     1.0000 ± 0.0000
0shot        Mixed         250     0.5256 ± 0.0151     0.5259 ± 0.0148     1.0000 ± 0.0000
0shot        ALL          1000     0.4822 ± 0.0036     0.4832 ± 0.0034     1.0000 ± 0.0000
1shot        AND           250     0.6360 ± 0.0183     0.6368 ± 0.0183     1.0000 ± 0.0000
1shot        OR            250     0.5288 ± 0.0230     0.5263 ± 0.0256     1.0000 ± 0.0000
1shot        NEITHER       250     0.0832 ± 0.0095     0.0785 ± 0.0121     1.0000 ± 0.0000
1shot        Mixed         250     0.4568 ± 0.0077     0.4574 ± 0.0085     1.0000 ± 0.0000
1shot        ALL          1000     0.4262 ± 0.0064     0.4262 ± 0.0071     1.0000 ± 0.0000
2shot        AND           250     0.5936 ± 0.0227     0.6010 ± 0.0203     0.9720 ± 0.0085
2shot        OR            250     0.5136 ± 0.0211     0.5100 ± 0.0206     0.9896 ± 0.0046
2shot        NEITHER       250     0.0904 ± 0.0073     0.0834 ± 0.0055     0.9840 ± 0.0049
2shot        Mixed         250     0.4288 ± 0.0216     0.4348 ± 0.0226     0.9544 ± 0.0092
2shot        ALL          1000     0.4066 ± 0.0051     0.4089 ± 0.0054     0.9750 ± 0.0039
3shot        AND           250     0.5256 ± 0.0322     0.5284 ± 0.0344     0.9680 ± 0.0117
3shot        OR            250     0.4808 ± 0.0095     0.4802 ± 0.0095     0.9824 ± 0.0100
3shot        NEITHER       250     0.0896 ± 0.0159     0.0862 ± 0.0154     0.9584 ± 0.0073
3shot        Mixed         250     0.4328 ± 0.0156     0.4405 ± 0.0159     0.9528 ± 0.0091
3shot        ALL          1000     0.3822 ± 0.0075     0.3860 ± 0.0083     0.9654 ± 0.0015

SECOND_250 — FIVE-RUN MEAN ± SAMPLE STANDARD DEVIATION
====================================================================================================
Setting      Operator        N            Accuracy            Macro-F1         Answer rate
----------------------------------------------------------------------------------------------------
0shot        AND           250     0.6384 ± 0.0207     0.6377 ± 0.0205     1.0000 ± 0.0000
0shot        OR            250     0.5632 ± 0.0128     0.5633 ± 0.0125     1.0000 ± 0.0000
0shot        NEITHER       250     0.1512 ± 0.0257     0.1406 ± 0.0253     1.0000 ± 0.0000
0shot        Mixed         250     0.4656 ± 0.0054     0.4649 ± 0.0057     1.0000 ± 0.0000
0shot        ALL          1000     0.4546 ± 0.0069     0.4541 ± 0.0068     1.0000 ± 0.0000
1shot        AND           250     0.6824 ± 0.0203     0.6821 ± 0.0193     1.0000 ± 0.0000
1shot        OR            250     0.5304 ± 0.0213     0.5251 ± 0.0226     1.0000 ± 0.0000
1shot        NEITHER       250     0.0720 ± 0.0080     0.0685 ± 0.0079     1.0000 ± 0.0000
1shot        Mixed         250     0.4376 ± 0.0217     0.4349 ± 0.0223     0.9984 ± 0.0022
1shot        ALL          1000     0.4306 ± 0.0069     0.4292 ± 0.0068     0.9996 ± 0.0005
2shot        AND           250     0.6096 ± 0.0248     0.6164 ± 0.0250     0.9664 ± 0.0108
2shot        OR            250     0.4760 ± 0.0276     0.4691 ± 0.0304     0.9856 ± 0.0022
2shot        NEITHER       250     0.0816 ± 0.0088     0.0733 ± 0.0109     0.9880 ± 0.0057
2shot        Mixed         250     0.4104 ± 0.0161     0.4162 ± 0.0193     0.9632 ± 0.0115
2shot        ALL          1000     0.3944 ± 0.0089     0.3955 ± 0.0104     0.9758 ± 0.0049
3shot        AND           250     0.5776 ± 0.0331     0.5821 ± 0.0324     0.9720 ± 0.0075
3shot        OR            250     0.4912 ± 0.0237     0.4926 ± 0.0245     0.9840 ± 0.0080
3shot        NEITHER       250     0.0768 ± 0.0153     0.0710 ± 0.0162     0.9704 ± 0.0119
3shot        Mixed         250     0.4344 ± 0.0061     0.4424 ± 0.0062     0.9600 ± 0.0117
3shot        ALL          1000     0.3950 ± 0.0078     0.3983 ± 0.0095     0.9716 ± 0.0048
```

#### Structured Inference LCSQA

```
================================================================================
REPRODUCIBILITY SUMMARY (mean & std across 5 repeats)
================================================================================
Method                 Operator       Mean      Std      Var      Min      Max
--------------------------------------------------------------------------------
confidence_mc          AND           0.676    0.000   0.0000    0.676    0.676
confidence_mc          OR            0.673    0.002   0.0000    0.672    0.676
confidence_mc          NEITHER       0.708    0.000   0.0000    0.708    0.708
confidence_mc          Mixed         0.592    0.003   0.0000    0.588    0.596
confidence_mc          Overall       0.662    0.000   0.0000    0.662    0.663
independent            AND           0.672    0.000   0.0000    0.672    0.672
independent            OR            0.356    0.000   0.0000    0.356    0.356
independent            NEITHER       0.596    0.000   0.0000    0.596    0.596
independent            Mixed         0.424    0.000   0.0000    0.424    0.424
independent            Overall       0.512    0.000   0.0000    0.512    0.512
generation_sampling    AND           0.581    0.026   0.0007    0.544    0.604
generation_sampling    OR            0.594    0.020   0.0004    0.564    0.616
generation_sampling    NEITHER       0.656    0.013   0.0002    0.644    0.672
generation_sampling    Mixed         0.565    0.010   0.0001    0.552    0.580
generation_sampling    Overall       0.599    0.009   0.0001    0.589    0.609
verbalized             AND           0.463    0.020   0.0004    0.436    0.484
verbalized             OR            0.325    0.025   0.0006    0.288    0.356
verbalized             NEITHER       0.480    0.024   0.0006    0.448    0.508
verbalized             Mixed         0.426    0.028   0.0008    0.384    0.456
verbalized             Overall       0.423    0.018   0.0003    0.403    0.450
```

#### Direct Prompting LSATA

```
FIVE-RUN MEAN ± SAMPLE STANDARD DEVIATION
====================================================================================================
Setting        Operator        N            Accuracy            Macro-F1         Answer rate
----------------------------------------------------------------------------------------------------
0shot          AND           250     0.6360 ± 0.0157     0.6571 ± 0.0131     0.9256 ± 0.0140
0shot          OR            250     0.5768 ± 0.0131     0.5911 ± 0.0146     0.9336 ± 0.0115
0shot          NEITHER       250     0.1296 ± 0.0215     0.1318 ± 0.0219     0.8360 ± 0.0162
0shot          Mixed         250     0.3728 ± 0.0314     0.3844 ± 0.0312     0.9200 ± 0.0190
0shot          ALL          1000     0.4288 ± 0.0110     0.4505 ± 0.0125     0.9038 ± 0.0092
1shot          AND           250     0.7208 ± 0.0087     0.7169 ± 0.0097     0.9960 ± 0.0000
1shot          OR            250     0.6144 ± 0.0115     0.6090 ± 0.0121     0.9960 ± 0.0000
1shot          NEITHER       250     0.0928 ± 0.0203     0.0938 ± 0.0202     0.9960 ± 0.0000
1shot          Mixed         250     0.3656 ± 0.0248     0.3590 ± 0.0246     0.9960 ± 0.0000
1shot          ALL          1000     0.4484 ± 0.0126     0.4473 ± 0.0125     0.9960 ± 0.0000
2shot          AND           250     0.7136 ± 0.0122     0.7086 ± 0.0122     0.9960 ± 0.0000
2shot          OR            250     0.6136 ± 0.0061     0.6047 ± 0.0051     0.9960 ± 0.0000
2shot          NEITHER       250     0.0944 ± 0.0128     0.0944 ± 0.0128     0.9960 ± 0.0000
2shot          Mixed         250     0.3544 ± 0.0122     0.3510 ± 0.0125     0.9960 ± 0.0000
2shot          ALL          1000     0.4440 ± 0.0077     0.4428 ± 0.0076     0.9960 ± 0.0000
3shot          AND           250     0.6864 ± 0.0115     0.6841 ± 0.0117     0.9920 ± 0.0000
3shot          OR            250     0.5624 ± 0.0078     0.5532 ± 0.0079     0.9920 ± 0.0000
3shot          NEITHER       250     0.0872 ± 0.0125     0.0864 ± 0.0118     0.9920 ± 0.0000
3shot          Mixed         250     0.3464 ± 0.0234     0.3443 ± 0.0231     0.9920 ± 0.0000
3shot          ALL          1000     0.4206 ± 0.0069     0.4210 ± 0.0064     0.9920 ± 0.0000
```

#### Structured Inference LSATA

```
================================================================================
REPRODUCIBILITY SUMMARY (mean & std across 5 repeats)
================================================================================
Method                 Operator       Mean      Std      Var      Min      Max
--------------------------------------------------------------------------------
confidence_mc          AND           0.676    0.000   0.0000    0.676    0.676
confidence_mc          OR            0.628    0.000   0.0000    0.628    0.628
confidence_mc          NEITHER       0.732    0.000   0.0000    0.732    0.732
confidence_mc          Mixed         0.644    0.000   0.0000    0.644    0.644
confidence_mc          Overall       0.670    0.000   0.0000    0.670    0.670
independent            AND           0.740    0.000   0.0000    0.740    0.740
independent            OR            0.420    0.000   0.0000    0.420    0.420
independent            NEITHER       0.680    0.000   0.0000    0.680    0.680
independent            Mixed         0.524    0.000   0.0000    0.524    0.524
independent            Overall       0.591    0.000   0.0000    0.591    0.591
generation_sampling    AND           0.636    0.017   0.0003    0.616    0.660
generation_sampling    OR            0.612    0.021   0.0004    0.588    0.644
generation_sampling    NEITHER       0.626    0.007   0.0001    0.620    0.636
generation_sampling    Mixed         0.615    0.013   0.0002    0.600    0.632
generation_sampling    Overall       0.622    0.010   0.0001    0.613    0.638
verbalized             AND           0.568    0.030   0.0009    0.532    0.600
verbalized             OR            0.356    0.032   0.0011    0.332    0.408
verbalized             NEITHER       0.375    0.040   0.0016    0.308    0.404
verbalized             Mixed         0.401    0.025   0.0006    0.380    0.432
verbalized             Overall       0.425    0.009   0.0001    0.415    0.437
```

Qualitatively consistent with the paper's Table 7/8 ordering: 

- paired multiple choice strongest
- verbalized weakest
- stochastic strategies in between with visible run-to-run variance
- deterministic strategies at zero variance.

Absolute numbers are not directly comparable to the paper's tables, which report on the (unreleased) test split and use Macro-F1; this run uses the validation split and reports accuracy as computed by the released pipeline.

## Findings

The paper's core claim of NEITHER recoverability via structured inference appears to be confirmed, as is evident by the much higher scores obtained by structured inference over direct prompting.

### 1. Calibration layer (Sec. 3.5) is not implemented

**The paper's relative calibration: Platt, isotonic, and the
4-feature logistic regression from section 3.5 Score Calibration does not exist in this codebase.**

As a result, full reproduction remains blocked until if/when an implementation of section 3.5 Score Calibration is released.

### 2. OR tie pathology in the independent route
See more in the branch located [here](https://github.com/dzur658/structured-compositional-reasoning/tree/ind-or-scoring)

### 3. Sampling configuration
Appendix A states temperature 0.7; the released code sets temperature 0, which raises a vllm error under n>1 concurrent
greedy requests. Also, in each of the `run_structured_inference.py` files temperature is set as a constant twice, once at the beginning of either script with `TEMPERATURE` and again above the `run` function's definition with `SAMPLE_TEMPERATURE`. Only `SAMPLE_TEMPERATURE` needs to be changed to avoid the vllm error (changed to a temperature value of 0.7 in line with Appendix A). The fix is available for both files in [fix-vllm-greedy](https://github.com/dzur658/structured-compositional-reasoning/tree/fix-vllm-greedy).

### 4. Metrics
The released structured pipeline reports accuracy; the paper's headline metric is Macro-F1. Macro-F1 computation appears only in the baseline scripts. However, metrics (Accuracy, Precision, Recall, F1-score, and H@2) are printed to the console during runtime, meaning it is possible from the output to calculate Macro-F1 by hand. However, such scores would be generated without calibration, which would not compare with the tables in the paper.

*Print-Outs are consistent across methods for structured inference*

<details>
  <summary>Example Print-Out Metrics for generation sampling</summary>

  ```
  ======================================================================
  METHOD: generation_sampling
======================================================================

-- generation_sampling  repeat 1/5 --
Adding requests: 100%|███████████████████████████████████████████████████████████████████████████████████████████████| 5839/5839 [00:11<00:00, 523.97it/s]
Processed prompts: 100%|████████████████████████████████| 58390/58390 [04:21<00:00, 222.95it/s, est. speed input: 342957.36 toks/s, output: 222.95 toks/s]

----------------------------------------------------------------
  Metrics (generation_sampling repeat 1)
----------------------------------------------------------------
  Operator        N     Acc       P       R      F1     H@2
----------------------------------------------------------------
  AND           250   0.652   0.652   0.648   0.647   0.856
  OR            250   0.620   0.635   0.622   0.615   0.716
  NEITHER       250   0.644   0.649   0.644   0.643   0.804
  Mixed         250   0.592   0.598   0.592   0.594   0.724
  Overall      1000   0.627   0.627   0.627   0.626   0.775
----------------------------------------------------------------

-- generation_sampling  repeat 2/5 --
Adding requests: 100%|███████████████████████████████████████████████████████████████████████████████████████████████| 5839/5839 [00:11<00:00, 529.34it/s]
Processed prompts: 100%|████████████████████████████████| 58390/58390 [03:25<00:00, 284.45it/s, est. speed input: 437546.91 toks/s, output: 284.45 toks/s]

----------------------------------------------------------------
  Metrics (generation_sampling repeat 2)
----------------------------------------------------------------
  Operator        N     Acc       P       R      F1     H@2
----------------------------------------------------------------
  AND           250   0.600   0.602   0.595   0.595   0.816
  OR            250   0.588   0.615   0.595   0.590   0.688
  NEITHER       250   0.648   0.659   0.649   0.649   0.792
  Mixed         250   0.592   0.596   0.592   0.593   0.728
  Overall      1000   0.607   0.609   0.607   0.607   0.756
----------------------------------------------------------------

-- generation_sampling  repeat 3/5 --
Adding requests: 100%|███████████████████████████████████████████████████████████████████████████████████████████████| 5839/5839 [00:11<00:00, 517.81it/s]
Processed prompts: 100%|████████████████████████████████| 58390/58390 [03:23<00:00, 286.74it/s, est. speed input: 441077.01 toks/s, output: 286.74 toks/s]

----------------------------------------------------------------
  Metrics (generation_sampling repeat 3)
----------------------------------------------------------------
  Operator        N     Acc       P       R      F1     H@2
----------------------------------------------------------------
  AND           250   0.640   0.645   0.638   0.636   0.860
  OR            250   0.644   0.671   0.650   0.642   0.744
  NEITHER       250   0.652   0.656   0.651   0.650   0.836
  Mixed         250   0.616   0.621   0.616   0.617   0.748
  Overall      1000   0.638   0.642   0.638   0.637   0.797
----------------------------------------------------------------

-- generation_sampling  repeat 4/5 --
Adding requests: 100%|███████████████████████████████████████████████████████████████████████████████████████████████| 5839/5839 [00:11<00:00, 516.02it/s]
Processed prompts: 100%|████████████████████████████████| 58390/58390 [03:28<00:00, 280.48it/s, est. speed input: 431451.04 toks/s, output: 280.48 toks/s]

----------------------------------------------------------------
  Metrics (generation_sampling repeat 4)
----------------------------------------------------------------
  Operator        N     Acc       P       R      F1     H@2
----------------------------------------------------------------
  AND           250   0.628   0.628   0.623   0.621   0.836
  OR            250   0.604   0.624   0.606   0.600   0.724
  NEITHER       250   0.636   0.645   0.635   0.635   0.792
  Mixed         250   0.600   0.605   0.600   0.601   0.732
  Overall      1000   0.617   0.618   0.616   0.616   0.771
----------------------------------------------------------------

-- generation_sampling  repeat 5/5 --
Adding requests: 100%|███████████████████████████████████████████████████████████████████████████████████████████████| 5839/5839 [00:09<00:00, 587.43it/s]
Processed prompts: 100%|████████████████████████████████| 58390/58390 [03:09<00:00, 307.53it/s, est. speed input: 473062.79 toks/s, output: 307.53 toks/s]

----------------------------------------------------------------
  Metrics (generation_sampling repeat 5)
----------------------------------------------------------------
  Operator        N     Acc       P       R      F1     H@2
----------------------------------------------------------------
  AND           250   0.620   0.621   0.617   0.616   0.848
  OR            250   0.644   0.651   0.649   0.641   0.752
  NEITHER       250   0.636   0.643   0.635   0.634   0.784
  Mixed         250   0.580   0.587   0.580   0.582   0.724
  Overall      1000   0.620   0.620   0.620   0.619   0.777
----------------------------------------------------------------
  ```

</details>

### 5. Default test-set sample sizes break on validation
`lsata/run_structured_inference.py` defaults to 500 samples per operator (sized for the 2000-instance test split). On the validation split (250/operator) `sample_stratified` raises:

```python
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

Fix: pass n=250 (or auto-derive from split size) when running
validation. Patch included in `fix-vllm-greedy`.

## Branch layout

- `main` — untouched upstream
- `fix-vllm-greedy` — sampling fix (temperature and validation sample size)
- `ind-or-scoring` — see OR section

## Notes and caveats

- All findings verified against the repo as released at [6efd2d54bcfe42483122fef43d8f70ed48fd5dfa](https://github.com/obedjunias19/structured-compositional-reasoning/commit/6efd2d54bcfe42483122fef43d8f70ed48fd5dfa)

## Acknowledgments

Thank you to all those who worked on this paper, and the release of the public artifacts.

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
