# Running the experiments

Run each subexperiment by hand, one case at a time. Evaluate its results, decide, edit the `SEL` block at the top of the later scripts, then start the next one. Each script takes only the case and calls `train_agent.sh` directly, with every setting written out. Seeds run strictly one after another, so each has the machine to itself and its runtime stays clean. A rerun skips finished seeds, so use a new experiment name if a setting changes.

Setup, once: set `CONDA_ENV`, `JAVA_BIN` and `ECLIPSE_PLUGINS` in `.env` (see `.env.example`), then run `server-executables/build.sh`. `server-executables/build/` is self-contained: on a machine without Eclipse, copy it over and skip the build.

Setup, once per case: generate the RQ1 dataset that all RQ1 runs share (sizes span all benchmarks of the case):

```sh
./generate_dataset.sh --case cra --sizes 16-176 --train 20,40,60 --val 10 --test 10 --seed 1 --out models/generated/cra/rq1/seed-1
./generate_dataset.sh --case slb --sizes 9-110  --train 20,40,60 --val 10 --test 10 --seed 1 --out models/generated/slb/rq1/seed-1
./generate_dataset.sh --case sp  --sizes 37-136 --train 20,40,60 --val 10 --test 10 --seed 1 --out models/generated/sp/rq1/seed-1
```

No generated model equals a benchmark or another generated model, so the splits are disjoint. A folder that already holds a dataset is skipped, so move an older `seed-1` away first. RQ3 generates its datasets itself, one per seed, just before training it.

**How runs work**

- **RQ1a** trains until 10 validation phases in a row (one every 10 iterations, 50 episodes per validation model) bring no gain of 1% in the mean validation return, capped at 1M FEs (`--select-on val`).
- **RQ1b–e and RQ3** train to convergence: no gain of 1% in the training return across 15 passes over the training set, after at least 100k FEs. Both stop at 500k FEs at the latest; lower RQ3's cap in `rq3_ranges.sh` if RQ1 shows convergence well below it.
- **RQ2** trains for exactly 100k FEs per instance.
- **Fitness evaluations:** one FE is one step that returned a successor model.
- **Best policy:** the metric behind stopping (training return, or validation return in RQ1a) picks `best_model.pth`. After training, it samples one batch of 100 solutions (50 in RQ1a) on each model of the test dataset; RQ1a adds one greedy episode per model (`--final-mode both`).
- **Periodic validation** (`--val-path`) records solutions and returns per validation model. It picks the best policy only with `--select-on val`, as in RQ1a.

For the commands below (`c` is `cra`, `slb` or `sp`):

```sh
c=cra; R=GNN-MT/models/rl/$c; A=evaluation/scripts
```

Each run folder `$R/<experiment>/<seed>-<ts>/` holds:

| File | Holds |
| --- | --- |
| `best_model.pth` | Policy with the best training return |
| `ac_<iter>-<fe>-<metric>.pth` | Checkpoints at improvements, or at every tick with `--keep-checkpoints all` |
| `checkpoints.csv` | Per tick (one pass over the training set): iteration, FE, stored file, training metric, mean and std of raw return in the window, `is_best`. These describe the moving policy on training models |
| `final_eval/test/<model>/nn-solutions.txt` | FE line, then the archive of the batch of 100 |
| `train_log.jsonl` | Per iteration: training metrics, invalid-action share |
| `run_args.json` | Arguments, dataset paths, commit, hardware |
| `runtimes.json` | Training and mask seconds, FEs to convergence; per model, the batch's seconds and FEs. Written last, so it marks the seed complete |

## RQ1a: encoder

```sh
experiments/rq1a_encoder.sh $c
```

**Produces:** `$R/rq1a-{sage,gatv2}-n{20,40,60}/<seed>-<ts>/`, trained on `models/generated/$c/rq1/seed-1/train-<n>/`, validated on its `val/` and tested on its `test/`. Besides the files above:
- `eval_solutions/stochastic/<model>/iter<N>-t<FE>/nn-solutions.txt`: solutions of each validation phase per validation model.
- `val_returns.csv`: mean and std return per validation phase and model.
- `checkpoints.csv`: one row per validation phase, with the mean validation return as the metric.
- `final_eval/test-greedy/<model>/nn-solutions.txt`: the single greedy episode of the best policy.

**Analyze:**

```sh
O=evaluation/rq1-learnability/encoder/$c
python $A/hv_over_fes.py     --runs $R/rq1a-* --source validation --out $O/val
python $A/reward_over_fes.py --runs $R/rq1a-* --out $O
python $A/rank_configs.py    --runs $R/rq1a-* --out $O
python $A/compare_hv.py      --runs $R/rq1a-* --out $O
python $A/compare_hv.py      --runs $R/rq1a-* --dataset test-greedy --out $O
python $A/runtimes.py        --runs $R/rq1a-* --out $O
```

**Gives:** validation HV over FEs (`$O/val/`), `reward_over_fes.pdf` (here the validation return per phase), `ranks.csv`, `ranks.tex`, `hv_test.csv`, `hv_test.tex` (sampled) and `hv_test-greedy.csv`, `hv_test-greedy.tex` (greedy), `runtimes.csv`.

**You decide:** the encoder per domain. Set it in the `SEL` block of the later scripts, e.g. `cra) SEL="graph_encoder=gatv2" ;;`.

## RQ1b: hyperparameter sensitivity, from the chosen encoder

```sh
experiments/rq1b_sensitivity.sh $c
```

**Produces:** `$R/rq1b-*/<seed>-<ts>/`.

**Analyze,** with the chosen encoder's RQ1a folder as the base setting:

```sh
O=evaluation/rq1-learnability/sensitivity/$c
python $A/rank_configs.py --runs $R/rq1a-<chosen> $R/rq1b-* --out $O
python $A/compare_hv.py   --runs $R/rq1a-<chosen> $R/rq1b-* --reference rq1a-<chosen> --out $O
python $A/runtimes.py     --runs $R/rq1b-* --out $O
```

**Gives:** `$O/ranks.csv`, `ranks.tex`, `hv_test.csv`, `hv_test.tex`, `runtimes.csv`.

**You decide:** the setting per domain. Add any changed values to `SEL` in the later scripts.

## RQ1c: masking on vs. off

```sh
experiments/rq1c_masking.sh $c
```

**Produces:** `$R/rq1c-masking-on/` and `$R/rq1c-masking-off/`, with invalid-action share in `train_log.jsonl` and mask time in `runtimes.json`. The masking-on runs are the selected setting and the reference for the next subexperiments.

**Analyze:**

```sh
O=evaluation/rq1-learnability/masking/$c
python $A/compare_hv.py --runs $R/rq1c-masking-* --reference rq1c-masking-on --out $O
python $A/runtimes.py   --runs $R/rq1c-masking-* --out $O
```

**Gives:** `$O/hv_test.csv`, `hv_test.tex`, `runtimes.csv`.

**You decide:** nothing carries forward, unless masking off wins.

## RQ1d: training-set size

```sh
experiments/rq1d_training_size.sh $c
```

**Produces:** `$R/rq1d-n20/<seed>-<ts>/`, trained on `train-20/`, which is contained in `train-40/`.

**Analyze:**

```sh
O=evaluation/rq1-learnability/training-size/$c
python $A/compare_hv.py      --runs $R/rq1c-masking-on $R/rq1d-n20 --reference rq1c-masking-on --out $O
python $A/reward_over_fes.py --runs $R/rq1c-masking-on $R/rq1d-n20 --out $O
python $A/runtimes.py        --runs $R/rq1c-masking-on $R/rq1d-n20 --out $O
```

**Gives:** `$O/hv_test.csv`, `hv_test.tex`, `reward_over_fes.pdf`, `runtimes.csv`.

**You decide:** RQ3's training size (`--train 40`, `train-40`, `es_tick_episodes=40`) and FE cap (`--fes`, just above the FEs to convergence that RQ1 shows), in `rq3_ranges.sh`.

## RQ1e: discounting

```sh
experiments/rq1e_gamma.sh $c
```

**Produces:** `$R/rq1e-gamma1/<seed>-<ts>/`.

**Analyze:**

```sh
O=evaluation/rq1-learnability/gamma/$c
python $A/compare_hv.py --runs $R/rq1c-masking-on $R/rq1e-gamma1 --reference rq1c-masking-on --out $O
python $A/runtimes.py   --runs $R/rq1c-masking-on $R/rq1e-gamma1 --out $O
```

**Gives:** `$O/hv_test.csv`, `hv_test.tex`, `runtimes.csv`.

**You decide:** whether to add `gamma=1.0` to `SEL`.

## RQ1f: continued sampling

Trains nothing: the stored policies of `rq1c-masking-on` sample 10 batches of 100 solutions on each held-out model.

```sh
experiments/rq1f_sampling.sh $c
```

**Produces:** `$R/rq1c-masking-on/<seed>-<ts>/sampling_curve/heldout/<model>/nn-solutions.txt`, with an FE line and the archive per batch, and `runtimes.json` with the seconds per batch.

**Analyze:**

```sh
O=evaluation/rq1-learnability/sampling/$c
python $A/hv_over_fes.py --runs $R/rq1c-masking-on --source sampling --dataset heldout --out $O
```

**Gives:** `$O/hv_over_fes_sampling.pdf`, `.csv`.

**You decide:** whether RQ3 should sample more than one batch. To change it, raise `--eval-k` in `rq3_ranges.sh`, or sample stored RQ3 policies with `run_policy.sh --mode batches`.

## RQ2: single instances (30 seeds, 100k FE)

```sh
experiments/rq2_instances.sh $c
```

**Produces:** `$R/rq2-<instance>/<seed>-<ts>/` with `final_eval/test/<instance>/`, and `server-executables/output/nn_logs/$c/rq2-<instance>/<seed>-<ts>/running-solutions.txt`, the rollout archive per rollout.

**Analyze,** per instance:

```sh
O=evaluation/rq2-performance/$c/<instance>
python $A/hv_over_fes.py --runs $R/rq2-<instance> --source training --out $O
python $A/compare_hv.py  --runs $R/rq2-<instance> --source training --out $O
python $A/runtimes.py    --runs $R/rq2-<instance> --out $O
```

**Gives:** `$O/hv_over_fes_training.pdf`, `hv_training.csv`, `hv_training.tex`, `runtimes.csv` (training seconds, and the seconds of one batch of 100 on the instance).

Pending until baselines exist: comparison against them, convergence budget, runtime break-even.

## RQ3: generalization (30 seeds, a new dataset per seed)

| Dataset | CRA features | SLB stacks | SP items | Benchmarks in range |
| --- | --- | --- | --- | --- |
| D_B1 | 16–19 | 9–11 | 37–46 | B1 |
| D_{[B1,B2]} | 16–38 | 9–27 | 37–70 | B1, B2 |
| D_{[B1,B3]} | 16–88 | 9–55 | 37–130 | B1–B3 |
| D_{[B1,B4]} | 16–176 | 9–110 | 37–136 | all |

Benchmarks outside a range are the extrapolation cases.

```sh
experiments/rq3_ranges.sh $c
```

**Produces:** `models/generated/$c/rq3-<range>/seed-<i>/` (run `i` of 30, generated just before it trains) and `$R/rq3-<range>/<seed>-<ts>/`, with `final_eval/test/<benchmark>/`, plus `best_model.pth` for later reuse with `run_policy.sh --checkpoint`.

**Analyze:**

```sh
O=evaluation/rq3-generalization/$c
python $A/compare_hv.py      --runs $R/rq3-* --out $O                          # ranges per benchmark
python $A/compare_hv.py      --runs $R/rq3-* $R/rq2-* --dataset any --out $O/vs-rq2  # reuse vs. per-instance, by model name
python $A/reward_over_fes.py --runs $R/rq3-* --out $O
python $A/runtimes.py        --runs $R/rq3-* --out $O
```

**Gives:** `$O/hv_test.csv`, `hv_test.tex`, `vs-rq2/hv_any.csv`, `vs-rq2/hv_any.tex`, `reward_over_fes.pdf`, `runtimes.csv`.

Pending until baselines exist: comparison against them and break-even n\*.

## Run counts

| Subexperiments | Runs | FEs per run |
| --- | --- | --- |
| RQ1a | 6 settings × 3 domains × 5 seeds = 90 | Until validation patience, up to 1M |
| RQ1b–e | 12 settings × 3 domains × 5 seeds = 180 | To convergence, 100k–500k |
| RQ1f | No training | 10 batches of 100 per held-out model |
| RQ2 | 4 × 3 × 30 = 360 | 100k |
| RQ3 | 4 × 3 × 30 = 360 | To convergence, up to 500k |
