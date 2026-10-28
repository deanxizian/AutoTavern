# Paper-informed PPO training

This directory adds a training implementation to the published TavernGym environment.
It is **not** the paper's TavernAgent, whose training code, weights and calibrated opponent
libraries were not released. The architecture and PPO settings are informed by
[the technical report, Section 6 and Appendix C](https://arxiv.org/html/2610.33845v1#A3).
Beating the bundled heuristic bots does not establish strength against humans or other agents.

## Environment

Use Python 3.10+, the Rust version in `rust-toolchain.toml`, and a CUDA-compatible PyTorch
installation. Install with `pip install -e ".[train,test]"`. On a GPU image with a working
PyTorch installation, a `python -m venv --system-site-packages .venv` environment can reuse
that installation without replacing its CUDA libraries. Check `torch.cuda.is_available()`
and `taverngym.build_is_current()` before training.

## Method

- One learning seat against seven bundled heuristic bots, in independent Rust batch games.
- An entity transformer (width 384, eight layers, eight attention heads), card identity,
  static card features and frozen rule-text features, signed-log numeric inputs, structured
  pointer action scores, a value head and an auxiliary eight-way placement head. CUDA
  matrix operations use bfloat16; action probabilities, PPO ratios and losses use float32.
  A small MLP remains available via `--model mlp` for diagnostics.
- Optional supervised initialization from the heuristic bot (262,144 decisions by default),
  followed by masked PPO, GAE, terminal placement reward, gradient clipping and KL stopping.
  The heuristic supplies operation examples, not expert strategy labels.
- Default discount 1.0 and GAE lambda 0.995. Only the environment's placement reward is
  optimized; no invented reward for buying, upgrading or high minion stats is added.
- The simulator's legal-action mask applies during both collection and optimization.
  Rejected actions or a game stopped without a valid placement fail the run explicitly.
- A cap of 128 actions per recruit turn prevents infinite rearrangement/freezing loops.
  This is a training/evaluation convention, not an official game rule. All reported
  comparisons use the same cap. The other simulator settings remain at their defaults.

Differences from the report are explicit: the public observation produces 49 tokens here,
and omits attachment identities and some persistent-effect details. Text features use the
public `sentence-transformers/all-MiniLM-L6-v2`; the report does not specify its text encoder.
Stage one uses heuristic opponents and a supervised operation warm-up, not the authors'
private human-calibrated libraries or their simplified-rule curriculum. It starts at learning
rate 1e-4 for the warm-up and 2e-5 for PPO; the report's 6e-6 to 5e-6 schedule is for later
stages of an already trained agent. KL stopping is checked before every minibatch update,
not only at the end of an epoch.
No human-level performance claim follows from this initial stage.

Auxiliary placement labels come only from episodes that actually finish in the collected
rollout. Later stages can use `--init-from best.pt --warmup-samples 0 --lr 0.000006`: a frozen
copy of the initial policy supplies a KL anchor (coefficient 0.1, first 100 updates).
Exact continuation instead uses `--resume`; it restores the optimizer, games and RNGs.

The PPO update structure follows the clipped objective described in
[CleanRL's PPO implementation](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo.py).
For action-mask semantics, see
[Maskable PPO's documentation](https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html).
The implementation here directly uses TavernGym's batch API rather than an SB3 wrapper.

## Start and inspect

From the repository root on a Linux GPU server:

```bash
.venv/bin/python -m training.prepare_text --out runs/assets/card-text.pt
bash tools/train_server.sh start runs/ppo-baseline \
  --text-embeddings runs/assets/card-text.pt --minibatch 1024 --max-hours 24
bash tools/train_server.sh status runs/ppo-baseline
tail -f runs/ppo-baseline/train.log
```

The first stage has a 24-hour cumulative training budget and stops after 30 validation
rounds without a new best mean placement. These limits are adjustable; a longer run is not
evidence of a stronger model. Reassess learning curves before increasing the budget.

Artifacts in each run directory:

| File | Meaning |
| --- | --- |
| `manifest.json`, `training_source_*` | Arguments, engine and training-code fingerprints, source snapshot, device and model size |
| `metrics.jsonl`, `train.log` | Training progress, failures and evaluation results |
| `status.json` | Last reported state; check PID too, as a crashed process can leave stale status |
| `random_baseline.json`, `bot_baseline.json` | Reference policies on validation seeds |
| `latest.pt` | Model, optimizer, all live games, RNG states, counters and elapsed time |
| `best.pt`, `best_eval.json` | Checkpoint selected by validation mean placement |
| `tensorboard/` | TensorBoard scalar logs |

Checkpoints are written atomically, every five minutes and at evaluation or graceful stop.
Only load trusted checkpoint files: training state is serialized with PyTorch/pickle.
TensorBoard can be run on loopback and accessed through an SSH tunnel; no public port is needed.

## Evaluation and continuation

Validation uses different seeds from training, an equal number of games at all eight seats,
and each lane's first complete episode. It reports mean placement (lower is better), an
approximate 95% confidence interval, top-four rate, win rate and counts for each placement.
Validation is used for checkpoint selection and must not be described as a fresh test set.
Policy evaluation is greedy, while training samples from the policy.

After model selection, use a fresh seed range for an independent test:

```bash
.venv/bin/python -m training.evaluate runs/ppo-baseline/best.pt \
  --games 512 --seed 900000001 --out runs/ppo-baseline/test_eval.json
```

Graceful stop and exact continuation:

```bash
bash tools/train_server.sh stop runs/ppo-baseline
# Wait until status reports stopped and latest.pt is written.
rm runs/ppo-baseline/STOP
bash tools/train_server.sh resume runs/ppo-baseline --max-hours 48
```

Use the original `--envs`, `--seed`, `--gamma`, `--gae-lambda`, `--action-cap` and other
settings when resuming a customized run. The new time limit is cumulative, not additional.
Resuming is rejected if the simulator/card fingerprint differs. A failed update preserves
the last good checkpoint. A kill -9 or machine loss may lose progress since that checkpoint.

This first stage does not implement an opponent league, self-play, search, or an LLM agent.
Those are separate improvements to assess against this reproducible baseline. TavernBench
data is not used to train this policy or to choose its checkpoints.
