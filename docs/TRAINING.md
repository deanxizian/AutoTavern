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

The initial stage uses heuristic opponents. TavernBench data is not used to train this
policy or to choose its checkpoints. Search and an LLM agent are not implemented.

For seat-balanced matches against frozen learned checkpoints, use `python -m training.arena`.
The [readiness protocol](MODEL_READINESS.md) defines separate bot, learned-opponent,
independent-audit and live-state checks before a manually operated client trial.

## Frozen-opponent training

After establishing the first-stage baseline, freeze its strongest checkpoints and start
a new stage with `--init-from`, not `--resume`. Opponents use only their player-visible
observations. Repeated pool entries set their relative frequency; the learner rotates
through all eight seats as lanes start new episodes. For example:

```bash
bash tools/train_server.sh start runs/league \
  --init-from runs/frozen/champion.pt --warmup-samples 0 \
  --opponents runs/frozen/champion.pt runs/frozen/previous.pt bot bot \
  --cancel-guard --envs 128 --threads 8 --rollout 256 --minibatch 2048 \
  --lr 0.000006 --seed 20261030 --eval-seed 810000001 \
  --eval-games 256 --eval-every 10 --max-hours 6 --skip-baselines
```

`LeagueEnv` advances each lane to the learner's next decision or terminal placement.
It holds already-ready lanes unchanged while other lanes finish their opponents'
actions. Waiting states do not enter PPO, and the terminal placement reward is attached
to the preceding learner action. Opponent actions for a tick are chosen before any seat
acts, then applied in seat order, with readiness captured at the start of the tick.
Game/RNG state, opponent identities, episode/seat assignments and decoder history are
included in checkpoints. Exact resume rejects changed opponent hashes or decoding mode.

This is a fixed opponent pool, not the report's human-calibrated opponent libraries and
not a continuously updated self-play population. It uses Python Game lanes and batches
frozen-policy inference, so measure its throughput before scaling the run. Native game
guards remain active and a guard failure stops the run with its last good checkpoint.

In this stage, `best.pt` is selected by mean placement against the fixed validation pool,
not by the earlier bot benchmark. Each validation writes `league_eval_XXXXXX.json` with
individual lobby outcomes, the opponent manifest, behavior counters and cutoff checks.
The standalone bot benchmark remains available for regression checks. Do not compare
the two validation curves as though the opponent population had stayed unchanged.

## Cancellation decoding

Greedy policies can repeatedly open a target selector, cancel it and reopen it without
changing the visible state. `--cancel-guard` retains cancellation but excludes that same
opener after an exact return to the same player observation and native legal mask in
the same turn. It does not force a target or add an illegal action. Other states and the
next turn remain available. The short history is part of the resumable policy state.

Use the same option in `training.evaluate` and `training.arena` when assessing this
decoder. Arena applies it equally to all learned policies; the heuristic is unchanged.
Reports record the decoder version. Scores from plain greedy and guarded decoding
describe different deployed policies and must be labeled accordingly.
