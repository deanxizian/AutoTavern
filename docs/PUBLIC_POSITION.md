# Public position decoder

`training.position` packages an entity actor with a frozen encoder and a ridge model
for a small stationing decision. It reads the same public features, card IDs and legal
mask as the actor. It does not read a native game, hidden opponent boards or the client.
The ridge artifact is learned separately; the repository does not supply trained weights.

When the actor chooses End, version 1 can choose one legal Move with predicted utility
above its frozen threshold, then ends the turn when End is legal again. Pending choices
still use the actor. It only considers turns 4–20, boards with at least four minions,
at least four legal Moves, and a known, alive, non-ghost next opponent. A Move inserts
one minion at another position; it is not a swap.

The ridge's `encoder_sha256` must match its exact training encoder. A newer actor may
use that same frozen encoder, but the new combination requires its own whole-game
evaluation. Do not feed the ridge embeddings from the newer actor merely because
their dimensions match.

## Package and evaluate

Use trusted local checkpoints. The command copies and hashes all components, validates
their binding and rule profile, and creates a new directory without overwriting one.
If actor and encoder use the same input file, the bundle shares its copy.

```bash
.venv/bin/python -m training.position \
  --actor runs/frozen/candidate.pt \
  --encoder runs/frozen/position-encoder.pt \
  --ridge runs/position/ridge.pt --threshold 0.025 \
  --directory runs/frozen/position-candidate

.venv/bin/python -m training.arena runs/frozen/position-candidate/actor.pt \
  --position-bundle runs/frozen/position-candidate/position.json \
  --cancel-guard --games 256 --seed 920000001 \
  --opponents runs/frozen/reference.pt runs/frozen/reference.pt \
    runs/frozen/reference.pt runs/frozen/reference.pt runs/frozen/reference.pt \
    runs/frozen/reference.pt runs/frozen/reference.pt \
  --out runs/arena/position-development.json

.venv/bin/python -m training.evaluate runs/frozen/position-candidate/actor.pt \
  --position-bundle runs/frozen/position-candidate/position.json \
  --cancel-guard --games 512 --seed 930000001 \
  --out runs/arena/position-bots-development.json
```

Choose seeds unused for training when assessing strength. These are development commands,
not a final audit. Reports identify the decoder and all component hashes. The
`--position-bundle` option applies to the candidate. To preserve an enhanced reference,
pass `--opponent-position-bundles` with exactly seven bundle paths or `none`, aligned
with the seven `--opponents` entries. A bundle requires a learned actor in that role;
all learned roles use the cancellation guard. Each bundle's actor, encoder, ridge and
threshold are checked independently. Plain and enhanced roles can share actor weights
without sharing their decoding behavior.

Arena format 4 reports include `role_decoding` in candidate/opponent order, alongside
`role_models`. `position_decoders` records the roles covered by each shared decoder,
its aggregate counters and events with stable slot IDs. Candidate-only events remain
in `position_decoder`; aggregate counters are omitted there when the same decoder
also controls opponents, so opponent actions are not counted as candidate actions.
The two commands use different decision drivers, so compare policies within one driver.
The bundle's rule profile cannot be overridden by the evaluation flags.

CUDA scoring uses FP32 tokens with process-wide matmul precision `highest`. The CLI
sets this once before loading policies. The entity actor continues to use its usual
BF16 autocast. The decoder neither changes global precision inside forward nor installs
temporary module hooks.

## History and readiness

Library callers provide stable scheduler slot IDs with `set_context` before each
forward call. Use `save_context`/`load_context` for JSON history restoration, and
`reset_context(identity)` when a game ends. Resetting all context starts a new match
and clears its counters. Only pass ready observations. Forward records the returned
decision as executed; a speculative preview must restore its history if the action
is not executed.

The normal arena resets terminal slots. The single-seat bot evaluator gives auto-reset
episodes new identities and only reports Move events from the first counted episode.

A loadable bundle or a passing implementation test does not establish playing strength.
`training.readiness freeze` accepts `--candidate-position-bundle`,
`--champion-position-bundle` and `--reference-position-bundle`. Format 2 preserves all
components and per-role decoders, checks reference-selection evidence and excludes
encoder-lineage and research seeds before drawing the final seeds. The existing
thresholds and 3,072-game protocol in [MODEL_READINESS.md](MODEL_READINESS.md) are
unchanged. Client integration remains a separate check after simulator qualification.
