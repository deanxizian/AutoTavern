# Readiness for a supervised live-game trial

This protocol defines when a checkpoint is worth a small, manually operated live-game
trial. It does not establish human rating, tournament strength, or equivalence to the
paper's unreleased TavernAgent. Thresholds are set before the new arena is used for
model selection. Keep failed results, and report any later protocol revision explicitly.

## Development and final evaluation

Training seeds, checkpoint-selection validation seeds, and arena development seeds are
development data. Repeatedly inspecting a seed makes it development data as well.
After choosing a candidate and freezing its opponents, record their SHA-256 hashes;
only then draw three new audit seeds. Run the entire protocol once on those seeds.
If it fails and training is changed, retain the failure and use new audit seeds for the
next candidate. Do not select the most flattering seed or report only the best matchup.

The initial operating budget is the existing single GPU and the trainer's 24-hour
cumulative limit, with a stop after 30 validation rounds without improvement. Review
learning curves, opponent diversity and failed games before extending training. Extra
compute alone is not a readiness criterion.

## Simulator gates

Use the unchanged simulator rules and a 128-action cap throughout each comparison.

1. Against the seven built-in bots: at least 512 games on each of three audit seeds,
   balanced across all eight seats. Across the 1,536 games, mean placement's 95% upper
   confidence bound must be at most 2.6, top-four rate's 95% Wilson lower bound at least
   80%, and first-place rate's lower bound at least 35%. Each seed must independently
   have mean placement below 3.0. This is a deliberately strong *bot* benchmark.
2. Against frozen learned opponents: one candidate per lobby, with at least 256 complete
   lobbies per audit seed for each of two suites. Suite A uses seven copies of the prior
   stage's strongest frozen model. Suite B uses two copies each of two distinct learned
   checkpoints and three built-in bots. Choose the reference checkpoints before auditing;
   include the strongest available prior stage, not only the supervised warm-up.
   In each suite, aggregate candidate mean placement's 95% upper bound must be below
   4.0, top-four rate's lower bound at least 60%, and every seed's mean below 4.5.
3. Every counted episode must complete with a placement in 1..8. No rejected legal
   actions, invalid masks, non-finite predictions or simulator guard terminations may be
   discarded. Investigate failures and rerun a corrected protocol with fresh audit seeds.
4. Freeze a deployable checkpoint and retain all audit JSON, code/engine fingerprints,
   exact opponent identities and per-lobby placements. Verify loading and inference on
   the intended deployment device, including measured latency on representative states.

`training.arena` controls all eight seats and rotates every role through every seat.
This uses a different action interleaving from the single-seat built-in-bot benchmark;
compare policies within the same protocol, not scores from the two drivers directly.
It records one first complete episode per lane, not the fastest N results. There is one
candidate result per independent lobby; do not treat seven opponent results in the same
lobby as seven independent candidate samples.

Audited evaluation replays terminal transitions from pre-step game copies before the
batch environment's auto-reset can hide them. It verifies that placements match and
rejects combat truncation, turn-limit endings and driver stops. A report without
`guards_checked: true` is development evidence only. Guard failures are retained in a
`.failed.json` file. Candidate behavior reports count repeated visible observations,
operation types and decisions per recruit turn; review these together with inference
latency before expecting a person to execute the recommendations.

Example development match (use immutable copies of checkpoints):

```bash
.venv/bin/python -m training.arena runs/frozen/candidate.pt \
  --opponents runs/frozen/reference.pt runs/frozen/reference.pt \
    runs/frozen/reference.pt runs/frozen/reference.pt runs/frozen/reference.pt \
    runs/frozen/reference.pt runs/frozen/reference.pt \
  --games 256 --seed 920000001 --out runs/arena/development-01.json
```

## Live-state and manual-action gates

The released repository exposes simulator state and action APIs. It does not include
a Hearthstone-client state reader or controller. A client log reader is a separate
adapter: [HearthSim's hslog](https://github.com/HearthSim/python-hslog) parses Power.log
packets and can reconstruct entities and their tags. Its generic entity exporter ignores
Choices and Options packets, so using it alone does not provide TavernGym's choice order
or legal-action mask.

Before supplying model recommendations for a live game:

- Verify the actual client version against the simulator's supported patch, including
  relevant card and rule changes. An unfamiliar card, unsupported mechanic or missing
  cost/target must make the adapter request clarification, not silently invent a value.
- Validate a complete live log against visible hero selection, recruit phase, shop,
  hand, board order/stats, gold, tavern tier, choices and at least one combat transition.
  Preserve only player-visible information. Opponent memory may contain previously
  observed boards, never inferred access to hidden current hands or boards.
- Handle partial lines, log replacement/truncation, new sessions, reconnects and delayed
  updates. Mark stale or incomplete state explicitly and withhold a recommendation.
- Map client card IDs, entity IDs and positions to the exact model vocabulary; validate
  legal actions against current client choices. A simulator save file is not a substitute
  for a verified live-state conversion.
- Present one specific operation with the named card, target and position. After the
  user acts, read the resulting state before suggesting the next operation. The user
  retains control of the Hearthstone client.

Passing simulator gates can justify preparing a trial, but full live readiness also
requires this adapter validation or an explicitly checked manual state-entry workflow.
Report the remaining gap when notifying the user; never describe an untested integration
as a working client interface.
