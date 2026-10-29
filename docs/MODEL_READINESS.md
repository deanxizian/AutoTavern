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

Record a bounded training budget and an early-stop rule before each stage on the
existing single GPU. Review learning curves, opponent diversity and failed games
before extending training. Extra compute alone is not a readiness criterion.

## Simulator gates

Freeze the simulator rules profile and use a 128-action cap throughout each comparison.
For current-client preparation, audit the published patch changes first (see
[PATCH_COMPATIBILITY.md](PATCH_COMPATIBILITY.md)); apply the same profile to every seat.
Legacy-rule scores do not certify a candidate under a new profile.

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

Training may use an explicit external turn horizon to bootstrap and archive unfinished
games (see [TRAINING.md](TRAINING.md)). This does not change any gate above: validation
and final evaluation retain complete lobbies, and any native guard ending fails the
assessment. Training truncation counts must remain visible alongside placement scores.

`training.arena` controls all eight seats and rotates every role through every seat.
This uses a different action interleaving from the single-seat built-in-bot benchmark;
compare policies within the same protocol, not scores from the two drivers directly.
It records one first complete episode per lane, not the fastest N results. There is one
candidate result per independent lobby; do not treat seven opponent results in the same
lobby as seven independent candidate samples.

Audited arena evaluation retains terminal games and verifies their placements directly.
The single-seat batch benchmark replays terminal transitions from pre-step game copies
before auto-reset can hide them. Both verify that placements match and
reject combat truncation, turn-limit endings and driver stops. A report without
`guards_checked: true` is development evidence only. Guard failures are retained in a
`.failed.json` file. Candidate behavior reports count repeated visible observations,
operation types and decisions per recruit turn; review these together with inference
latency before expecting a person to execute the recommendations.

Both evaluation commands retain other runtime failures (including non-finite policy
outputs, masked actions and driver stops) in `.failed.json` too, with requested seed,
game count and checkpoint provenance. A failed run has no successful score report;
do not reuse its output path or treat the absence of a result as a completed test.

When assessing a history-aware cancellation decoder, freeze its version along with the
weights. Apply it equally to candidate and learned references in arena matches, retain
its mode in every report, and validate its state restoration. A decoder change requires
a new assessment; earlier plain-greedy scores do not certify the guarded policy.

Freeze the decision driver too. Arena and league training now use
`round_robin_player_views_v1`: a player observes the game at its actual decision point,
after earlier seats have acted. This fixes stale actions when a recruit-phase death
gives Bigglesworth a Discover. Earlier simultaneous-action arena scores remain
development evidence; new candidates require fresh evaluation with the corrected driver.

Example development match (use immutable copies of checkpoints):

```bash
.venv/bin/python -m training.arena runs/frozen/candidate.pt \
  --opponents runs/frozen/reference.pt runs/frozen/reference.pt \
    runs/frozen/reference.pt runs/frozen/reference.pt runs/frozen/reference.pt \
    runs/frozen/reference.pt runs/frozen/reference.pt \
  --games 256 --seed 920000001 --out runs/arena/development-01.json
```

## Frozen final audit

After development evidence supports a candidate, use `training.readiness` to copy the
candidate and two chosen references into a new audit directory. Choose the strongest
available prior-stage reference from the retained comparisons; the tool checks their
identities, not the strategic judgment behind that choice. Supply those comparison
reports and a short selection note. Only after the copies are hashed does it draw three
random audit seeds, excluding recorded model-lineage and history seeds. The freezer
scans JSON and JSONL recursively under `--history-root` (default `runs`), including
research plans, failed confirmations and training metrics. Use `--seed-history` for
additional JSON/JSONL files outside that directory. A portable `seed-history.json`
retains the source hashes and excluded seeds; malformed histories fail the freeze.

```bash
.venv/bin/python -m training.readiness freeze runs/frozen/candidate.pt \
  --champion runs/frozen/prior-champion.pt --reference runs/frozen/second-reference.pt \
  --reference-evidence runs/arena/reference-comparison.json \
  --reference-selection-note "Strongest prior stage and its strongest earlier reference" \
  --directory runs/audits/candidate-01
.venv/bin/python -m training.readiness run runs/audits/candidate-01/manifest.json
.venv/bin/python -m training.readiness verify runs/audits/candidate-01/manifest.json
```

For a policy with the public position decoder, add
`--candidate-position-bundle path/to/position.json` to `freeze`. Freeze an enhanced
champion or reference with `--champion-position-bundle` or
`--reference-position-bundle` respectively. All three actor checkpoints remain
required and distinct. Format 2 copies each bundle's exact manifest, actor, encoder
and ridge components before drawing seeds, and includes the encoder's training
lineage in the exclusions. Reference evidence must cover the selected policies'
actor, encoder, ridge, threshold, precision and cancellation guard; an earlier
plain-policy comparison alone does not justify selecting its enhanced version.

The runner supplies the frozen bundles to their exact roles in both learned suites.
The verifier checks per-role decoder identities and Move-event ownership, thresholds,
turn eligibility and counters in addition to the existing gates. This changes the
policy representation, not the performance thresholds or number of games. Plain
format-1 audits remain supported. Copying a bundle or passing protocol tests does
not establish model readiness.

The one-shot runner executes all nine assessments (3,072 complete games), retains logs
and raw reports, and stops on a failed assessment. It will not reuse an existing run.
After an interrupted or failed audit, preserve its directory and freeze a new audit
after resolving the cause; do not replace only the unfavorable seed. A valid set of
nine completed assessments can still fail the performance thresholds.

The verifier recomputes scores and confidence bounds from raw placements. It rejects
missing or duplicated lobbies, unbalanced seats, changed models/opponents/rules/decoder,
unchecked guards, changed source, evaluations predating the freeze, and mismatched
reported scores. It checks every seed as well as the combined results. `readiness.json`
records the manifest and report hashes. A result with `performance_gates_passed: true` establishes the numerical performance
and complete-game checks. Deployment loading/latency still need verification before a
trial notification, and `live_ready` remains false until the separate checks below.

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
