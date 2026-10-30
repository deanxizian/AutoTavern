# Local model integration

The native companion can load a pinned trainer checkpoint in a separate local
process. `tools/client_model.py` uses the existing actor, embedded expert strategy,
public attachment/growth inputs, optional public position decoder and cancellation
guard. It verifies the checkpoint checksum, engine build, complete ordered card
vocabulary and observation/action dimensions before serving requests.

**Current status: experimental real-game manual advice is enabled for mapped
inputs and fully resolved actions. This is not a model-strength certification
or a claim of universal mechanic coverage.** The actor requires 3,227
float features, 124 card slots and a 245-action mask. The current training actor
also requires attachment bags, 16 public growth counters and an expert strategy
view. Missing data stays unknown (`NaN` / `-1` internally), and is reported as a
specific gap. It is never sent to the actor or replaced with zeros.

## Environment and selection

Keep the reader's `runs/client-env` independent of the model environment:

```bash
uv venv runs/model-env --python 3.13
uv pip install --python runs/model-env/bin/python -e '.[client-model]'
```

Building the native module requires the Rust version in `Cargo.toml`. An editable
installation is required here so the trainer's source-fingerprint check can find
this checkout. No training, remote GPU work, embeddings download or checkpoint
modification is needed. Load only trusted checkpoints produced by the trainer.

The window reads optional `runs/client-model/selected.json` at launch:

```json
{
  "format_version": 1,
  "checkpoint": "/absolute/path/to/bundle/actor.pt",
  "sha256": "the exact checkpoint SHA256",
  "position_bundle": "/absolute/path/to/bundle/position.json",
  "position_sha256": "the exact manifest SHA256",
  "python": "/absolute/path/to/AutoTavern/runs/model-env/bin/python",
  "threads": 4
}
```

Omit the position fields only for an intentionally actor-only configuration.
Position components must be inside their bundle, match its checksums and agree on
engine, rules and actor identity. Copy a completed artifact rather than following
a trainer's changing `best.pt`. Starting the companion does not auto-upgrade the
selected model. The model's strategy knowledge is loaded from its checkpoint;
missing public strategy inputs are not replaced by a zero prior.

`./script/build_and_run.sh` still starts the native window. Without a model
configuration it remains a reader. A model loading error is shown separately from
the log reader, which continues working. The UI shows the selected update and
input coverage in its state-review tab. A fresh, frame-matched model result can
enable the window's `live_ready`. The diagnostic log snapshot's flags and the
model admission/readiness gates remain separate.

## Verification commands

```bash
runs/model-env/bin/python -m tools.client_model inspect \
  --checkpoint runs/client-model/u20/bundle/actor.pt \
  --position-bundle runs/client-model/u20/bundle/position.json
runs/model-env/bin/python -m tools.client_model benchmark \
  --checkpoint runs/client-model/u20/bundle/actor.pt \
  --position-bundle runs/client-model/u20/bundle/position.json \
  --decisions 96 --out runs/client-model/u20/benchmark.json
runs/model-env/bin/python -m tools.client_model replay \
  --checkpoint runs/client-model/u20/bundle/actor.pt \
  --position-bundle runs/client-model/u20/bundle/position.json \
  --recording runs/client-observation/20261030T031618Z \
  --out runs/client-model/u20/replay-second-game.json
```

The benchmark uses native **simulator observations**, including hero selection,
recruitment and pending decisions, and reports the actual number collected before
the game ends. It measures inference and repeatability, not strength, live input
correctness or end-to-end screen latency. Replays report every incomplete input;
`complete_inputs: 0` means that no recorded frame was inferred.

`tools/client_action_replay.py` separately reads a complete raw recording and binds
each actual client input against the immediately preceding entity/options state.
It distinguishes bound actions, unresolved target decisions, unresolved modes,
shop rearrangements and same-position no-ops. A matched submission is not proof
that the server applied it.

## Recorded match and automatic regressions

See [2026-10-30 match review](CLIENT_MATCH_20261030.md) for the reproduced bugs,
scope, measurements and remaining limitations. The complete retained recordings
are private local inputs; the checked-in fixture contains reduced, sanitized
visible snapshots and their original source digest.

```bash
runs/model-env/bin/python -m tools.client_policy_replay \
  runs/client-observation/20261030T124300Z/Power.log --player-id 6 \
  --races UNDEAD ELEMENTAL QUILBOAR MURLOC DRAGON \
  --out runs/client-model/mapping-replay-3 --infer
runs/model-env/bin/python -m tools.client_stream_replay \
  runs/client-observation/20261030T124300Z/Power.log --player-id 6 \
  --races UNDEAD ELEMENTAL QUILBOAR MURLOC DRAGON \
  --out runs/client-model/mapping-replay-3/stream-report.json
runs/model-env/bin/python -m pytest -q tests/test_client_match_regressions.py \
  tests/test_client_model.py tests/test_client_stream.py tests/test_client_window.py
```

The first command checks every actual input boundary and every complete Options
packet, including the first menu of each turn. The second streams timestamped
writes with a deterministic clock, verifies quiet-EOF handling, and compares the
final entities, actions and public history against batch parsing. It requires no
game client, account or repeated human actions. The u19 pass covered 286 input
boundaries and 274 menus, all inferred. The continuation audit separately exercised
19,006 legal actions and 792 second/third-stage branches with no gaps, errors or
excluded final recommendations. A successful inference is not a strength assessment.

## Lifecycle and continuation contract

Only the newest queued snapshot is processed. Frame identity includes the log
session/game, snapshot and current Options/Choices state. New frames, stale logs,
mode changes and game completion invalidate in-flight results. The model process
has a bounded response timeout and exits when the companion closes. Parsing and
UI polling do not wait for inference. Speculative inference restores both position
and cancellation history, including on exceptions.

The adapter reconstructs current resources, ordered zones, choices, supported
growth counters and attachments, hero skins and previously observed opponent
boards. It never reads hidden opponent hands. Lobby tribes are bound to a game
identity, persisted across companion restarts, and supplied from the visible
hero banner or explicit confirmation. Power.log alone does not supply the list.

Options are interpreted by legal targets and mode semantics; their numeric order
is not stable. Locked hero offers remain in their displayed slots but are masked.
Targeted spells, powers and Activate use separate model decisions. A Choose One
spell with targets resolves both its mode and that mode's exact target list.
Cancellation removes the opening candidate and re-runs selection; it never picks
the first target arbitrarily. The report retains all excluded candidates.

Targeted/Choose One minions now use `tools/client_native/`, an isolated Rust bridge
to the unmodified training engine. It hydrates explicit public own-player fields,
preserves already-applied auras, and projects the summon before the continuation.
Only its own-player consequence is used; actual public opponent observations are
retained. `continuation_projection.observed=false` distinguishes this hypothetical
consequence from a log observation. Random transitions are marked explicitly.
The bridge checks its compiled source fingerprint against the loaded engine.

Run `bash script/build_client_native.sh` before backend-only testing; the desktop
launcher runs this automatically. `tests/test_client_continuations.py` compares
post-summon vectors/cards against native steps, including golden rewards, attached
scripts, Choose One, linked discard, Quest rewards and Timewarp leave/Chronum.

The adapter also handles explicit multiple-power links, secondary-race tags,
special Gem variants, paired/grouped hand cards, held cards, persistent effects,
and health purchases. Submitted operations are confirmed only after server evidence;
the operation list exposes the distinction. This does not certify universal future
patch coverage. Vision is tested with real OCR on generated banner pixels; live
capture permission and a new hero-selection screen still need field validation.

The model and its frozen native engine were not retrained or replaced by these
client fixes. Numerical encoder parity is checked against native observations
through a complete simulated game, separately from real-log mapping tests.

Checksums and numerical performance gates remain separate from these adapter
checks; see [MODEL_READINESS.md](MODEL_READINESS.md).
