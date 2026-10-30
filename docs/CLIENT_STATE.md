# Read-only client state

AutoTavern's simulator API does not read the Hearthstone client. The optional
`tools/client_state.py` bridge reads the client's `Power.log` with
[HearthSim's hslog parser](https://github.com/HearthSim/python-hslog).
It exports recorded entities, card IDs, numeric tags, controller/zone/position,
and the latest option and choice packets. It also records bounded, classified
client input history: buying, selling, playing, spells, upgrades, refreshes,
freeze toggles, board/shop position inputs, minion activation and choices.
An input packet is not proof that its effects completed. Card names are joined from this
repository's catalog; missing cards remain explicitly unknown.

This is a diagnostic snapshot, **not yet a validated input to the trained model**.
Every output has `model_ready: false`. Shop ownership, Battlegrounds turn phases,
persistent effects, client-to-simulator action mapping, and current patch compatibility
still need verification against a real game. The installed app version is not proof
of the version that produced an older log. A snapshot is the last recorded state,
not proof that the client is currently running or still displaying that state.

For a native macOS companion with a floating window, state review, simulator
action previews and human feedback, see [CLIENT_WINDOW.md](CLIENT_WINDOW.md).
Launch it with `./script/build_and_run.sh` or `打开对战助手.command`.

## Read-only macOS audit

The standard-library-only audit does not launch the client, enable logging, read
account files, inspect raw gameplay logs, or run a model:

```bash
python3 tools/client_audit.py --out runs/client/local-audit.json
# If discovery finds multiple installations or a custom location:
python3 tools/client_audit.py --client-app /path/to/Hearthstone.app \
  --logs /path/to/Hearthstone/Logs --out runs/client/local-audit.json
```

It checks conventional application locations (including mounted-volume roots),
`Info.plist`, the Power section of the macOS `log.config`, the client process, and
Power.log sizes/times. `--config` selects a different configuration file. A configured
`FilePrinting=true` is not evidence that game events were written. The report keeps
installed build, catalog version, unverified log/server rules and model readiness
separate. Reports belong in ignored `runs/`; do not commit real logs or account data.

## Run without compiling the simulator

Use Python 3.10 or newer in an isolated environment:

```bash
python3 -m venv runs/client-env
runs/client-env/bin/pip install hslog==1.20.0 hearthstone==9.22.0
runs/client-env/bin/python tools/client_state.py \
  --logs /path/to/Hearthstone/Logs --out runs/client/current.json
```

`--log /path/to/Power.log` reads a specific recording. `--watch` refreshes the local
JSON when a log changes or a newer session appears. On macOS, optionally pass
`--client-app /path/to/Hearthstone.app` to record the installed app build.
`--player-id N` overrides hslog's inferred friendly player. Inference is labeled as
unvalidated for the current game; no hard-coded assumption that player 1 is local
is made.

Logging must already be enabled before starting the client. The
[Hearthstone Deck Tracker logging guide](https://github.com/HearthSim/Hearthstone-Deck-Tracker/wiki/Setting-up-the-log.config)
documents that configuration. The reader does not launch or control the game and
does not change logging settings. Empty logs are reported as `empty_log`.

## Recording identity, freshness and missing information

`status` describes parsing, not permission to recommend an action. Each output has
`live_ready: false`, a `freshness` object, and `validation.issues` with stable codes
and Chinese explanations. The default `--max-age 30` marks an idle log stale after
30 seconds. A recent filesystem modification is only a recording-age check, never
proof of a matching live screen. The snapshot's `captured_at` does not renew the
log's observation time. Increasing the age limit is useful for offline replay and
does not establish live readiness.

`--watch` updates on file identity/content changes and when the age status changes,
including when no new bytes arrive. A newer named `Hearthstone_YYYY_MM_DD_HH_MM_SS`
session takes precedence even if its Power.log is absent (`missing_log`) or empty;
the reader never falls back to an old session in that case. Replacements or
truncations during a read invalidate that sample. Keep a stable, complete log before
interpreting its entities.

The watch command and native window now keep an `IncrementalReader` alive. It
consumes only appended bytes, retains partial trailing lines, and walks only new
parser nodes. Mutable entity/choice/option packets are revisited when they grow;
an unfinished block remains flagged. A complete Discover menu can be actionable
inside a block that is waiting for the user's answer. Mutable menus require a
following packet or a whole-line EOF quiet for 25 ms; idle polling can complete
this boundary without re-parsing the game. This is a bounded settling heuristic,
not a protocol end marker. Own state changes still invalidate Options; unrelated
opponent leaderboard/hero-power updates do not.
Session changes, file identity changes, truncation, same-size rewrites and changed
prefix/boundary checks rebuild the reader. This fast path assumes normal append-only
client logging; it is not a verifier for arbitrary edits in the middle of a log.
The default watch interval is 50 ms. `performance` separates bytes consumed,
parse time, total read time and cache hits. Initial full reads are separate from
incremental latency; filesystem detection and screen rendering add their own delay.

`unknown_card_ids` now contains unknown gameplay entities. Unmapped buttons,
drag targets and enchantments appear in `unmapped_internal_ids`; declared hero
skins appear in `unmapped_skin_ids`. Their original IDs and tags remain intact,
and their effect/skin mapping remains an explicit blocker. Classification does
not establish that an internal effect is supported. The pinned hslog workaround
for missing Battlegrounds block endings is retained as counted, redacted
`parser_diagnostics`, rather than re-logging the same warning on each update.
`game_status: complete` identifies a recorded game-over state.

Parsed samples carry a `snapshot_id` bound to the log identity, latest game bytes
and selected perspective. Re-reading the same bytes retains that ID even as they
become stale. Missing/invalidated options, stale choices, unknown card IDs and
unconfirmed perspective are explicit blockers. `last_options` and `last_choices`
are diagnostic client packets, **not a simulator action mask**. No code here derives
a policy observation in the reader itself. The separate
[model adapter](CLIENT_MODEL.md) now reconstructs supported public features and
binds model actions to these packets. The native window also has an independent
display projection for recorded player/shop/combat ownership. Reader validation
flags are not the model adapter's per-frame readiness result.

For a separate human-entry workflow, see [MANUAL_STATE.md](MANUAL_STATE.md).
It validates a visible record and lists missing fields without fabricating model
inputs. The [action renderer](MANUAL_ADVICE.md) accepts simulator recordings only.

## Validation before recommendations

Start with a complete log from the beginning of a game. Compare the exported hero,
hand, board, shop, resources, choices and targets with what the user can see, including
after buying, selling, discovering, targeting, combat and reconnecting. Preserve
entity identity and ordering; never substitute an unknown card with a similar card.
Resolve client/server rule differences before enabling model recommendations.
The user performs game actions manually.

The reader ignores incomplete trailing lines, flags open blocks and concurrent writes,
and reports parser failures without echoing raw log text. It parses only the latest game
and does not carry an earlier game's entities into a new session. Hidden card identities
remain unknown; account names and account IDs are not exported. Output files are local
and written with mode 0600. Raw logs, snapshots and models belong in ignored `runs/`.

Synthetic integration tests cover entity updates and hiding, options/targets/choices,
partial writes, open blocks, log rotation, unknown cards, version mismatch and redaction:

```bash
runs/client-env/bin/pip install pytest
runs/client-env/bin/python -m pytest tests/test_client_state.py tests/test_client_audit.py \
  tests/test_manual_state.py tests/test_manual_advice.py -q
```

Passing those tests establishes parser behavior for the tested packet forms. It does
not establish compatibility with a complete current Battlegrounds session.

For a retained private recording with `timeline.jsonl`, compressed snapshots and a
`recording-end.json` manifest, replay every original capture boundary:

```bash
runs/client-env/bin/python tools/client_benchmark.py \
  --recording runs/client-observation/<recording> \
  --out runs/client-performance/optimized.json
runs/client-env/bin/python -m pytest tests/test_client_stream.py -q
```

The benchmark compares incremental results with both batch replay and the original
recorded visible state at each boundary, and reports cold, idle and incremental
latency separately. Real logs stay local under ignored `runs/`.
