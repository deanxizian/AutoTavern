# Native macOS companion window

The local SwiftUI app shows the client connection, recorded visible entities,
one current model instruction at a time. The default mode is
**真实日志**. An optional separate process loads a selected training model for
manual real-game guidance; the window never sends input to Hearthstone.

## Open the window

Double-click `打开对战助手.command` in the repository, or run:

```bash
./script/build_and_run.sh
```

The Codex **Run** action calls the same script. Requires macOS 15+, an Xcode Swift
toolchain and the parser environment from [CLIENT_STATE.md](CLIENT_STATE.md):
`runs/client-env/bin/python` with `hslog==1.20.0` and `hearthstone==9.22.0`.
The launcher also builds the isolated Rust public-continuation bridge; it reuses
the local Rust toolchain under `runs/model-build` when present. No Python or Swift
package dependencies are installed by the launcher.
Model setup and the optional `runs/client-model/selected.json` configuration are
described in [CLIENT_MODEL.md](CLIENT_MODEL.md). The log reader remains usable
without Torch. A configured window displays the selected model update and input
conversion coverage; incomplete inputs withhold inference and live instructions.
For a build without opening the app, use `--build-only`; `--verify` also reports
the local bridge launch. The app, build cache and local receipts are confined to
ignored `runs/client-window/`. The launcher stops/rebuilds only this companion app.
The staged local development bundle is ad-hoc signed and signature-verified after
its Info.plist is written; this does not constitute distribution notarization.
The GUI receives its temporary loopback connection file through a private system
temporary directory, removed when the bridge exits. It does not need to open the
external repository volume during window initialization; recordings remain in `runs/`.

The title-bar pin toggles floating-window behavior. The pinned window can accompany
other Spaces and full-screen apps via macOS auxiliary-window behavior. Drag the
title bar to place it beside the game; resize it for the available display. ⌘R
refreshes the diagnostic reading. The window follows system appearance.
The window polls every 50 ms; its backend parses appended bytes only. Identical
state revisions do not trigger repeated SwiftUI state replacements. The state
review lists recent observed client submissions, read timing, unknown gameplay
cards, internal entities awaiting mapping and counted parser compatibility fixes.
Recorded game-over state is shown as “本局已结束”. These are observations, not
recommendations or execution confirmations.
The menu's “紧凑窗口” toggle (⇧⌘K) switches between 360 × 520 and 420 × 700 content
sizes while keeping the upper-right corner in place and the window on its display.

Closing the window quits the app and its local service. Double-click the launcher
again to reopen it. It does not start, stop, restart or change the game client.

## Modes and evidence boundaries

- **真实日志** detects the running client and latest session, then follows actual
  Power.log changes. Missing, partial, old or unparseable recordings are labeled.
  HearthMirror supplies the current scene and confirms the local player when a
  matching game binding is available. If unavailable, the UI falls back to the
  last recognized `currMode` from LoadingScreen.log and labels it as a recorded
  scene. Card entities and their positions always come from Power.log.
  The card list groups the local player's hand/board and current visible
  shop or combat opponent using controller, player/hero linkage and combat phase.
  The shared dummy controller is never treated as the shop during combat. Inactive
  entities are omitted; unresolved choices remain visible. Unknown ownership stays
  in its own group, and opposing hands are excluded. This display projection
  is separate from the model adapter. The selected model supplies experimental
  live advice only for complete inputs and resolved, frame-matched actions.
  Remaining mechanic limitations are listed in [CLIENT_MODEL.md](CLIENT_MODEL.md).
- **操作演练** is explicitly labeled synthetic. Fixed purchase, play, freeze and
  wait steps exercise the window without a game, model or simulator build. These
  steps are UI fixtures, not strategy recommendations or current-patch evidence.
  Rehearsal is available only through the tools menu, not the primary live workflow.
- **模拟器回放** imports a selected legal action from a saved simulator view. Use
  the menu's “导入模拟器预览…” command with a local JSON object containing
  `scope: "simulator_action_preview"`, `view: <saved Game.view object>` and
  `action: <integer>`. Rendering uses [MANUAL_ADVICE.md](MANUAL_ADVICE.md), including
  its card, position, cost, target and action-mask checks. Unknown data blocks
  rendering. A diagnostic client snapshot cannot be substituted for a view.

Untrusted ready flags on a log/preview cannot enable advice. There is no endpoint
for injecting a real-client proposal. Only the selected model worker can produce
one, and its frame must still match the current game. Expired instructions do not
reappear on unchanged input. Source changes, stale logs and game completion clear
the instruction. Model loading does not certify model strength.

Lobby tribes are read automatically through the HearthSim-signed HearthMirror
framework in the locally installed HSTracker app. A separate native helper checks
the client process, log session and game handle; the backend retries and requires
two consistent reads before caching the five tribes for this log game. The UI
does not capture screens, use OCR, or request Screen Recording permission.
Manual entry remains in a collapsed disclosure as a fallback. The source and
reader status are available in the API, and failures are recorded locally.
See [CLIENT_RACES.md](CLIENT_RACES.md) for dependencies, permissions and validation,
and [CLIENT_DATA_SOURCES.md](CLIENT_DATA_SOURCES.md) for the source of each field.

## Recording a live session

The recorder follows logs independently of the window, including while the
companion restarts. It stores private raw logs, compressed model-input snapshots,
the actual window output and a SHA256 manifest under a new local directory:

```bash
runs/model-env/bin/python -m tools.client_record \
  --logs "/Applications/Hearthstone/Logs" \
  --out "runs/client-observation/my-session"
```

Use the log directory shown in the companion if the game is installed elsewhere.
The recorder stops after a completed game or on Ctrl-C. Replaced/truncated logs
receive separate archive files, and window metadata is attached only to snapshots
from the same game. Recordings and model weights stay outside version control.

## Feedback is not a game action

“已执行” / “未执行” is shown only for rehearsal/imported previews, not live play.
Live play follows newly observed game state without requiring these clicks.
Preview feedback records only what the person reports. Feedback binds to a
unique proposal and frame; duplicate, changed-mode or expired proposals are
rejected. Each preview expires after 60 seconds. Acknowledging it clears the old
instruction immediately and waits for another frame. No policy decoder or actual
game state is mutated. Receipts explicitly retain `execution_verified: false`.

In rehearsal, “载入下一帧” advances the synthetic example only after reported
execution. Declining or expiry reloads the same example with a new identity.
Imported previews require another explicit import. A disconnected UI hides the
instruction and disables feedback until a fresh read succeeds.

## Local interface

The Python bridge binds a random port on `127.0.0.1`, requires a per-launch token,
rejects browser Origin/foreign Host requests, limits input to 4 MB and never logs
request bodies or tokens. The connection file is mode 0600 and removed on exit.
Do not publish the service or share that connection file. The Swift app reads its
connection file automatically; the user does not configure API keys.

| Endpoint | Purpose |
| --- | --- |
| `GET /api/state` | Current diagnostic / preview state; no private opponent projection. |
| `POST /api/refresh` | Force another local read. |
| `POST /api/lobby-races` | Manually set five tribes for the exact `game_key`; cannot claim a memory source. |
| `POST /api/mode` | Switch between `live` and `rehearsal`; invalidate prior proposals. |
| `POST /api/preview` | Validate and display a simulator-only view/action recording. |
| `POST /api/feedback` | Record `proposal_id` plus `executed` or `not_executed`. |
| `POST /api/next-frame` | Load an explicit synthetic frame, never advance a real game. |

Run the dependency-free session/API regressions alongside the log tests with:

```bash
runs/client-env/bin/python -m pytest tests/test_client_window.py \
  tests/test_client_state.py tests/test_client_audit.py \
  tests/test_manual_advice.py tests/test_manual_state.py -q
```

Passing these tests and the native interaction checks establishes window and
recording behavior. It does not establish model strength or live-game compatibility.
