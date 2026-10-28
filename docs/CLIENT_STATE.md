# Read-only client state

AutoTavern's simulator API does not read the Hearthstone client. The optional
`tools/client_state.py` bridge reads the client's `Power.log` with
[HearthSim's hslog parser](https://github.com/HearthSim/python-hslog).
It exports recorded entities, card IDs, numeric tags, controller/zone/position,
and the latest option and choice packets. Card names are joined from this
repository's catalog; missing cards remain explicitly unknown.

This is a diagnostic snapshot, **not yet a validated input to the trained model**.
Every output has `model_ready: false`. Shop ownership, Battlegrounds turn phases,
persistent effects, client-to-simulator action mapping, and current patch compatibility
still need verification against a real game. The installed app version is not proof
of the version that produced an older log. A snapshot is the last recorded state,
not proof that the client is currently running or still displaying that state.

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
runs/client-env/bin/python -m pytest tests/test_client_state.py -q
```

Passing those tests establishes parser behavior for the tested packet forms. It does
not establish compatibility with a complete current Battlegrounds session.
