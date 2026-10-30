# Manual action instructions

`tools/manual_advice.py` converts one already selected simulator action into a
Chinese instruction with named cards, 1-based zone positions and visible entity
references. It is a delivery component for a future supervised trial. It does not
run a policy, control Hearthstone, or convert client logs into model observations.

Pass a saved `Game.view(seat)` JSON and an action that was legal at that exact
decision point:

```bash
python tools/manual_advice.py --view runs/example-view.json --action 8
```

The structured output contains `scope: simulator_action_preview` and
`live_ready: false`. A rendered instruction is evidence of action translation,
not evidence that the model or client adapter meets [MODEL_READINESS.md](MODEL_READINESS.md).
Positions refer to the simulator view; matching those positions to the real
client must still be validated. The input is a recording, with no freshness claim.
The diagnostic output of `tools/client_state.py` is always rejected, even if someone
changes its `model_ready` flag.

Each preview includes a `view_id` over the acting player's state, phase, turn,
version, action layout and mask, excluding opponents. To reject a recording changed
since an earlier preview, supply `--expected-view-id <previous-view-id>`. Reordered
cards, changed costs or a changed mask invalidate the preview. This ID binds a saved
simulator recording; it cannot verify live-client freshness. Every rendered result
has `execution_recorded: false`; printing it never acknowledges a user action.

Manual records use a separate typed schema and are also rejected, even if the
readiness flag is removed. See [MANUAL_STATE.md](MANUAL_STATE.md) for creating and
checking those records, finding card IDs and comparing captures after a human action.

The renderer distinguishes normal play from Magnetic, resolves targets by entity
identity in the player's own board/shop/hand, names both a quest and its reward,
and handles the separate Timewarped Tavern exit option. A partially frozen shop
is frozen, not unfrozen. EndTurn becomes “wait for combat”; it does not invent a
client button. Dynamic purchase prices explicitly distinguish Gold and Health.
Refresh instructions omit the cost because the native view's `refresh_price`
alone does not say whether Malchezaar charges Health. A live adapter must verify
that cost before supplying model advice; the renderer does not reconstruct rules.

Unknown cards, unresolved or ambiguous targets, unavailable actions, mismatched
action layouts and incomplete fields return `status: blocked` with no fallback
action. Choose One is deliberately blocked: the native view supplies two copies
of the source card, not the actual labels of the two modes. Real-client mode
identity/order must be verified before that branch can be enabled.

`Game.view` is a debug view that can include opponents' private current boards.
This renderer reads only its action layout/mask/phase and the acting player's own
fields, and exports only the references needed for the selected instruction. A
policy must continue to use the player-visible `Game.observe`; never pass the full
debug view as policy input or expose it as a public client snapshot.

In a future interactive adapter, bind a proposal to a specific current snapshot,
then read the resulting state after the user acts. Record cancellation-decoder
history only for the action actually executed. Merely rendering a proposal must
not mutate the game or mark that action as executed.

Run the dependency-free translation tests with:

```bash
python -m pytest tests/test_manual_advice.py -q
```
