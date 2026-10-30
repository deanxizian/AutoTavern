# Manual visible-state records

`tools/manual_state.py` prepares a local, human-entered record and checks missing
fields, card identity, ordered positions, targets, prices and observation time.
It uses only the Python standard library and the existing card catalog. It does
not import TavernGym, run a policy, operate Hearthstone or translate a recording
into model observations. Preparation and synthetic replay can be done offline.
This work does not require starting a real game before the model passes its gates.

## Create, fill and check

```bash
python3 tools/manual_state.py template --out runs/client/manual-01.json
python3 tools/manual_state.py lookup '人气选手'
python3 tools/manual_state.py check runs/client/manual-01.json \
  --out runs/client/manual-01-check.json
```

`template` creates a new file with mode 0600 and refuses to overwrite an existing
file. Leave unknown values as `null`; use `[]` only after checking that a list is
empty. Copy each zone from left to right. Fill `observed_at` with the actual time
the screen was checked, including timezone, for example `2026-10-30T09:00:00+08:00`.
Do not update that timestamp merely to make an old record pass. A template does not
claim that observation has taken place.

`lookup` searches exact catalog IDs, Chinese/English names and substrings. It lists
all matches up to 20 and reports the total. A name can have several IDs; confirm the
right normal/Golden form. Catalog membership is not proof of current-patch support,
availability in the pool or server behavior. Unknown IDs stay unknown. Printed
base stats and costs are not copied into the record automatically.

The template fields are:

| Field | What the human records |
| --- | --- |
| `capture_id` | Keep the new template's ID; use a new template for each observation. |
| `client_build` | Installed client build; server hotfixes remain unverified. |
| `phase`, `turn` | `hero_select`, `recruit`, `combat` or `game_over`, and visible turn. |
| `hero` | Card object for the selected hero; null before hero selection. |
| `resources` | Visible health, armor, gold and tavern tier; null is allowed before hero selection. |
| `zones` | Ordered `board`, `hand`, `shop` arrays; positions are implicit, starting at 1. |
| `hero_powers` | Visible powers and their current price/currency, including passive powers if displayed. |
| `visible_effects` | Exact visible text for quests, rewards, trinkets, counters and other persistent effects; do not guess hidden state. |
| `pending` | Current selection kind, ordered card options and/or visible targets. |
| `legal_actions` | Human-confirmed visible operation labels, with exact source/target/position/cost where relevant. This is not a 245-action mask. |
| `confirmed` | Set true only after comparing all entered information to the visible screen. |

A **synthetic format example**, not a current game or recommended purchase:

```json
{
  "card_id": "AT_121",
  "attack": 9,
  "health": 12,
  "golden": false,
  "frozen": false,
  "cost": {"amount": 2, "currency": "health"}
}
```

Put such a card in the appropriate zone. `card_id` is required. Minions require
observed `attack`, `health` and `golden`. Shop cards additionally require `frozen`
and `cost`; currencies are `gold`, `health` or `chronum`. Heroes can be just
`{"card_id": "TB_BaconShop_HERO_01"}`. Hero powers require `card_id` and `cost`.
Optional `visible_text` records exact on-screen text. Optional `entity_id` must be
an actual known log identity; omit it if unavailable. Do not invent an entity ID
from the card position. The local capture ID and ordered position identify manual
entries without pretending to know the client's internal entity IDs.

No selection window is recorded as:

```json
{"kind": "none", "options": [], "targets": []}
```

Supported selection labels are `hero`, `hero_power`, `discover`, `target`,
`choose_one`, `trinket`, `quest` and `timewarp`. Options are ordered card objects.
Choose One entries must include the actual mode `label`; repeating the source card
name does not establish mode identity. A target is an explicit reference such as
`{"zone": "shop", "position": 2}`. Targets outside the entered visible zones
remain unsupported. This is a limited capture schema: modes, quest/reward relations,
Timewarp exit controls and other mechanics still need full model/action mapping.

## Interpreting validation

Exit code 2 and `status: needs_input` report missing, inconsistent, unknown or stale
information. Each issue includes its field path, stable code and a Chinese prompt.
No default card, target, cost or fallback action is supplied. A malformed file is
`blocked`; parser errors do not echo the original content.

`record_complete` means the implemented record checks passed. It does **not** certify
screen accuracy, full mechanic coverage, current legal actions, model readiness or
strategic strength. The validator cannot inspect the screen. Both `model_ready` and
`live_ready` always remain false; manual-to-model conversion, action mapping, server
rules and model admission remain explicit blockers. Do not pass this JSON to
`Game.view`, `Game.observe` or the action renderer.

The default maximum age is 30 seconds. Missing/timezone-free timestamps, timestamps
more than 2 seconds in the future, and expired observations block the record.
`--max-age` can widen the replay window for an offline fixture; it never authorizes
live advice. The check's `state_id` hashes the entire entered record. Passing
`--expected-state-id <prior-id>` detects any subsequent edit, including a changed
card order, while retaining the separate freshness check.

## After a manually performed operation

Once a supervised trial is separately authorized, the human remains responsible
for every client operation. Keep the before record and create a new template for
the actual visible after-state:

```bash
python3 tools/manual_state.py template --out runs/client/manual-02.json
# Fill the new observation before checking it:
python3 tools/manual_state.py check runs/client/manual-02.json \
  --previous runs/client/manual-01.json --out runs/client/manual-02-check.json
```

A reused capture ID, non-increasing time, malformed earlier record, or no observed
state change blocks progression. A valid older record may be stale because it is
historical; the new record must pass its own age check. `changed_fields` identifies
which groups changed, and `execution_verified` remains false. A changed screen
does not prove the proposed action happened: the human may cancel, perform a
different operation, or observe an automatic combat transition. No cancellation
decoder history or policy state is updated here.

## Interface still needed from the future model integration

The model side must supply a frozen, admitted model/rules/vocabulary contract and
an actor that consumes only a validated player-visible observation plus a current
legal-action mask. The adapter must establish client entity/card/zone mappings,
phase, full persistent-effect state, costs and choice/target order before creating
that observation. A proposal must carry the exact capture ID and observation hash;
after the user acts, compare a new validated observation and acknowledge the action
actually performed before advancing decoder history. These requirements do not
change training code or constitute an implementation of the missing conversion.

See [CLIENT_STATE.md](CLIENT_STATE.md), [MANUAL_ADVICE.md](MANUAL_ADVICE.md) and
[MODEL_READINESS.md](MODEL_READINESS.md) for the separate evidence boundaries.
