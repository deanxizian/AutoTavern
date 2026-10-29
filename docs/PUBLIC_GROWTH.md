# Optional own-player growth observations

The legacy observation can hide persistent growth that matters when choosing a
summoning or scaling card. For example, without an Eternal Knight visible on the
board, in hand or in the shop, the same inputs can represent different accumulated
Eternal Knight death bonuses. A card identity or an attachment identity alone does
not supply that missing amount.

`growth_version=1` adds a fixed 16-value input to the entity policy, obtained only
from `Game.strategy_view(player).me.stats`. It does not read serialized game state,
opponent-private information or RNG. The legacy observation, native engine, actions,
rules and cancellation guard are unchanged. This extension is independent of the
original paper's encoder.

## Observation contract

The exact order is defined by `training.growth.FEATURE_NAMES`:

```text
eternal_knight_deaths, knight_bonus, undead_atk, undead_hp,
beetle_atk, beetle_hp, volumizer_atk, volumizer_hp,
elem_give_atk, elem_give_hp, tavern_elemental_atk, tavern_elemental_hp,
baller_bonus, lobster_bonus, magnetized, spellcraft_cast
```

All fields must be present as signed 32-bit integers in public-view schema 1.
Missing values fail explicitly; unknown information must not be substituted with
zero. Values are converted to float32 and transformed with signed `log1p` inside
the actor, like existing numeric features. Very large integer values therefore
have float32 precision, rather than an exact integer encoding. Other stats and all
opponent fields are ignored by this adapter.

These are the learner's accumulated counters, reconstructible in principle from
player-visible history or displays. A live client adapter still needs a separately
validated reconstruction of every field. Simulator support does not prove client
log coverage. This allowlist does not encode every persistent effect or remember
the complete action history.

## Policy and training

A bias-free `growth_proj` maps these values to the global token. It starts at zero
so that a checked weights-only transition can preserve the old policy's initial
outputs. The checkpoint architecture records the observation version; old policies
remain unchanged unless explicitly converted. Conversion must preserve the source
artifact and record its hash, retained parameters and prediction comparisons.

Use `--public-growth --model entity --opponents ... --warmup-samples 0` with matching
text embeddings for a fresh model, or an explicitly converted `--init-from` file.
`--growth-lr` optionally gives the new projection a separate learning rate. It can
coexist with `--attachment-lr`. A conversion starts a new stage; it is not an exact
resume of an old optimizer and environment. `--resume` requires the same growth
version and learning rate, then restores all normal training state.

PPO saves growth inputs at each decision and reuses those exact rows during every
minibatch update. Final observations retain the counters before an environment
reset when bootstrapping a training time limit. When expert advice is enabled,
advice and growth share one public-view read per row. Frozen opponents and both
evaluation drivers collect growth only for policies requiring it. These opponents
currently use eager inference. A growth-aware actor can use the existing public
position decoder, whose frozen encoder retains the legacy inputs.

## Verification boundary

`tests/test_growth.py` exercises the missing-information fixture, zero-residual
prediction equality, finite trainable gradients, checkpoint loading, row shuffling,
public-view reuse, league recovery, truncation bootstrapping, evaluation drivers,
position decoding, optimizer groups and a real PPO save/resume comparison.

Passing these checks establishes input and training compatibility. It does not
establish stronger play, better switching decisions or manual-test readiness.
Those require bounded training and the unchanged independent evaluation gates.
