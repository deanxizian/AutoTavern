# Audited 37.0 / 37.0.1 behavior changes

The bundled card vocabulary and default rules remain 36.6.3.253932. The optional
`{"hotfix_37_0_1": true}` configuration applies the confirmed changes below to that
vocabulary. This is a training rules profile, not certification that a live 37.0 client
can already supply complete model observations. See [CLIENT_STATE.md](CLIENT_STATE.md)
and [MODEL_READINESS.md](MODEL_READINESS.md).

The primary behavior sources are Blizzard's
[37.0 patch notes](https://hearthstone.blizzard.com/en-us/news/24301454/37-0-patch-notes)
and [37.0.1 server hotfix](https://us.forums.blizzard.com/en/hearthstone/t/3701-hotfix-patch/166622).
Client build numbers alone do not identify server-side hotfixes.

| Published behavior | Simulator coverage |
| --- | --- |
| Leeroy cannot receive Jaws of Death | Profile excludes normal and Golden Leeroy; legacy behavior retained when disabled. |
| Heroic Broodmother cannot receive Invulnerability | Profile excludes both forms. Other eligible minions remain eligible. |
| Tier 4 Dark Paradox cannot receive Toreth's Blessing | Explicit exclusion for the Tier 4 forms. They already lack the required Divine Shield in the pinned data; see the ambiguity below. |
| Hooktusk responds to Nellie's Ship Discover | Profile emits a separate crew-discovery event, once per completed pick. Normal/Golden Ship and Hooktusk are covered. No hand card is invented. |
| Recruit-phase Reborn stays consumed after combat | Regression follows recruit death through combat into the next recruit phase. |
| Mask of Ancient Ones preserves summoned Deity stats | Regression transforms a grown Deity, awakens it and checks its actual summoned stats. |
| Full Hoarding Hyena warbands finish combat | Normal/Golden boards, 32 seeds each; attack-limit termination is a failure. |
| Drek'Thar and Vanndar do not trigger twice | Regression covers locked/unlocked turns and immediate/delayed board space in a complete combat. |
| Gunpowder Courier shows the correct Gold remaining | Existing dynamic `shown` hook counts five Gold; regression checks spending through a reset. The historical static printed text still says six. |

These regressions are in `crates/taverngym/tests/patch_37.rs`. They are specific simulator
checks; they do not prove every possible interaction matches the live server.

## Card-data comparison and unresolved identity

HearthSim's client extracts were compared at fixed commits:

- [36.6.3.253932](https://github.com/HearthSim/hsdata/tree/27fbd2323aec20cab1c6ecc7d5f55cf12729f956)
- [37.0.0.255790](https://github.com/HearthSim/hsdata/tree/6ebd160cc0c685b10265f4d6bfe2ef69c3a0f226)

The union of `CardDefs.Bacon.xml` and `CardDefs.xml` covers all 3,983 simulator card IDs at
both commits. Comparing those entries found no changes to their combat stats, costs,
tiers or pools. English/Chinese changes are formatting/localization and Assemble-related
tags on four shared constructed cards. The only added Bacon entry is a Bob skin. This
comparison cannot reveal server scripts, so the published behavior fixes are audited
separately. Download SHA-256 values for reproducibility:

| Build / file | SHA-256 |
| --- | --- |
| 253932 / CardDefs.Bacon.xml | `c385e3d67bfc2c52408059fbb0399b41164c8d1b78d3dd1619a6fb1882116690` |
| 255790 / CardDefs.Bacon.xml | `a90627ec881d7be892e914ea51bde6681fabac3000afdb29f73ef02144e04d90` |
| 253932 / CardDefs.xml | `6d75f63414c9ea0c1edaad9e2549aae7322cc0fde0f7953748ceb9c91ac1f419` |
| 255790 / CardDefs.xml | `f951cda7eb91dced049ad515adb714bf787f32b1590d21a308578fa34807b9fa` |

The hotfix says **Tier 4 Dark Paradox**. Both extracts identify `BG36_360t5` (Golden:
`BG36_360_Gt5`) as Tier 4 with Rally and no Divine Shield. The different `BG36_360t4`
form is Tier 5 with a multi-hit shield. Do not infer tiers from ID suffixes, silently
move a card's tier, or broaden the exclusion to the Tier 5 form. Verify the relevant
live card ID/tier before claiming this ambiguity is resolved.

Only Hooktusk's response to ship crew discovery is confirmed by the hotfix. Crew has no
hand entity. Existing Sinstone/Burth behavior is preserved rather than manufacturing a
copyable hand card or consuming a hand-copy counter. Their live crew interactions still
need direct evidence.

## Training and checkpoint transitions

Pass `--hotfix-37-0-1` to `training.train` to select this profile. `training.arena` and
`training.evaluate` default to their candidate's training profile; use the same flag
to evaluate a transferred legacy policy under the new rules (`--no-hotfix-37-0-1` is an
explicit legacy comparison). The rule is stored in manifests, evaluations and full game checkpoints. Exact
resume rejects a change to the profile; every lobby uses the same rules for all seats.

An engine update changes the build fingerprint. Old optimizer/environment/RNG state
must not be resumed under the new engine. A policy may be reused only after recording
and comparing its observation contract:

```bash
# Before replacing the old native build, from its matching source checkout:
python -m training.transfer record --out runs/contracts/old-engine.json

# After rebuilding and validating the new engine:
python -m training.transfer weights runs/frozen/old-policy.pt \
  --source-contract runs/contracts/old-engine.json \
  --out runs/frozen/transferred-policy.pt --reason 'Audited 37.0.1 behavior changes'
```

The transfer checks card order/catalog/static features, every action name/index, numeric
layout and the source defining observations, actions and policy encoding. It preserves
the exact weights and source checkpoint/engine hashes. It removes all resume state and
marks the artifact as policy-only. Use it with `--init-from --warmup-samples 0`, or as a
frozen opponent, and start a fresh stage. New checkpoints retain their initial-policy
provenance. Evaluate all transferred opponents under the same chosen rules; previous
scores do not become current-rules evidence merely because the weights can load.
