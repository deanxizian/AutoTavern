# Conditional expert strategy and PPO

`training.strategy` applies a frozen, versioned knowledge bundle to player-visible
decisions. The bundle can describe hero selection priors, hero-specific leveling
curves, power timing and targets, composition entry requirements, purchases,
Discover choices, component retention, spell targets and positioning. The rules are
explicit engineering hypotheses; successful execution does not establish stronger play.

The current private research bundle combines the user's opening advice with the
116 hero guides and 31 composition guides recorded from HSReplay Tier7 on
2026-10-29. It lives under ignored `runs/research/expert-strategy-v1/`, alongside its
compiler, source references and coverage checks. Subscription source text and
tables are not required by the public test suite or included in the repository.

The user's ordinary curve is turn 2 level, turns 3 and 4 use the 3-gold Dark Gift
button, turn 5 level; early Fresh Perspective offers receive a preference. Dark
Gift, refreshing the shop and collecting the Aberration tribe are separate actions.
Hero-specific curves take precedence. Composition advice activates from owned
starting components and any specified economy, health, timing or growth conditions.
It does not activate just because a core card appears in the shop. Source passages
with no actionable advice or unresolved ambiguity remain marked in the bundle.

HSReplay hero averages provide bounded selection priors. Missing Top1 measurements
remain missing; the available Top50 and Top1 averages have equal weight when both
exist. Tribe averages are only small tie breakers after composition entry. These
aggregation choices, score weights and thresholds are hypotheses to test, not
claims that the source recommends those weights. They are never PPO rewards,
model win-rate estimates, or unconditional instructions to force a composition.

## Public information boundary

`Game.strategy_view(player)` returns schema version 1 JSON. It includes the player's
own resources, visible cards and effects, power progress, pending decisions and
their source card, plus public opponent scoreboard information and personally
observed board memories. It excludes seeds, RNG, configuration, combat replays and
current hidden opponent boards, hands and shops. Privileged diagnostic board views
are refused. Reading the accessor changes neither state nor RNG.

Only explicit public power histories are included: Cookie's own previously chosen
pot cards and Kurtrus's remaining purchase count. Hidden power memories, including
the answer to Holmes's guessing question, are never returned. Opponent guesses
must use the public opponent summary or remembered boards, never serialized games.

This accessor supplements the unchanged tensor and attachment contracts. Its first
implementation constructs human-readable live text and JSON and is relatively
expensive; measure end-to-end throughput before increasing the training budget.

## Training and inference

Use `--strategy-bundle /absolute/path/knowledge.json` with an entity league and
`--warmup-samples 0`. `--strategy-coef` defaults to `0.02`. An explicitly transferred
weights-only policy can initialize a new stage via `--init-from`.

The teacher produces a legal-action logit prior bounded to `[-6, 6]`. Advice is
soft: the learned actor can prefer another legal action. The game mask and
cancellation guard retain authority. The same prior is stored with each sampled
transition and reused after rollout flattening and minibatch shuffling, so PPO's
old and new likelihoods refer to the same advised policy. A small auxiliary loss
teaches the underlying actor the relative preferences; only completed placements
train the placement head, and the usual returns train the critic.

Checkpoints embed the entire immutable bundle and its canonical checksum. Loading
checks the checksum and native card table. Resume rejects changed knowledge or a
changed teaching coefficient; optimizer, games, RNG and rule application counts
are restored. A model requiring advice fails if the input prior is absent. Legacy
policies receive no extra inputs and retain their previous behavior. Position
wrappers pass advice to the actor while their frozen encoder remains unchanged.

Adding the native accessor changes the build fingerprint. `training.transfer`
continues rejecting changed source contracts. A separately evidenced transition
must verify unchanged gameplay, legacy observations, actions, card vocabulary,
weights and predictions, preserve the old artifacts, and write new weights-only
files with parent hashes. It must never transfer the old optimizer or environment
as an exact resume. A ridge encoder binding also requires an explicit transition
with unchanged coefficients and matching position scores.

## Verification and strength evaluation

`tests/test_strategy_view.py` checks state/RNG purity, hidden information isolation,
visible progress and real pending-source mapping. `tests/test_strategy.py` checks
opening exceptions, owned-component entry, effect identity, target entity mapping,
legal masks, PPO likelihood reuse, learning gradients, knowledge checksums and exact
save/resume through real training updates. Private coverage probes additionally
exercise all 116 hero profiles and 31 composition entry conditions.

Use complete, balanced, guarded development games to compare the base actor, advice
without further learning and PPO with advice. Keep the same opponents, seeds and
other decoders. Log the bundle checksum and actual triggered rules. These results
remain development evidence; the independent requirements in
[MODEL_READINESS.md](MODEL_READINESS.md) still apply before inviting a manual trial.
