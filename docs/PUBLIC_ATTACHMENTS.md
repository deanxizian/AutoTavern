# Optional public attachment observations

The legacy observation records attachment counts but not their card identities. Two
minions can therefore have identical policy inputs while their attachments cause
different effects on the next action. `Game.attachment_observation(player)` adds the
missing identities without changing `observe()`, legal actions, card indexing or
gameplay rules. This is an independent baseline extension, not the paper's encoder.

## Native contract

The accessor returns two little-endian int32 byte arrays: offsets and card indices.
There are `OBS_CARDS + 1` offsets. The slice `indices[offsets[i]:offsets[i+1]]`
belongs to slot `i` in the unchanged observation's card array. Card indices use the
same vocabulary as `observe()`. Empty and non-minion slots have empty slices.

The accessor includes the player's board, hand, shop, visible Discover/Timewarp/Target
options, and remembered opponent boards in the same order as the legacy observation.
It never exposes current hidden opponent boards, hands or shops. Privileged debug
board views are refused. Reading it does not change state or RNG. Duplicate
attachments and source order are preserved; there is no silent fixed-length cap.
An int32 overflow fails explicitly.

`training.attachments.AttachmentRows` stores counts and identities on CPU, preserves
row order through rollout concatenation and minibatch shuffling, and constructs
int64 CSR tensors only for the selected policy batch. Policy inputs come from the
native accessor, not serialized games or debug views. Save files are used only to
restore the simulator or create verification fixtures.

## Policy and training

An entity architecture with `attachment_version=1` requires these extra inputs.
Its bias-free `attachment_proj` transforms the existing identity/static/text card
representations, averages each attachment bag and adds the result to its host card
representation. Existing count features retain multiplicity information. This
version does not encode attachment order, per-effect counters or other persistent
effect identities. The pooling implementation uses
[`embedding_bag`](https://docs.pytorch.org/docs/2.12/generated/torch.nn.functional.embedding_bag.html).

The new projection starts at zero, allowing a verified weights-only transition that
preserves all old parameters and initial predictions. Training uses
`--public-attachments --model entity --opponents ... --warmup-samples 0`. Supply
matching text embeddings for a fresh model or an explicitly converted checkpoint
with `--init-from`. Existing legacy checkpoints remain legacy models by default.
`--attachment-lr` optionally gives only the new projection its own learning rate;
all existing parameters retain `--lr`. Record both rates before a bounded experiment.
`--resume` requires the same observation version and restores optimizer, environment
and RNG state, including the same attachment optimizer groups and rate. Changing an
engine fingerprint is never an exact resume.

League opponents and both evaluation drivers supply extra inputs only to policies
that require them. Ragged-input opponents currently use eager inference even when
fixed-size CUDA graphs are enabled for legacy opponents. The public position decoder
can use an attachment-aware actor, while its frozen ridge feature family retains
the exact original legacy encoder. The cancellation decoder remains version 1 and
continues hashing the legacy public observation and mask.

## Verification and transition boundaries

`tests/test_attachments.py` covers information boundaries, pending-choice ordering,
lossless ragged storage, real effect differences, zero-residual output equality,
finite gradients, opponent inference, truncation bootstrapping, both evaluation
drivers, position decoding, and exact PPO save/resume equivalence.

Adding a native API changes the build fingerprint. `training.transfer` deliberately
continues rejecting this source-contract change. Any explicit migration must retain
the original artifacts, prove that gameplay and legacy observation contracts are
unchanged, compare actual policy predictions, record parent hashes and create new
weights-only files. A position ridge artifact also requires an explicit, documented
encoder binding transition with unchanged coefficients and verified feature/score
equality. Never relabel an old checkpoint, optimizer state or ridge training run as
having been produced on the new engine.

Compatibility tests and finite gradients do not establish stronger play. Use a
bounded training experiment and unchanged development/readiness gates before any
promotion or manual trial. Client log integration remains a separate validation.
