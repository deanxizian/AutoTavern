# Current experiment overview

Current validated release in `deanxizian/eric-3702`: **V8**, internally C007 best75, full Adam11600. Eric upstream historical V8/V9 remain separate.

- [V8 report and exact opponent/mode/game counts](20261110-eric-v8-3702/README.md)
- [Numerical summary and raw-evidence SHA values](20261110-eric-v8-3702/summary.json)
- [Actual bounded training recipe](20261110-eric-v8-3702/training-recipe.json)
- [Action/cycle/forced-End distributions](20261110-eric-v8-3702/behavior-summary.json)
- [Model lifecycle and checkpoint SHA](../configs/eric-3702-models.json)

The stage completed100PPO/13,107,200 new samples/1600 newAdam; best is75PPO/1200 newAdam, cumulative11600. Independent test:2048 complete games, sampled768/model and greedy256/model, same2V1+2V2+2V7+V0. Original mean-placement gates passed; test did not select the checkpoint.

Historical configurations and reproduction source needed by regression tests remain archived here. Complete original experiment records remain unchanged in [Eric baseline719a349](https://github.com/firefighter-eric/AutoTavern/tree/719a3491eb0b95c16f5d028d549d944cbc7cd286/experiments), its Git history and the original server checkout. The publication omits upstream raw curves, per-game traces, process state and W&B data. No original server records or models were deleted.
