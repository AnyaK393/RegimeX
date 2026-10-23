# RegimeX - Team Handoff

Read this first if you are picking the project up. Start with `RUN_ME.txt` to set up, then `docs/REPORT.md` (the story) and `docs/PHASE2_RESULTS.md` (all tables).

## Where the project stands

Done: data pipeline, causal regime labels for all five stocks, two trading environments, PPO training (single stock and pooled), baseline ladder, regime-stratified evaluation, bootstrap tests, walk-forward validation, SHAP, walk-forward hyper-parameter selection, report draft.

Headline result (honest): **no configuration shows a statistically supported out-of-sample benefit of the regime-conditioned reward** over an equal-average-risk-aversion regime-blind agent. The v3 position-sizing agents do adapt to regimes and roughly halve drawdown versus Buy & Hold, but a plain "hold 50 %" rule matches their Sharpe.

## Which script does what

| Goal | Script |
|---|---|
| Regime labels (all five stocks) | `src/build_multistock.py [TICKER ...]` (train-only KMeans + causal P90 rule). `src/regime_relabel.py` is the old RELIANCE path (full-sample KMeans); do not use it. |
| v1 env (BUY/HOLD/SELL) | `src/trading_env.py` |
| v3 env (target exposure, mean-variance reward) | `src/exposure_env.py` |
| Train | `src/train_ppo.py` (RELIANCE), `src/train_pooled.py` (five stocks; add `--exposure --scaled --gamma-base X` for v3) |
| Evaluate | `src/evaluate.py`, `src/evaluate_pooled.py`, `src/evaluate_v3.py` |
| Walk-forward / selection | `src/walk_forward.py`, `src/wf_select_v3.py` + `src/wf_select_report.py` |
| Explain | `src/explain_shap.py` (v1), `src/explain_shap_v3.py` |
| Tests | `src/test_regime_reward.py`, `src/test_exposure_env.py` |

## Rules that keep the results honest (please keep following them)

1. **Do not tune anything on the test window (2024-06-03 onward).** It has already been looked at several times. Choose settings on validation or on pre-2024 walk-forward folds, write the selection rule down first, and look at the test split once.
2. Compare adaptive and blind agents with **equal average risk aversion** (the blind agent uses the train-frequency-weighted mean gamma).
3. Report every grid point you ran, including the losers, and say when a result is not significant.
4. Always evaluate through the same environment (same costs) for every strategy.
5. Use several seeds; seed-to-seed Sharpe variation (about 0.1-0.3) is as large as most effects.

## Known problems and open ideas

* Regime-adaptive vs blind differences are small relative to noise; the test window (393 days/stock) is short. More history or more stocks is the most direct lever.
* Agents are over-cautious; constant 50 % exposure beats them on Sharpe in the test window. Ideas: a reward that targets Sharpe/Sortino directly, an exposure floor, or an ensemble over seeds.
* RELIANCE labels are now causal (train-only KMeans; 39 days changed) and the pre-registered v3 gamma 0.75 agents were retrained on them (`docs/PHASE2_RESULTS.md`, end of section 10). Conclusion unchanged; the earlier "adaptive de-risks more in High_Volatility" result did not replicate and is withdrawn. **Still on the old labels:** `models/ppo_regimex`, `models/ppo_pooled`, `models/ppo_pooled_v3` (gamma 3.0) (plus the non-causal `results/v3_select`). Their observations include the regime, so re-evaluating them now mixes old models with new labels; retrain before re-evaluating.
* Walk-forward is now strictly causal: `walk_forward.py` refits KMeans per fold, and `wf_select_v3.py --causal` / `wf_select_report.py --causal` do the same for the gamma grid (`results/v3_select_causal*`). The causal grid still selects gamma 0.75. `results/v3_select` (old labels) is kept for comparison only.
* Bootstrap p-values ignore seed variance (they resample days of the seed-averaged series). Use more seeds and report per-seed spread, or a seed-level test, for behavioural claims.
* High_Volatility is thin in validation (9 days for RELIANCE); use walk-forward folds for regime-specific claims.
* Possible extensions: LSTM / recurrent policies (the GPU could matter there; the installed PyTorch is the CPU build), transaction-cost sensitivity, other markets, a regime-transition feature, SAC/DQN comparison.

## Practicalities

* Run commands from the repo root; paths are relative.
* The final models (~5 MB) and the five `*_regimes.csv` files are tracked. Checkpoints, logs and raw data are not; they are regenerated.
* One batch of six 1M-step trainings takes about 35-40 min on 16 CPU cores (set `OMP_NUM_THREADS=1` and run one process per core). The GPU does not speed up these small MLP policies.
* Use your own git identity (`git config user.name` / `user.email`), pull before you push, and never force-push `main`.
