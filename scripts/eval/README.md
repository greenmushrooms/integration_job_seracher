# Eval Methodology

## What we're measuring
Which prompt variant gives qwen3:14b the best job-fit scoring for each profile?
"Best" = surfaces the right jobs in the right order, not necessarily perfect absolute scores.

## Pipeline
- Two-stage: extract (qwen3:8b -> `{title, summary}`) -> eval (qwen3:14b -> `{verdict, match_scores, ...}`)
- Current winner: **v3_fewshot** -- "honest career advisor" system prompt + 3 profile-specific synthetic examples (Pivot/Lateral/Step Up) inline in the prompt

## Ground truth
- 20 clean jobs per profile, labeled by Claude Opus 4.6
- Opus re-evaluated ALL 40 jobs (not just disagreements) to catch subtle mislabels
- Initial Sonnet labels had ~25% errors on edge cases -- always re-label the full set
- Ground truth is a proxy -- never validated against user's own judgment

## Metrics (in order of importance)
1. **Spearman's rho (rank correlation)** -- does the variant preserve the ordering? Primary metric.
2. **Pairwise concordance** -- % of job pairs ranked in correct order. More intuitive for small n.
3. **Match recall** -- did we surface Step Up + Lateral jobs above threshold? Operationally critical.
4. **Verdict accuracy** -- useful but misleading with Pivot-heavy sets (always-Pivot gets 50-60%).
5. **MAE** -- least important. Consistent bias is fine if rank is preserved.

## Mistakes to avoid
1. **Don't tune and measure on the same set.** Separate test/train or you'll overfit.
2. **Don't use MAE as primary metric for a ranking pipeline.** Rank preservation > score calibration.
3. **Don't trust ground truth labels without re-examination.** Re-labeling changed 8/20 Slava, 10/20 Kezia.
4. **Don't include intern/entry-level in metrics.** They're trivially correct and inflate accuracy.
5. **Extract fields matter.** Synthetic `domain` field was anchoring eval toward Pivot. Minimal `{title, summary}` was better.

## What worked well
- Profile-specific fewshot examples
- Separate extract and eval stages
- Go router's inline prompt override (Python passes prompt, no Go changes needed)
- Batch eval script with extract caching

## Production config
- Extract: qwen3:8b, `{title, summary}` only, num_ctx=16384
- Eval: qwen3:14b, v3_fewshot prompt (profile-specific), num_ctx=12288
- Both topics priority=5, scheduler batches by model

## Files
- `ground_truth_slava.json`, `ground_truth_kezia.json` -- 20 jobs each, Opus 4.6 labeled
- `agent_eval.py` -- standalone eval runner with all metrics

## Next time
- Extend to 50-100 jobs per profile for statistical power
- Add reliability test: run same prompt N times, measure variance
- Validate ground truth against actual user feedback
