# W&B analysis methodology

## Purpose and module ownership

The analysis is deliberately split by responsibility:

| Module | Responsibility |
| --- | --- |
| `fetch.py` | Export run metadata, config, summary, and full scalar history from W&B. |
| `loader.py` | Validate and load a local export without contacting W&B. |
| `processing.py` | Normalize nested configs, resolve duplicate runs, calculate stable gaps, and select test/best-validation metrics. |
| `sanity.py` | Check matrix coverage, completion, finite training signals, validation evidence, and feasibility. |
| `comparisons.py` | Compare decoders within problems, aggregate across problems, make paired seed comparisons, and evaluate the recurrent hypothesis. |
| `visualization.py` | Create coverage, sanity, learning-curve, decoder, and hypothesis figures. |
| `results.py` | Write reusable CSV, compressed history, and JSON artifacts. |
| `report.py` | Turn the artifacts into a human-readable Markdown report. |
| `pipeline.py` | Orchestrate the pure local analysis stages. |
| `cli.py` | Expose independent `fetch`, `analyze`, and combined `run` commands. |

This boundary keeps cloud access out of data processing and makes every result
reproducible from the snapshot in
`~/local_db/compare-architectures/outputs/wandb_analysis/raw`.

## Experimental sanity check

Sanity has three separate questions:

1. **Coverage:** Is every expected problem/decoder/mode/seed/scale cell present
   exactly once, and is a finished run available?
2. **Execution health:** Did the run reach its configured epoch count, and are
   its training and validation metrics finite?
3. **Learning evidence:** Does held-out quality improve, and are evaluation
   solutions feasible?

Supervised and RL runs should not use the same convergence rule:

- Supervised learning checks that the final-window supervised loss is lower
  than the initial window, while validation quality is treated separately.
- RL checks reward and held-out quality. Policy-gradient loss is required to be
  finite but is not required to decrease monotonically because its scale and
  sign change with sampled actions, advantages, and baseline updates.
- Final held-out feasibility below `0.999999` is a failure by default. An
  intermediate train/validation evaluation below the threshold is a warning if
  final held-out outputs recover to full feasibility. The threshold is
  configurable from the CLI.

Warnings such as a flat validation trajectory should trigger inspection, but
they are not conflated with missing history, incomplete runs, non-finite
metrics, or infeasible outputs. Failed and incomplete runs remain visible in
the output tables but are excluded from decoder performance comparisons.

## Performance metric

Test metrics from the restored best-validation checkpoint are preferred. When
test metrics are missing, the analysis selects the best validation row and
marks the source as `val_best`.

When solver reference objectives are logged, the comparable within-problem
metric is the stable aggregate reference gap:

\[
\operatorname{gap}_{\%} = 100
\frac{\operatorname{regret}(\bar y, \bar y^*)}{|\bar y^*|}.
\]

For minimization, regret is \(\bar y-\bar y^*\); for maximization it is
\(\bar y^*-\bar y\). This ratio of aggregate objectives is preferable to an
average of per-instance percentages when individual references can be close to
zero. The analysis negates the gap so its internal `quality_value` always has
the convention “higher is better.” If references do not exist, it uses the raw
objective with its direction corrected.

New runs use `reference_objective`/`reference_gap`; legacy
`target_objective`/`optimal_gap` histories are normalized on import. The run
fingerprint includes the resolved configuration, dataset samples, and source
state. Exact reruns are deduplicated by that fingerprint, while different
configurations occupying the same matrix cell are reported as
`configuration_variant` and excluded from comparisons rather than silently
selecting the newest one. Cross-decoder and cross-problem tables also group on a
comparison-condition fingerprint derived from the training budget, common model
controls, parameter-budget regime, and source fingerprint. Fixed-encoder and
total-parameter-matched runs therefore cannot be pooled accidentally.

Raw objective or gap values are never averaged across different problem
categories. Each matched seed-level problem cell first receives a decoder rank
and z-score. Cross-problem summaries then average those normalized values with
equal weight per problem.

## Decoder comparisons

Two views answer complementary questions:

- **Within each problem:** decoder mean, seed variance, stable gap, feasibility,
  normalized score, and matched-seed pairwise differences.
- **Within each decoder:** its equal-weight rank/standardized profile across all
  problem categories, split by training mode and scale.

Pairwise rows are matched on problem, mode, scale, seed, and encoder. The CSV
reports effect size, seed wins/ties/losses, a paired t-test, and a Holm-adjusted
p-value within each problem. Three seeds provide limited inferential power, so
effect size and consistency matter more than a thresholded p-value.

## Predeclared recurrent-decoder hypothesis

The hypothesis is operationalized before looking at outcomes:

- Recurrent decoders: `lstm_pointer`, `gru_pointer`.
- Nonrecurrent autoregressive comparators: `attention_model`,
  `attention_model_without_glimpse`, `transformer_pointer`.
- Full-topology problems: TSP and CVRP.
- Partial-selection problems: orienteering, knapsack, MIS, maximum clique, and
  minimum vertex cover.

By default, `sigmoid_subset` remains in coverage, sanity, and descriptive
comparisons but is excluded from this test because it is an independent subset
policy rather than an autoregressive decoder. Analyses using
`--exclude-decoders sigmoid_subset` remove it from every analysis surface.

Only cells containing all five hypothesis decoders are used. Within each
problem/mode/scale/seed/encoder cell, quality is standardized across the five
decoders. The recurrent advantage is the mean recurrent z-score minus the mean
nonrecurrent z-score. The reported interaction is:

\[
\overline{A}_{\text{partial selection}} -
\overline{A}_{\text{full topology}}.
\]

A positive value supports the proposed direction. The report includes a
problem-bootstrap interval and an exact permutation test over problem labels.
With seven categories split 5 versus 2, only 21 label assignments exist, so the
p-value is coarse.

This experiment cannot isolate solution scope causally: partial selection is
confounded with graph structure, encoder family, feasibility constraints,
problem semantics, and sometimes output length. A stronger follow-up design
would add controlled tasks that vary solution density while holding the
topology, encoder, data budget, and objective family fixed. Plot recurrent
advantage against actual selected-node fraction as a continuous variable rather
than relying only on a binary problem label.

## Result and report directories

Generated artifacts live under
`~/local_db/compare-architectures/outputs/wandb_analysis/results`:

- `coverage.csv` and `figures/coverage.png`
- `run_sanity.csv` and `figures/sanity_status.png`
- `final_metrics.csv`
- `decoder_by_problem.csv` and its heatmap
- `decoder_across_problems.csv` and its bar chart
- `pairwise_decoder_comparisons.csv`
- `hypothesis_problem_contrasts.csv`, `hypothesis_tests.csv`, and the hypothesis
  figure
- mode-specific learning curves
- `history.csv.gz` and `analysis.json`

The human-readable report is written to
`docs/W&B Architecture Comparison Analysis.md`. Its table and figure links use
absolute artifact paths, so moving the report away from the generated result
directory does not break them. Use `--report-path docs/<name>.md` to retain
separate reports for different experiment batches.

Do not treat missing plots as negative results. They mean the required metric or
complete matched comparison was absent from the export; the corresponding CSV
and sanity table identify that data-availability issue.

When analyzing a tag-scoped export that intentionally contains only one training
mode, pass `--expected-modes supervised` or `--expected-modes rl`. Coverage then
checks the intended tagged matrix instead of counting the other mode as missing.

Pass `--exclude-decoders <decoder> [...]` when a decoder must be removed from
the analysis itself. The exclusion applies to run selection, coverage, sanity,
history, final metrics, comparisons, reports, and all figures. It does not
delete or alter the raw W&B snapshot, so the batch can be reanalyzed later with
a different decoder scope.

Use `--exclude-problems <problem> [...]` for the same full-pipeline exclusion
by problem. For example, `--exclude-problems orienteering` removes
orienteering from every result table, statistical comparison, report section,
and figure without changing the raw export.
