# Evidence and reproduction record

This package supports the recommendation-centered ACL analysis in `section.tex`. It uses the existing method and evaluation configurations. No new metric, split, annotation scheme, or data conversion was introduced.

## Scope and metrics

| Setting | Questions | Native answer metric | Data origin |
|---|---:|---|---|
| SH-Doc QA | 100 | Substring exact match | RULER through MemoryAgentBench |
| MH-Doc QA | 100 | Substring exact match | RULER through MemoryAgentBench |
| FactConsolidation-SH | 100 | Substring exact match | MQuAKE-derived MemoryAgentBench |
| FactConsolidation-MH | 100 | Substring exact match | MQuAKE-derived MemoryAgentBench |
| LoCoMo | 1,986 | Released category-specific metrics | LoCoMo |
| 2WikiMultiHopQA | 1,000 | Native answer F1 | HippoRAG's released 2Wiki configuration |

These are three benchmark suites and four underlying data origins. All configured questions are retained. LoCoMo category 5 uses refusal correctness, so its overall score is a mixture of native category scores, not pure answer F1. The 2Wiki configuration has 6,119 passages and is not the original dataset's entire question collection. The study uses the existing seed-42 development evaluation. Four readers share one-shot source selections.

`experiments/report_memory_findings.py` recomputed 152,156 native answer scores across main conditions, LoCoMo context controls, and FC interventions. `data/verification.json` records completion. Native substring EM can credit an answer containing the gold substring without establishing full semantic equivalence; the case claims retain the native-score interpretation.

## Recommendation contrast

Full AMOR versus `without_propagation` holds context construction fixed. The latter uses dense source selection instead of graph recommendation and retains BM25 rank fusion and context augmentation. It is not the original dense baseline. `data/paired_outcomes.csv` contains all gain/equality/loss counts; `recommendation_effect.tex` displays the six native-score differences. LoCoMo's two negative Qwen differences remain visible.

## Matched weighting comparison

`experiments/compare_fact_weighting.py` uses the current simplified entity--source graph, frozen recognition, initial scores, corpus, damping, rank fusion, and five-source budget. Conditions are unit weights, original AMOR weights, and each combined with CatRAG's released or published key-fact boost. Every original undirected edge becomes reciprocal directed arcs. Original scores agree numerically and final source order matches the unmodified AMOR output for every question.

The released CatRAG operator is copied from `baseline_algorithms/CatRAG/src/catrag/CatRAG.py`, revision `ce06b459e2987a47342dfd23c71246c70d6de273`, lines 1825–1895. It multiplies supported entity-to-source arcs by 2.5, then preserves each entity's total outgoing source mass. Each memory-group pilot compares the transferred helper against execution of the unchanged released block. The separate paper condition uses the published multiplicative rule with beta 2.5, yielding a multiplier of 3.5 without this normalization. Neither condition changes reverse arcs. This is an operator transfer, not the complete CatRAG system; the concrete constructions differ in directionality, magnitudes, and normalization as well as which facts contribute.

All 3,386 selections are complete in `analysis/fact_weighting/complete.json`. Recognizer fallback questions remain in the task totals: 8/5/1/18/317/56 in the task order above. Cache guards prevent new extraction, recognition fallback caused by a cache miss, or new embeddings. An earlier tuple-normalization bug was detected and its invalid output preserved separately at `analysis/fact_weighting_invalid_tuple_normalization_6873706`; it is excluded from all tables.

`experiments/evaluate_fact_weighting.py` verifies exact current-AMOR and previously evaluated unit-weight contexts for every query before reusing those answers. It builds the two released-operator contexts using the existing renderer, then runs a native pilot and full evaluation for every reader/task/condition. The four full runs contain 27,088 answers, plus 48 one-question pilots. `experiments/report_fact_weighting.py` checks completed conditions and recomputes native scores. Completion must be established from `data/fact_weighting_answer_verification.json`, not inferred from job submission.

The report also counts identical rendered prompts with different scores on repeated generation. Case contents and ordering are checked, preserving LoCoMo's seeded answer-option order. All outcomes remain in the native task averages. Decoding follows the existing benchmark settings: temperature 0.7 for document/FC tasks, 0.4 for LoCoMo, and 0.0 for 2Wiki. These repeat variations are not interpreted as source-selection changes or silently removed.

All four reader runs and their pilots completed successfully. The report recomputed 40,632 answer scores across the reused unit condition and two new controls; the original AMOR scores were verified in the earlier 152,156-score audit. In the unit-to-AMOR comparison, 602 of 1,000 2Wiki prompts are identical, with 7/7/1/8 score differences across the four readers; 1,438 LoCoMo prompts are identical, with 41/43/11/51 score differences. The additional control pairs are reported in full in `data/fact_weighting_answer_pairs.csv`. These observations remain in the record; no adjusted metric or selective removal of repeated-generation differences is used. Source-selection completeness and the cached graph interventions are unaffected by answer-generation variation.

The graph-control native 2Wiki completeness of 46.8% is different from the full HippoRAG 2 pipeline's 46.9%. Removing pure synonym edges from the original graph gives 66.7%; it must not be omitted when interpreting the original-graph gap. The current simplified AMOR reaches 68.9%, not the older projected version's 69.1%.

## Sources, interventions, and interpretation

2Wiki source membership uses the original supporting-passage annotations. Among the 944 questions with accepted facts, direct-source gold coverage is already 1,259/1,260 for both unit and AMOR weights. The 88 gains/five losses among other source occurrences explain the observed incremental coverage; their exclusivity should not be portrayed as surprising despite that ceiling. Official question types are retained without new clustering or annotation. The film case was the first bridge-comparison completeness gain in original evaluation order; both completeness losses are also retained in the original case artifact.

The FC-MH intervention replaces the accepted current authorship with the actual older extracted tuple, keeping its confidence and graph fixed. The older tuple has reversed argument direction; no synthetic forward relation was invented. The corrected authorship stays in final context, while Eliot's death-location source disappears. Current/restored requests and answers agree within each reader. The population candidate comparison has all 100 questions and preserves the graph/context policy.

The FC-SH comparison preserves five original texts and the ten-fact budget while changing augmentation candidates. The single-case deletion removes only the appended UK assertion. It does not isolate whether an incorrect USA answer was caused by the obsolete source, model knowledge, or reduced correction salience. The contemplated old-source deletion experiment was not run. One Qwen4 population item changes score despite unchanged context on repeated generation; retain this variation instead of attributing every answer difference to treatment.

LoCoMo's 2×2 presentation comparison holds five central turns fixed and independently includes their original neighbors and selected facts. Native categories supply the populations. The Nate and Melanie interventions are inspected local cases; no automatic semantic failure labels are inferred from the remaining scores. Nine unresolved released evidence IDs are not repaired or dropped from native QA. The 104 isolated annotated-turn occurrences and their recovery counts concern resolvable answerable evidence occurrences, not 104 distinct questions or a newly defined answer metric.

## Artifact locations

Main results root: `/oscar/scratch/zliu328/agent-memory-outputs/optimization_simplified_amor_seed42_20260929`.

All question-level originals, predictions, traces, and existing controls are under its `analysis/` directory. Key files are `failure_questions.jsonl`, `memory_case_traces.json`, `recommendation_cases.json`, `recommendation_source_coverage.json`, `memory_interventions/`, `context_components/`, `context_controls/`, and `fact_weighting/`.

Small verified tables and plotted values accompany this package in `data/`. Figure generators are `experiments/plot_memory_recommendation.py` and `experiments/plot_memory_findings.py`. Rendering uses the official ACL style with unchanged margins and fonts. Review records are under `reviews/`; only a new final review of the completed artifact can approve the authorized email.
