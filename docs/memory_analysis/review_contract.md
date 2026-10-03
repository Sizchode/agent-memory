# Analysis revision requirements

The user clarified the governing claim: **recommendation is the key to solving agent memory tasks**. Organize the analysis around what memory tasks require and why recommendation supplies useful evidence; implementation details are supporting mechanisms, not the story's main findings.

## Independent review

- Every critique round must use a newly spawned GPT-6 reviewer with no conversation history.
- Do not reuse a reviewer for another critique round or send a previous review to a new reviewer.
- Provide the current manuscript, original evidence, the user's requirements, and primary-literature access. Ask the reviewer to attack novelty and causal support independently.
- Earlier follow-ups to `acl_critic` do not satisfy this fresh-review requirement. Keep their record, but do not describe them as independent rounds.
- Fresh rounds used independent GPT-6 Astra agents: `acl_recommendation_review_02`, followed by `acl_manuscript_review_03`; neither inherited turns or previous review files. Round 03 accepted the current manuscript, with final artifact requirements subsequently verified.
- Passing requires 2–3 independently supported findings centered on recommendation, quantitative and qualitative evidence across the required data, a readable figure/example table, and approximately 2.5 ACL pages including figures and tables. Each finding must connect task requirements, recommendation's role, population results, and a concrete example or controlled intervention. Do not count a limitation as an extra finding or split one finding to inflate their number. A reviewer's preference does not authorize changing data, metrics, or the implemented method.
- Send the authorized email to zhenkeliu@163.com only after a fresh final critic passes the completed artifact.

## Evidence and scope

All six configured evaluation sets remain in the analysis: SH-Doc, MH-Doc, FC-SH, FC-MH, LoCoMo, and 2Wiki, totaling 3,386 questions. They comprise three benchmark suites and four underlying data origins. Use native metrics and released configurations; do not present the two FC variants as independent benchmark families.

`recommendation_effect.tex` presents all six recommendation contrasts; its numbers come from the audited `data/paired_outcomes.csv`. Graph-weight controls, fact-selection controls, and context-presentation controls have different interventions and must retain those distinctions.

The already submitted all-six weighting comparison continues as supporting evidence. The proposed deletion of the old USA assertion has not been implemented or submitted; the user corrected the research direction before that experiment began. Existing FC-SH results establish the appended correction's effect, not that the old source caused the wrong answer.

## Deliverable

Rebuild `section.tex` around the recommendation claim, using task-level quantities and source-verified examples. The current module-centered draft is superseded. Preserve raw results and adverse outcomes. Keep formula/operator detail in the supporting record when it does not explain the core finding. Final review must inspect the actual rendered ACL layout, not just a word count.
