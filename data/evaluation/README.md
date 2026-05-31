# Legal QA Evaluation Dataset

`legal_qa_eval_30.jsonl` contains a small evaluation set for the current corpus.
Each line is one JSON object.

Core fields:
- `id`: stable case id.
- `category`: broad legal/domain group.
- `type`: evaluation pattern, such as `single_article`, `multi_article`, `table_lookup`, or `out_of_corpus`.
- `question`: user question to run through the RAG system.
- `expected_doc_ids`: document ids that should be retrieved/cited.
- `expected_articles`, `expected_clauses`, `expected_points`: expected legal structure when applicable.
- `expected_level`: expected chunk level, such as `article`, `clause`, `point`, or `table`.
- `expected_facts`: facts that should appear in a grounded answer.
- `must_cite`: whether the answer should include legal citations.
- `must_have_citation_url`: whether citations should include source URLs.
- `must_not_include`: terms that should not appear in the answer.
- `answer_policy`: expected behavior, usually `grounded`, `insufficient_data`, or `out_of_scope`.

Suggested checks:
- Retrieval hit: top-k contains `expected_doc_ids` and, when set, expected article/clause/point metadata.
- Answer facts: all `expected_facts` appear after normalization.
- Citation display: user-facing answer uses `[1]`, `[2]`, not `[S1]`.
- Citation grounding: every display citation maps to a stored `source_id` and URL when required.
- Refusal behavior: `insufficient_data` and `out_of_scope` cases should not invent legal answers from unrelated documents.
