# M4 backend comparison

> **Draft: Pinecone hybrid pending.** The Pinecone hybrid run hit Pinecone's monthly read (egress) limit and is not included. It will be rerun, using the new local BM25 cache, once the limit resets.

Built from existing result files only; no new runs. All five runs use the same golden set (sha256 `2cbd9cd909ad…`, 20 questions), agent/judge model `gpt-5.4`, top_k=5, max_retries=2.

| Config | Answers passed | Retrieval passed | Mean latency (s) | p50 latency (s) | Total tokens | Cost (USD) | Rerank searches |
|---|---|---|---|---|---|---|---|
| Weaviate dense | 19/20 | 14/17 | 8.54 | 6.38 | 293,474 | $0.9010 | 0 |
| Weaviate hybrid | 19/20 | 16/17 | 9.29 | 6.25 | 299,359 | $0.9196 | 0 |
| Weaviate hybrid_reranked | 19/20 | 16/17 | 10.04 | 7.26 | 309,367 | $1.0124 | 30 |
| Pinecone dense | 19/20 | 14/17 | 9.00 | 6.65 | 278,283 | $0.8349 | 0 |
| Pinecone hybrid_reranked (baseline v2 run 1) | 19/20 | 16/17 | 10.14 | 6.77 | 309,191 | $0.9875 | 30 |

Sources:

- Weaviate dense: `eval_results/compare_weaviate_dense.json` (run `2026-10-14T12-11-37Z_weaviate_dense`, commit `3881c65`)
- Weaviate hybrid: `eval_results/compare_weaviate_hybrid.json` (run `2026-10-14T12-15-30Z_weaviate_hybrid`, commit `3881c65`)
- Weaviate hybrid_reranked: `eval_results/compare_weaviate_hybrid_reranked.json` (run `2026-10-14T12-19-37Z_weaviate_hybrid_reranked`, commit `3881c65`)
- Pinecone dense: `eval_results/compare_pinecone_dense.json` (run `2026-10-14T12-01-28Z_pinecone_dense`, commit `3881c65`)
- Pinecone hybrid_reranked (baseline v2 run 1): `eval_data/baselines/baseline_v2_pinecone_hybrid_reranked_run1.json` (run `2026-10-14T10-44-23Z_pinecone_hybrid_reranked`, commit `ac0864e`)

Notes:

- Total tokens = agent prompt + agent completion + judge prompt + judge completion (same as `src/eval/aggregate.py`).
- Cost is the run's `cost_usd.total` (agent + judge + rerank). The baseline costs were recomputed with the same pricing as the compare runs (commit `3881c65`).
- Retrieval is scored only on questions with a retrieval verdict (17 of 20).
- Pinecone hybrid is not included: 15/20 questions in `compare_pinecone_hybrid.json` failed with a Pinecone egress-limit 429, so that run is not a valid result.
- The baseline is the first of three runs; across all three, baseline v2 scored 19/20 every time.

## Questions whose verdicts differ between configs

### q09: What are Constellation Energy's largest growth initiatives disclosed across its 10-Ks for fiscal years 2024 and 2025, and what is the approximate dollar scale of each?

Differs on: Retrieval

| Config | Answer | Retrieval | Judge reason |
|---|---|---|---|
| Weaviate dense | FAIL | FAIL | **Retrieval:** The retrieved chunks do not include the FY2024 disclosure of the Crane Clean Energy Center / Three Mile Island Unit 1 restart or its dollar scale, and they also do not include the FY2025 disclosure of the Calpine acquisition or its dollar scale. The chunks mainly show FY2023/FY2024 general strategy language and unrelated capital allocation items. |
| Weaviate hybrid | FAIL | PASS | **Retrieval:** The retrieved chunks identify both initiatives and their scale: FY2024 discloses the Crane Clean Energy Center restart (restart of Three Mile Island Unit 1) at approximately $1.6 billion, and FY2025 discloses the Calpine acquisition as a major growth initiative, with capital needs tied to closing the acquisition referenced in FY2025. The key initiatives asked for are present across the two 10-Ks. |
| Weaviate hybrid_reranked | FAIL | FAIL | **Retrieval:** The retrieved chunks do not contain the two specific FY2024/FY2025 growth initiatives the reference requires with their approximate dollar scales. They are missing the FY2024 Crane Clean Energy Center restart disclosure and its scale, and the FY2025 Calpine acquisition disclosure and its transaction value. |
| Pinecone dense | FAIL | FAIL | **Retrieval:** The retrieved chunks do not contain either of the two specific initiatives named in the reference: there is no disclosure of the Crane Clean Energy Center / Three Mile Island Unit 1 restart in the FY2024 10-K, and no disclosure of the Calpine acquisition in the FY2025 10-K. They also do not provide the approximate dollar scale for those initiatives. |
| Pinecone hybrid_reranked (baseline v2 run 1) | FAIL | FAIL | **Retrieval:** The retrieved chunks do support the FY2025 initiative (Calpine acquisition) and give associated scale figures, but they do not contain the FY2024 10-K disclosure of the Crane Clean Energy Center restart. The only Crane-specific dollar figure shown is from a Cameco filing, not Constellation’s FY2024 10-K, so a needed fact is missing from the relevant company/period. |

### q12: How does Constellation Energy's disclosed view of nuclear fuel supply risk differ from Cameco's discussion of uranium supply risk, given their different positions in the nuclear fuel supply chain?

Differs on: Retrieval

| Config | Answer | Retrieval | Judge reason |
|---|---|---|---|
| Weaviate dense | PASS | FAIL | **Retrieval:** The chunks support Cameco as a producer/supplier describing heightened supply risk and geopolitical uncertainty in uranium/nuclear fuel supply, and they show Constellation as a utility/buyer procuring fuel across the cycle. However, they do not include the key Constellation fact that it does not anticipate difficulty obtaining the needed fuel and services; instead they only describe diversification and risk management steps. |
| Weaviate hybrid | PASS | FAIL | **Retrieval:** The chunks support Cameco as a producer discussing heightened uranium/nuclear-fuel supply risk driven by geopolitical uncertainty, and they show Constellation as a utility/buyer procuring uranium concentrates, conversion, enrichment, and fabrication with diverse suppliers and inventories. But they do not explicitly provide the key Constellation fact that it does not anticipate difficulty obtaining the fuel and services it needs, so the buyer-side confidence piece is missing. |
| Weaviate hybrid_reranked | PASS | PASS | **Retrieval:** The chunks contain Constellation’s buyer/utility perspective: it secures fuel from diverse suppliers and does not anticipate difficulty obtaining uranium concentrates, conversion, enrichment, and fabrication services. They also contain Cameco’s producer/miner perspective emphasizing heightened geopolitical supply risk and uncertainty about where future nuclear fuel supplies will come from; taken together, this supports the contrast as arising from opposite positions in the supply chain (buyer vs. producer). |
| Pinecone dense | PASS | FAIL | **Retrieval:** The chunks support Cameco as a producer/supplier facing significant geopolitical and supply uncertainty, and Constellation as a buyer/utility that procures uranium, conversion, enrichment, and fabrication through diverse suppliers and inventory management. However, they do not clearly include the key Constellation fact that it does not anticipate difficulty obtaining the fuel and services it needs; the retrieval only describes risk management and continuity efforts, not that explicit confidence statement. |
| Pinecone hybrid_reranked (baseline v2 run 1) | PASS | PASS | **Retrieval:** The retrieved chunks support both sides of the comparison: Constellation is described as a nuclear fuel buyer/utility that uses long-term contracts, diverse suppliers, and explicitly says it does not anticipate difficulty obtaining uranium concentrates, conversion, enrichment, and fabrication services; Cameco is described as a producer/supplier discussing heightened geopolitical supply risk and uncertainty about where future nuclear fuel supplies will come from. Together they also make clear the contrast stems from their different supply-chain positions (utility buyer vs. producer/supplier). |

### q15: How did Cameco's total uranium production (its share) change from fiscal year 2023 to fiscal year 2024?

Differs on: Retrieval

| Config | Answer | Retrieval | Judge reason |
|---|---|---|---|
| Weaviate dense | PASS | FAIL | **Retrieval:** The retrieved chunks directly provide Cameco's uranium production volume of 17.6 million lbs in 2023 and 23.4 million lbs in 2024, including the 33% increase. However, they do not include the operation-level breakdown or drivers (McArthur River/Key Lake from 9.4 to 14.2 and Cigar Lake from 8.2 to 9.2) that are part of the reference answer. |
| Weaviate hybrid | PASS | PASS | **Retrieval:** The retrieved chunks contain Cameco's total uranium production for 2023 (17.6 million lbs) and 2024 (23.4 million lbs), explicitly stating a 33% increase, and they also provide the operation-level changes for Cigar Lake (8.2 to 9.2) and McArthur River/Key Lake (9.4 to 14.2), which supports the main driver of the increase. PASS |
| Weaviate hybrid_reranked | PASS | PASS | **Retrieval:** The retrieved chunks explicitly provide Cameco's total production for 2023 and 2024 (17.6 million lbs and 23.4 million lbs, a 33% increase) and the component changes at McArthur River/Key Lake (9.4 to 14.2, with record production due to improved Key Lake mill performance) and Cigar Lake (8.2 to 9.2). These are all the facts needed to answer the question. |
| Pinecone dense | PASS | FAIL | **Retrieval:** The retrieved chunks include Cameco's total uranium production volumes for 2023 and 2024 (17.6 million lbs and 23.4 million lbs), which is enough to determine the year-over-year change and approximate 33% increase. However, they do not provide the operation-level breakdown or drivers for 2023 vs. 2024 (e.g., McArthur River/Key Lake 9.4 to 14.2 and Cigar Lake 8.2 to 9.2, improved Key Lake mill performance) needed for the full reference explanation. |
| Pinecone hybrid_reranked (baseline v2 run 1) | PASS | PASS | **Retrieval:** The retrieved chunks provide Cameco's total uranium production for 2023 and 2024 (17.6M lbs to 23.4M lbs, a 33% increase) and the operation-level breakdown showing McArthur River/Key Lake rising from 9.4M to 14.2M due to record production/improved Key Lake mill performance, while Cigar Lake increased from 8.2M to 9.2M. These facts are sufficient to produce the reference answer. |


## Known limitation: q09

q09 asks for Constellation Energy's largest growth initiatives in its FY2024 and FY2025 10-Ks and the approximate dollar scale of each (expected: the Crane Clean Energy Center restart and the Calpine acquisition). It fails the answer check in every config above and in all three baseline v2 runs.

**Cause: multi-hop discovery.** The question doesn't name either initiative, so the agent has to discover what they are before it can look up their size. The first search returns general strategy language and an FY2023 excerpt about "$875 million of growth capital expenditures" (uprates, wind repowering, hydrogen). The retries then search with that excerpt's vocabulary instead of finding the FY2024/FY2025 initiatives. On Weaviate, no excerpt in any run names Calpine or Crane. A typical retry query (Weaviate hybrid_reranked):

> Constellation Energy 2024 10-K "Constellation's Strategy and Outlook" growth capital expenditures nuclear uprates license renewals wind repowering hydrogen sustainable solutions data centers retail load approximate dollar scale 2025 10-K

The agent then correctly declines rather than guessing, but the rubric requires both initiatives, so it fails.

**Attempted fix (reverted).** We gave the reformulate step the retrieved excerpts and told it to search for named items whose requested attribute was missing ("Step A"). On Weaviate hybrid_reranked, q09 still failed 3/3, and the retry queries got worse: they copied the wrong-year excerpt's terms more heavily. q12's retrieval failed in both runs with the change (it passed in the comparison run), and questions that retry cost about 30% more. The change was reverted, so the agent is unchanged.

**Future work.**
- **Discovery step before lookup:** for questions that ask for unnamed items plus an attribute of each, first run a search to identify the items (e.g. major acquisitions or projects in each requested filing), then search for each named item's attribute separately.
- **Fiscal-year metadata filtering:** when the question names fiscal years, restrict retrieval (or retry queries) to filings from those years using the `fiscal_year` metadata, so out-of-period excerpts like the FY2023 capex chunk can't take over the search.
