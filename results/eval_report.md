# Clinical RAG — Evaluation Report

- Embeddings: `BAAI/bge-small-en-v1.5`
- Reranker: `BAAI/bge-reranker-base`
- LLM provider: `mock`
- De-identification backend: `presidio`


## 1. Retrieval quality (ablation)

| configuration | recall@1 | ndcg@1 | recall@3 | ndcg@3 | recall@5 | ndcg@5 | precision@1 | mrr |
|---|---|---|---|---|---|---|---|---|
| dense | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| bm25 | 0.920 | 0.920 | 1.000 | 0.970 | 1.000 | 0.970 | 0.920 | 0.960 |
| hybrid | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| hybrid+rerank | 0.960 | 0.960 | 1.000 | 0.980 | 1.000 | 0.980 | 0.960 | 0.973 |

*Reading it:* on this small, topically-separable corpus dense retrieval already saturates (recall@5 = 1.0), so hybrid adds no headroom and the cross-encoder even trades a hair of nDCG by reordering an already-correct top result. That is expected — BM25/dense/hybrid/re-rank differ most on *larger, more ambiguous* corpora where first-stage ranking is noisy. The ablation harness is built to surface exactly that difference; here it mostly confirms the task is easy for dense retrieval. BM25 alone is the weakest (recall@1 = 0.92), showing where pure lexical matching slips.


## 2. Generation quality (primary pipeline)

| metric | value |
|---|---|
| groundedness (faithfulness proxy) | 1.000 |
| answer token-F1 vs reference | 0.412 |
| citation coverage | 1.000 |

## 3. Privacy

| metric | value |
|---|---|
| queries evaluated | 25 |
| PHI entities pseudonymised | 68 |
| raw PHI leaked into prompt | 0 |
| leak rate | 0.000 |

**0 leaks** — no raw PHI reaches the prompt / LLM boundary.


## 4. Example traces

**Query:** Mr. John Smith (MRN 4839201, DOB 03/12/1958) was seen on 02/04/2025. His clinic BP readings are 152/96 and 148/94. How is hypertension diagnosed and what first-line treatment should we start?

**Pseudonymised:** Mr. [PERSON_1] ([MRN_1], DOB [DATE_TIME_1]) was seen on [DATE_TIME_2]. His clinic BP readings are 152/96 and 148/94. How is hypertension diagnosed and what first-line treatment should we start?

**Retrieved:** S1=htn-001, S2=htn-002, S3=depression-001, S4=lipids-001

**Answer:** Hypertension is diagnosed when office blood pressure is persistently 140/90 mmHg or higher on repeated readings, or when out-of-office monitoring confirms elevated values. [S1] First-line lifestyle measures are sodium restriction, weight loss, the DASH diet, regular aerobic exercise, and limiting alcohol. [S1] First-line antihypertensive drug classes include thiazide diuretics, ACE inhibitors, angiotensin receptor blockers, and calcium channel blockers. [S1]


**Query:** Patient Maria Garcia, age 62, phone 555-203-8890, presents with BP 190/125, a severe headache and confusion. Is this an emergency and how should BP be managed?

**Pseudonymised:** Patient [PERSON_1], [DATE_TIME_1], phone [PHONE_NUMBER_1], presents with BP 190/125, a severe headache and confusion. Is this an emergency and how should BP be managed?

**Retrieved:** S1=htn-002, S2=lipids-001, S3=ckd-001, S4=htn-001

**Answer:** A hypertensive emergency is severe blood pressure elevation, usually above 180/120 mmHg, accompanied by acute target-organ damage such as encephalopathy, intracranial hemorrhage, acute heart failure, aortic dissection, or acute kidney injury. [S1] Hypertensive urgency is severe elevation without acute organ damage and can usually be managed with oral agents and close outpatient follow-up. [S1] High-intensity statin therapy is recommended for patients with established atherosclerotic cardiovascular disease and for those with an LDL cholesterol of 190 mg/dL or higher. [S2]


**Query:** Dr. Alan Kim asks about patient Robert Brown (MRN 77120): fasting glucose 132 mg/dL and A1c 6.8%. Does he meet criteria for type 2 diabetes?

**Pseudonymised:** Dr. [PERSON_1] asks about patient [PERSON_2] ([MRN_1]): fasting glucose 132 mg/dL and A1c 6.8%. Does he meet criteria for type 2 diabetes?

**Retrieved:** S1=dm2-001, S2=dm2-002, S3=dm2-003, S4=hypogly-001

**Answer:** Type 2 diabetes is diagnosed by a fasting plasma glucose of 126 mg/dL or higher, a 2-hour plasma glucose of 200 mg/dL or higher during an oral glucose tolerance test, a hemoglobin A1c of 6.5 percent or higher, or a random glucose of 200 mg/dL or higher with classic symptoms. [S1] Prediabetes is defined by A1c 5.7 to 6.4 percent or impaired fasting glucose of 100 to 125 mg/dL. [S1] Metformin is the preferred initial pharmacologic therapy for type 2 diabetes unless contraindicated, alongside lifestyle modification. [S2]


