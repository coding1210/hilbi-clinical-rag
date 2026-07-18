# Clinical RAG — Evaluation Report

- Embeddings: `BAAI/bge-small-en-v1.5`
- Reranker: `BAAI/bge-reranker-base`
- LLM provider: `openai`
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
| groundedness (faithfulness proxy) | 0.881 |
| answer token-F1 vs reference | 0.589 |
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

**Answer:** Hypertension is diagnosed when office blood pressure is persistently 140/90 mmHg or higher on repeated readings, which applies to Mr. John Smith as his readings are 152/96 and 148/94 [S1]. 

For first-line treatment, lifestyle measures such as sodium restriction, weight loss, the DASH diet, regular aerobic exercise, and limiting alcohol should be initiated. Additionally, first-line antihypertensive drug classes include thiazide diuretics, ACE inhibitors, angiotensin receptor blockers, and calcium channel blockers [S1].


**Query:** Patient Maria Garcia, age 62, phone 555-203-8890, presents with BP 190/125, a severe headache and confusion. Is this an emergency and how should BP be managed?

**Pseudonymised:** Patient [PERSON_1], [DATE_TIME_1], phone [PHONE_NUMBER_1], presents with BP 190/125, a severe headache and confusion. Is this an emergency and how should BP be managed?

**Retrieved:** S1=htn-002, S2=lipids-001, S3=ckd-001, S4=htn-001

**Answer:** Yes, this is a hypertensive emergency. The blood pressure of 190/125 mmHg is above the threshold of 180/120 mmHg, and the presence of severe headache and confusion suggests acute target-organ damage, which is characteristic of a hypertensive emergency [S1]. 

Management requires admission and controlled intravenous blood pressure reduction to avoid complications from abrupt reductions, which can lead to cerebral or coronary hypoperfusion [S1].


**Query:** Dr. Alan Kim asks about patient Robert Brown (MRN 77120): fasting glucose 132 mg/dL and A1c 6.8%. Does he meet criteria for type 2 diabetes?

**Pseudonymised:** Dr. [PERSON_1] asks about patient [PERSON_2] ([MRN_1]): fasting glucose 132 mg/dL and A1c 6.8%. Does he meet criteria for type 2 diabetes?

**Retrieved:** S1=dm2-001, S2=dm2-002, S3=dm2-003, S4=hypogly-001

**Answer:** Yes, patient Robert Brown meets the criteria for type 2 diabetes. The fasting plasma glucose of 132 mg/dL is above the diagnostic threshold of 126 mg/dL or higher for type 2 diabetes, as stated in the sources [S1]. Additionally, the A1c of 6.8% is also above the threshold of 6.5% or higher for diagnosis. Therefore, both criteria confirm the diagnosis of type 2 diabetes.


