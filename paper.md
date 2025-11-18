
# **Automating Regulatory → Internal Requirement Mapping**

### *A detailed exploration of retrieval limits and a practical shift toward agentic, full-space reasoning*

---

## **1. Executive Summary**

This document summarizes the work I have done on improving the automation pipeline that maps external regulatory citations (NIST, DORA, ISO, etc.) to our internal cybersecurity requirements. The earlier team’s efforts were extremely useful in understanding the nature of the problem — their RAG-based experiments clearly exposed precision and recall gaps, particularly when citations were abstract and internal requirements were highly operational.

Building on that foundation, I explored multiple advanced retrieval strategies: sparse neural models, hybrid dense+sparse retrieval, reciprocal rank fusion, late-interaction architectures, late chunking, and LLM-based reranking. Each of these techniques is strong in isolation and widely used in modern search systems. However, across both theoretical reasoning and practical testing, all of them consistently ran into the same limitation: the problem we are solving is not a similarity problem at all — it is a reasoning problem.

Regulatory citations express intent, while internal requirements express mechanisms. These two layers almost never align in vocabulary or textual semantics. As a result, retrieval-based systems fail to surface many true matches, resulting in structurally high false-negative rates. Given that missing a valid requirement is far more costly than additional compute, the right direction is to expand the evaluation scope and shift toward domain-specialist agents that systematically reason through all relevant requirements in each domain, rather than restricting the evaluation to retrieved subsets.

---

## **2. Problem Overview: Intent vs. Mechanism**

Our internal cybersecurity framework contains thousands of requirements. Their purpose is to define *how* things are done operationally: which team performs which task, what cadence is required, which thresholds trigger which processes, and so on. They are prescriptive, procedural, and action-driven.

Regulatory citations, by contrast, are inherently high-level. They describe outcomes such as governance alignment, timely response, resilience, and oversight. They are intentionally abstract because regulators expect organizations to implement these outcomes through internal governance structures and policies.

This leads to a fundamental abstraction gap:

* Regulatory text speaks in **principles**.
* Internal requirements speak in **procedures**.

The challenge is that the two rarely resemble each other in wording, even when they are conceptually connected. This mismatch is the core obstacle for any retrieval-based system.

---

## **3. Real Examples of the Abstraction Mismatch**

### **Governance and Oversight**

A regulator may require *“appropriate oversight and alignment of cybersecurity activities with strategic objectives.”*
The actual internal requirement might involve *“assigning risk owners to business processes with quarterly governance reviews.”*

Although assigning owners and running governance reviews is *exactly* how alignment and oversight are operationalized, the textual similarity between these two statements is negligible.

### **Incident Response**

A citation might say, *“Ensure timely detection and response to incidents.”*
But our internal requirement may state, *“SOC escalates incidents to Tier-2 within 15 minutes based on severity thresholds.”*
Again, no shared vocabulary. Yet the 15-minute escalation rule is the operational enforcement of timely response.

### **Third-Party Assurance**

A regulation may require adherence to cybersecurity standards by third parties.
Our internal process might enforce this via *annual cybersecurity due-diligence questionnaires*.
The connection is clear to a human, but invisible to retrieval.

These examples are the rule, not the exception. They demonstrate why mere textual similarity is insufficient for mapping one to the other.

---

## **4. Previous Work and Initial Observations**

The earlier team had implemented a retrieval-augmented generation (RAG) pipeline where citations were embedded, relevant requirements retrieved from a vector database, and an LLM ranked and evaluated the retrieved subset. They also tested RAG Fusion by splitting citations into sub-citations to increase recall.

Their evaluations showed the same recurring pattern: while retrieval was able to return plausible-sounding documents, many truly relevant requirements never appeared in the retrieved set at all. This initiated the suspicion that the limitation was structural, not just a matter of tuning.

Their findings shaped the direction of my deeper exploration.

---

## **5. My Retrieval Experiments (Deep Dive + KT)**

### **5.1 Dense Embedding Retrieval**

Dense embeddings represent texts as vectors in a high-dimensional semantic space. Similar meanings should be close in this space, which works brilliantly for tasks like FAQ retrieval or natural-language QnA.

The limitation becomes clear once we frame the problem accurately:
Dense retrieval assumes that *relevant* texts are also *semantically similar*.
In our use case, relevant internal requirements are often semantically unrelated to the regulatory text. “Timely response” and “Tier-2 escalation within 15 minutes” have no semantic closeness, so embeddings cannot pull them together.

This breaks dense retrieval conceptually.

---

### **5.2 Sparse Neural Retrieval (SPLADE, ColBERT-style)**

Sparse neural models behave like learned versions of BM25. They identify the important words in a sentence, expand them, and produce sparse lexical vectors that capture token-level importance.

This is extremely powerful when domain-specific terminology matters.

However, sparse models **still depend on token overlap** or token-level relatedness.
Regulatory citations use governance vocabulary.
Internal requirements use operational vocabulary.
The two vocabularies rarely touch.

Sparse models cannot infer:

* Oversight → assigning risk owners
* Timely response → escalation SLAs
* Third-party adherence → annual questionnaires

Even the smartest lexical model cannot cross abstraction layers.

---

### **5.3 Hybrid Retrieval (Dense + Sparse Fusion)**

Hybrid retrieval blends semantic (dense) and lexical (sparse) signals.
In combination with RRF (explained below), hybrid models often represent the best of modern search systems.

In practice, hybrid retrieval did:

* increase candidate diversity
* reduce sensitivity to vocabulary drift
* stabilize rankings

But hybrid retrieval can only work with signals that exist.
If neither the dense nor sparse model sees a relationship, hybrid has nothing to promote.
And the relationships we care about are conceptual, not textual.

---

### **5.4 Reciprocal Rank Fusion (RRF)**

RRF merges multiple ranked lists by rewarding documents that appear in good positions across several retrieval strategies. It is widely used in production search pipelines and can dramatically improve ranking stability.

However, RRF cannot surface documents that never appear in any list.
If the dense model doesn’t see it, and the sparse model doesn’t see it, RRF cannot create relevance out of thin air.

This is where retrieval hits its hard ceiling.

---

### **5.5 Late Interaction Models**

Late-interaction architectures (e.g., ColBERT-V2) preserve token-level embeddings and compare query tokens against document tokens more precisely than dense embeddings.

They are excellent at multi-sentence similarity matching in:

* legal search
* biomedical search
* passage-level retrieval

But they are still fundamentally **retrieval** systems.
If no token-level relationships exist between citation and requirement, they cannot infer mechanisms.

They improve granularity, not abstraction bridging.

---

### **5.6 Late Chunking**

Late chunking allows us to embed larger contexts and only break them into chunks during retrieval, reducing fragmentation issues and improving retrieval consistency across long documents.

While this is highly relevant for handling PDF-based regulatory documents, it does not address the abstraction mismatch between intent and mechanism. Chunking improves structure; it does not create conceptual alignment.

---

### **5.7 LLM-Based Reranking with Qdrant**

Reranking retrieved candidates with an LLM significantly improves precision within the retrieved set. It is helpful for filtering and scoring plausible candidates.

However, reranking cannot fix missing candidates.
If retrieval fails to include a relevant requirement in the top-k pool, reranking has no opportunity to evaluate it.

Across many tests, correct mappings simply never surfaced.

---

### **5.8 Multi-Query / RAG Fusion Revisited**

RAG Fusion tries to increase recall by having an LLM generate multiple sub-queries.
This broadens the search and can sometimes recover variations of a concept.

However, sub-queries still contain the vocabulary of the original citation.
If the mapping is between governance intent and operational mechanisms, even ten sub-queries cannot unfold that conceptual bridge.

RAG Fusion increases width, not depth.

---

## **6. Combined Empirical and Theoretical Conclusion**

Across all methods — dense, sparse, hybrid, late interaction, late chunking, RRF, LLM reranking, and RAG Fusion — the pattern was consistent:

**Retrieval returns what is textually similar.
Our problem requires identifying what is conceptually linked.**

These are fundamentally different operations.

Retrieval is bounded by textual and semantic similarity.
Regulatory–requirement mapping hinges on multi-hop, domain-grounded reasoning.

Because of this mismatch, retrieval-based pipelines will always:

* miss many valid mappings (false negatives),
* surface semantically-plausible but operationally-irrelevant requirements,
* depend on vocabulary overlap that does not exist.

Therefore, the structural conclusion is clear:
**RAG is not suitable as the primary mechanism for this use case.**

---

## **7. Proposed Direction: Agentic Full-Space, Domain-Specialist Reasoning**

Since retrieval cannot reliably narrow the space, the solution is to **expand the space and let LLM agents evaluate more**, not less.

The proposed architecture uses domain-specialist agents for each cybersecurity domain. Instead of retrieving candidates, the system presents agents with entire batches of internal requirements — ensuring that no requirement is missed due to vocabulary mismatch. The agent evaluates each requirement against the citation using structured, step-by-step reasoning, producing coverage verdicts, rationale, and evidence.

Parallel execution, batching, caching, and heuristic prefilters ensure throughput remains manageable. This approach prioritizes completeness and accuracy, which is critical for compliance tasks.

This is a controlled, auditable, exhaustive reasoning pipeline — not a similarity pipeline.

---

## **8. Final Conclusion**

After a thorough conceptual and empirical study, the conclusion is well-founded: **retrieval-centric architectures cannot be the foundation of regulatory mapping.** The abstraction mismatch between regulatory intent and internal mechanisms ensures that the true matches often lack textual similarity.

Exhaustive, domain-based, reasoning-driven evaluation is the correct direction — even if it demands more computation — because the cost of missing a requirement far outweighs the cost of evaluating more requirements.

This agentic approach is accurate, defensible, extensible, and future-proof.

---

# **Glossary of Concepts (KT Reference)**

### **Dense Embeddings**

High-dimensional vectors capturing semantic similarity. Good for conceptual similarity but weak when relevant items are not semantically close.

### **Sparse Neural Retrieval (e.g., SPLADE, ColBERT)**

Models that produce BM25-like sparse token vectors with learned importance. Still depend heavily on token overlap.

### **Hybrid Retrieval**

Combination of dense and sparse retrieval to blend semantic and lexical signals.

### **Reciprocal Rank Fusion (RRF)**

Method to merge multiple ranked lists by rewarding documents that rank well across several retrieval systems.

### **Late Interaction Models**

Models that compute token-level similarity between queries and documents instead of holistic vector similarity. More granular but still similarity-bound.

### **Late Chunking**

Chunking long documents *after* embedding rather than before. Helps preserve context for long PDFs but still a retrieval technique.

### **RAG Fusion / Multi-Query Retrieval**

Technique where an LLM generates multiple versions of the query to increase recall. Expands search breadth but not conceptual depth.

### **LLM Reranking**

Reordering retrieved candidates using an LLM to improve precision. Cannot recover missing candidates.

---

If you'd like, I can now:
✔ generate an **executive-summary-only version**
✔ produce **agent design templates**
✔ produce a **Q&A appendix**
Just tell me.
