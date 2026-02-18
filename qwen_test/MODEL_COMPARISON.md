# Local vs Cloud LLM — Job Evaluation Model Comparison

Comparing local Ollama models against Claude Haiku (Anthropic API) for structured job-fit evaluation.

---

## Hardware

| Component | Spec |
|---|---|
| GPU | NVIDIA RTX 3070 (8 GB VRAM) |
| RAM | 16 GB system + 16 GB swap |
| CPU | Used for layers that overflow VRAM |

---

## Models Tested

| Model | Size | Mode | Scale |
|---|---|---|---|
| `qwen2.5:14b` | ~9 GB | Ollama tool-call | 1–100 (schema ignored) |
| `qwen2.5:32b` | ~19 GB | Ollama native API, `num_ctx=2048` | 1–100 |
| `qwen3:14b` | ~9.3 GB | Ollama tool-call | 1–100 (schema ignored) |
| `deepseek-r1:14b` | ~9 GB | Prompt-JSON fallback (no tool support) | 1–10 |
| `claude-haiku-4-5-20251001` | API | Anthropic tool-call | 1–10 |

> **Score scale note:** The tool schema specifies 1–10 but local qwen models return 1–100.
> DeepSeek and Haiku correctly use 1–10. Scores are reported as-is; /10 suffix marks the 1–10 models.

---

## Part 1 — Synthetic Benchmark

Single fixed job: *Shopify Staff Data Engineer* (from `qwen_request.json`)
Candidate: Marcus Chen (Senior Data Engineer resume)

| Model | Verdict | skills | career_lvl | exp_rel | culture | Avg | Time |
|---|---|---|---|---|---|---|---|
| qwen2.5:14b | Step Up | 50 | 60 | 70 | 40 | **55** | 41 s |
| qwen2.5:32b | Step Up | 75 | 60 | 80 | 70 | **71** | 157 s |
| qwen3:14b | Step Up | 75 | 60 | 70 | 80 | **71** | 77 s |
| deepseek-r1:14b | Step Up | 7.5/10 | 6/10 | 8.5/10 | 7.5/10 | **7.4/10** | 92 s |
| claude-haiku | ~~Lateral~~ | 7/10 | 6/10 | 8/10 | 8/10 | **7.3/10** | 7 s |

**Verdict accuracy note:** Senior → Staff *is* a Step Up by definition. Haiku returned "Lateral" — it conflated "stretch candidate" with "lateral career move". All local models correctly identified this as a Step Up.

### qwen2.5:14b false gaps (quality issue)
qwen2.5:14b flagged `Terraform`, `Kubernetes`, and `GCP basics` as gaps even though the candidate explicitly listed them. The other models did not make this error.

### Top gaps identified across models (Shopify Staff role)
- Scala / Java (critical — all models agree)
- Apache Flink
- GCP / BigQuery at petabyte scale
- MLOps / feature stores (Feast, Tecton)
- Staff-level leadership scope (8+ years, team of 6+)

---

## Part 2 — Real-Data Batch (5 Jobs, Profile: Slava)

5 most recent jobs from `public.jobspy_jobs`, models run sequentially with full Ollama memory flush between runs.

### Summary table

| Company | Title | qwen3:14b | deepseek-r1:14b | haiku |
|---|---|---|---|---|
| Fusemachines | Lead Spark Data Engineer | Pivot / 71 | Lateral / 7.5/10 | Pivot / 7.0/10 |
| KTek Resourcing | Senior Data Engineer | Pivot / 76 | Pivot / 7.5/10 | Pivot / 7.0/10 |
| SPECTRAFORCE | Senior Data Engineer | Lateral / 90 | Lateral / 7.5/10 | Step Up / 8.5/10 |
| Aarorn Technologies | Senior Data Engineer | Pivot / 65 | Pivot / 5.75/10 | Pivot / 6.25/10 |
| Adastra | Data Engineer | Lateral / 96 | Step Up / 8.5/10 | Step Up / 9.0/10 |

---

### Job 1 — Fusemachines: Lead Spark Data Engineer

**Role focus:** Azure ecosystem, Java, Apache Spark internals (Catalyst Optimizer), ANTLR/DSL, Delta Lake
**Consensus:** Pivot (qwen3 + haiku agree; deepseek says Lateral)

| Model | Verdict | Gaps identified |
|---|---|---|
| qwen3:14b | Pivot | Azure ecosystem, Java, ANTLR, Delta Lake, Spark Internals, Azure DevOps, ASTs |
| deepseek-r1:14b | Lateral | Azure ecosystem, ANTLR, Spark Internals (Catalyst), Java |
| claude-haiku | Pivot | Azure (AWS-focused candidate), Java, Spark Catalyst/Logical Plans, ANTLR, ASTs, Delta Lake, Azure DevOps, Azure certifications |

Haiku provides the most granular gap list. DeepSeek underweighted the cloud mismatch (AWS vs Azure is a pivot, not lateral).

---

### Job 2 — KTek Resourcing: Senior Data Engineer

**Role focus:** Databricks/Scala on Azure, Azure Data Factory, LangChain, Azure OpenAI, Vector DBs, RAG
**Consensus:** Pivot (all three agree)

| Model | Verdict | Gaps identified |
|---|---|---|
| qwen3:14b | Pivot | Azure Data Factory, Scala, Azure OpenAI, Vector DBs, RAG, CI/CD on Azure |
| deepseek-r1:14b | Pivot | Azure Data Factory, Databricks (Scala), Azure OpenAI, .Net/React APIs |
| claude-haiku | Pivot | Azure cloud, Azure Data Factory, Scala, Databricks, LangChain, Azure OpenAI, Vector DBs, RAG, GenAI experience |

Strong consensus. Haiku identified the emerging AI stack gaps (LangChain, RAG) most explicitly.

---

### Job 3 — SPECTRAFORCE: Senior Data Engineer

**Role focus:** Python, AWS (SageMaker preferred), ETL, tech modernization. No Azure, no Scala.
**Consensus split:** qwen3/deepseek say Lateral; haiku says Step Up

| Model | Verdict | Key notes |
|---|---|---|
| qwen3:14b | Lateral / 90 | Matches AWS, Python, ETL well. Gaps: SageMaker, SAS, WebFOCUS |
| deepseek-r1:14b | Lateral / 7.5/10 | Strong tech match; notes "some gaps in specific banking tools" |
| claude-haiku | Step Up / 8.5/10 | Exceptional alignment; candidate's 100x pipeline optimization directly relevant. Gaps: SageMaker, SAS, WebFOCUS, banking exp |

This is the strongest match of the 5 jobs. Role aligns closely with AWS/Python/ETL background. "Step Up" vs "Lateral" comes down to whether finance domain counts against the candidate — reasonable for haiku to call it Step Up.

---

### Job 4 — Aarorn Technologies: Senior Data Engineer

**Role focus:** Hadoop ecosystem (Hive, Hive Metastore), GCP (Dataproc, BigQuery, GCS, Cloud Composer)
**Consensus:** Pivot (all three agree)

| Model | Verdict | Gaps identified |
|---|---|---|
| qwen3:14b | Pivot / 65 | Hadoop, GCP Dataproc, Hive, BigQuery, Cloud Composer, Dataproc Serverless |
| deepseek-r1:14b | Pivot / 5.75/10 | Hadoop ecosystem, GCP stack, Security/IAM |
| claude-haiku | Pivot / 6.25/10 | GCP Dataproc, BigQuery, Hadoop/Hive, GCS, Cloud Composer, Hive-to-BQ migration, ORC/Parquet optimization |

All models correctly flag this as a major cloud-platform mismatch (AWS-native candidate for a GCP/Hadoop role). Haiku identifies the most specific gaps including migration patterns and table format experience.

---

### Job 5 — Adastra: Data Engineer

**Role focus:** AWS Redshift/Glue, SQL, data warehousing — core stack match
**Consensus:** Lateral/Step Up (strong positive signal)

| Model | Verdict | Key notes |
|---|---|---|
| qwen3:14b | Lateral / 96 | Near-perfect match. Only gaps: EMR (not required) and "implicit" big data frameworks |
| deepseek-r1:14b | Step Up / 8.5/10 | Strong match. Minimal gaps noted. Calls it Step Up (candidate overqualified?) |
| claude-haiku | Step Up / 9.0/10 | Exceptional match. Scores 10/10 experience_relevance. Only gap: Agile not explicit |

Candidate's profile is a textbook match for Adastra: AWS Redshift, Glue, S3, Lambda, Athena, dimensional modeling, and even the 100x pipeline optimization story aligns with their modernization focus. Recommended to apply.

---

## Part 3 — Model Assessment

### Speed

| Model | Synthetic test | Per-job (batch avg) |
|---|---|---|
| claude-haiku | **7 s** | ~5–6 s |
| qwen3:14b | 77 s | ~60–90 s |
| qwen2.5:14b | 41 s | — |
| deepseek-r1:14b | 92 s | ~80–120 s |
| qwen2.5:32b | 157 s | — |

### Quality (qualitative)

| Model | Verdict accuracy | Gap completeness | False positives | Notes |
|---|---|---|---|---|
| claude-haiku | High (1 wrong in synthetic) | Excellent — most detailed | None found | Incorrect "Lateral" for Sr→Staff transition |
| qwen3:14b | High | Good | Minor (qwen2.5 issue, not qwen3) | Best local model overall |
| deepseek-r1:14b | High | Moderate | None | Scores on correct 1–10 scale; reasoning in `<think>` blocks |
| qwen2.5:14b | Medium | Moderate | Yes — flags present skills as gaps | Not recommended; superseded by qwen3 |
| qwen2.5:32b | High | Good | None found | Same quality as qwen3 at 2× the time; requires special memory handling |

### Memory & reliability

| Model | VRAM | RAM | Reliability |
|---|---|---|---|
| qwen3:14b | ~7 GB | ~2 GB | Stable, loads cleanly |
| qwen2.5:14b | ~7 GB | ~2 GB | Stable |
| deepseek-r1:14b | ~7 GB | ~2 GB | Stable; no tool-call support |
| qwen2.5:32b | 5.4 GB (split) | 12.1 GB | Requires: native API, `num_ctx=2048`, run first after unload |
| claude-haiku | API | API | Always reliable; prompt caching on resume |

---

## Recommendation

**Use `qwen3:14b` as the local model.**
- Matches qwen2.5:32b quality at half the speed and with no memory gymnastics
- Native tool-call support (no prompt-JSON fallback needed)
- Stable on 8 GB VRAM

**Claude Haiku** remains best for:
- Highest gap completeness and most actionable summaries
- Speed (7 s vs 77 s per job)
- Cost at scale when Anthropic API budget allows

**Verdict taxonomy clarification:**
- Senior → Staff = **Step Up** (definitional; do not accept "Lateral")
- AWS-native candidate for Azure-first role = **Pivot** (not Lateral)
- Strong match with minor gaps at same level = **Lateral**

---

## Scripts

| Script | Purpose |
|---|---|
| `qwen_test/compare_models.py` | Synthetic benchmark against fixed `qwen_request.json` payload |
| `qwen_test/batch_compare.py` | Live batch: queries DB, runs N jobs through all 3 models |
| `qwen_test/batch_results.json` | Raw output from last batch run |
