\# AML Red Flag RAG Service Demo Spec



\## 1. Demo Goal



本 demo 的目標是將既有 AML Red Flag RAG notebook 實驗，整理成一個可本地啟動、可 API 呼叫、可用 Docker Compose 執行的最小服務原型。



此版本不追求完整企業級系統，而是展示以下能力：



1\. 將 notebook 實驗重構為可維護的 Python 模組。

2\. 將 RAG 查詢流程包裝成 FastAPI service。

3\. 使用 Docker Compose 提供可重現的本地執行環境。

4\. 保留檢索證據、引用來源、拒答邏輯與 debug 資訊。

5\. 為後續 Streamlit UI、資料入口、eval automation 預留乾淨介面。



\## 2. Scope



\### In Scope



本階段必須完成：



\* FastAPI backend

\* Dockerfile

\* docker-compose.yml

\* `.env.example`

\* `/health` API

\* `/query` API

\* mock LLM mode

\* artifact loading

\* retrieval debug output

\* README Quick Start 更新

\* 基本 smoke test



\### Optional but Allowed



若不破壞主流程，可以加入：



\* `/sources` API：列出目前載入的文件與 chunks 數量

\* `/debug/config` API：顯示目前 llm\_mode、index 狀態、模型名稱

\* 最小 Streamlit client，但本階段不強制



\### Out of Scope



本階段不做：



\* Kubernetes

\* 雲端部署

\* 使用者登入與權限系統

\* 自動監控資料夾

\* Google Drive / Notion / Gmail 等外部資料同步

\* 完整資料庫

\* 完整 CI/CD

\* 大規模 eval dashboard

\* 重新設計 RAG 演算法

\* 新增 LangChain / LlamaIndex 等重型框架

\* 把 notebook 裡還不穩定的實驗結果偽裝成正式功能



\## 3. Current Source Assumption



既有 notebook / script 已包含部分 RAG 實驗邏輯，例如：



\* PDF / markdown chunking

\* FAISS dense retrieval

\* BM25 sparse retrieval

\* Hybrid search / RRF fusion

\* metadata priority weighting

\* Pre-LLM Gate 或拒答邏輯

\* LLM generation 或 mock generation

\* multi-turn / intent routing 實驗



本 demo 不要求一次完整產品化所有功能。優先抽出穩定可跑的單輪 RAG 查詢流程。



\## 4. Target Architecture



```text

User / Client

&#x20;   |

&#x20;   v

FastAPI Service

&#x20;   |

&#x20;   +-- /health

&#x20;   +-- /query

&#x20;   +-- /sources

&#x20;   |

&#x20;   v

rag\_core/

&#x20;   |

&#x20;   +-- loaders.py       # 載入 artifacts / index / chunks

&#x20;   +-- retrieval.py     # dense / bm25 / hybrid search

&#x20;   +-- gate.py          # out-of-scope / refuse decision

&#x20;   +-- generation.py    # mock or API LLM answer

&#x20;   +-- schemas.py       # request / response schema

&#x20;   +-- config.py        # env and path settings

&#x20;   |

&#x20;   v

artifacts/

&#x20;   |

&#x20;   +-- chunks.json

&#x20;   +-- faiss.index

&#x20;   +-- bm25.pkl

&#x20;   +-- manifest.json

```



\## 5. Proposed Repository Structure



```text

aml-redflags-rag/

├── api/

│   ├── \_\_init\_\_.py

│   └── main.py

│

├── rag\_core/

│   ├── \_\_init\_\_.py

│   ├── config.py

│   ├── schemas.py

│   ├── loaders.py

│   ├── retrieval.py

│   ├── gate.py

│   └── generation.py

│

├── indexing/

│   └── build\_data\_v2.py

│

├── artifacts/

│   └── index/

│       ├── chunks.json

│       ├── faiss.index

│       ├── bm25.pkl

│       └── manifest.json

│

├── data/

│   └── sample\_queries.json

│

├── tests/

│   └── smoke\_test.py

│

├── Dockerfile

├── docker-compose.yml

├── requirements.txt

├── .env.example

├── README.md

└── DEMO\_SPEC.md

```



\## 6. API Design



\## 6.1 GET /health



Purpose: 確認服務是否啟動，並回報 artifacts 是否已載入。



Response example:



```json

{

&#x20; "status": "ok",

&#x20; "service": "aml-redflags-rag-api",

&#x20; "artifacts\_loaded": true,

&#x20; "llm\_mode": "mock",

&#x20; "index\_version": "demo-v1"

}

```



If artifacts are missing:



```json

{

&#x20; "status": "degraded",

&#x20; "service": "aml-redflags-rag-api",

&#x20; "artifacts\_loaded": false,

&#x20; "message": "Artifacts not found. Please run indexing/build\_data\_v2.py or mount artifacts/index."

}

```



\## 6.2 POST /query



Purpose: 對 AML 情境或問題執行 RAG 查詢。



Request example:



```json

{

&#x20; "query": "客戶短時間內多次將資金轉入虛擬資產交易所，且交易金額與其學生身分不符，這可能涉及哪些洗錢紅旗？",

&#x20; "top\_k": 5,

&#x20; "retrieval\_mode": "hybrid",

&#x20; "llm\_mode": "mock",

&#x20; "include\_debug": true

}

```



Response example:



```json

{

&#x20; "answer": "此情境可能涉及快速資金流轉、身分背景不符，以及虛擬資產相關風險。由於交易行為與學生身分不一致，且資金快速進入虛擬資產交易所，系統判定為 possible 或 confirmed，實際等級需依文件證據與交易細節確認。",

&#x20; "assessment": "possible",

&#x20; "identified\_flags": \[

&#x20;   {

&#x20;     "code": "RF-02",

&#x20;     "name": "Rapid Movement",

&#x20;     "reason": "資金在短時間內快速移動。"

&#x20;   },

&#x20;   {

&#x20;     "code": "RF-06",

&#x20;     "name": "Profile Mismatch",

&#x20;     "reason": "交易金額或模式與客戶身分不符。"

&#x20;   },

&#x20;   {

&#x20;     "code": "RF-07",

&#x20;     "name": "Virtual Asset Risk",

&#x20;     "reason": "情境涉及虛擬資產交易所。"

&#x20;   }

&#x20; ],

&#x20; "citations": \[

&#x20;   {

&#x20;     "chunk\_id": "chunk\_001",

&#x20;     "source": "FATF Virtual Assets Red Flags",

&#x20;     "excerpt": "..."

&#x20;   }

&#x20; ],

&#x20; "refusal": {

&#x20;   "refused": false,

&#x20;   "reason": null

&#x20; },

&#x20; "debug": {

&#x20;   "retrieval\_mode": "hybrid",

&#x20;   "top\_k": 5,

&#x20;   "dense\_used": true,

&#x20;   "bm25\_used": true,

&#x20;   "rrf\_used": true,

&#x20;   "gate\_decision": "allow",

&#x20;   "llm\_mode": "mock",

&#x20;   "fallback\_used": false,

&#x20;   "retrieved\_chunk\_ids": \[

&#x20;     "chunk\_001",

&#x20;     "chunk\_018",

&#x20;     "chunk\_044"

&#x20;   ]

&#x20; }

}

```



\## 6.3 GET /sources



Purpose: 顯示目前知識庫載入狀態。



Response example:



```json

{

&#x20; "index\_version": "demo-v1",

&#x20; "total\_chunks": 328,

&#x20; "sources": \[

&#x20;   {

&#x20;     "source\_name": "FATF Trade-Based Money Laundering Risk Indicators",

&#x20;     "language": "en",

&#x20;     "layer": "core",

&#x20;     "chunk\_count": 120

&#x20;   },

&#x20;   {

&#x20;     "source\_name": "FATF Virtual Assets Red Flag Indicators",

&#x20;     "language": "en",

&#x20;     "layer": "sector\_specific",

&#x20;     "chunk\_count": 88

&#x20;   },

&#x20;   {

&#x20;     "source\_name": "Taiwan AML Training Slides",

&#x20;     "language": "zh",

&#x20;     "layer": "knowledge\_bridge",

&#x20;     "chunk\_count": 120

&#x20;   }

&#x20; ]

}

```



\## 7. Environment Variables



`.env.example`



```env

APP\_ENV=local

API\_HOST=0.0.0.0

API\_PORT=8000



ARTIFACT\_DIR=artifacts/index

LLM\_MODE=mock

MODEL\_NAME=mock-local



GEMINI\_API\_KEY=

GROQ\_API\_KEY=



DEFAULT\_TOP\_K=5

DEFAULT\_RETRIEVAL\_MODE=hybrid

ENABLE\_DEBUG=true

```



\## 8. Docker Requirements



The project must support:



```bash

docker compose up --build

```



After startup, user should be able to visit:



```text

http://localhost:8000/health

```



and call:



```text

POST http://localhost:8000/query

```



\## 8.1 Dockerfile Requirement



Dockerfile should:



\* use Python 3.10 or 3.11

\* install dependencies from requirements.txt

\* copy project files

\* expose port 8000

\* run FastAPI with uvicorn



Expected command:



```bash

uvicorn api.main:app --host 0.0.0.0 --port 8000

```



\## 8.2 docker-compose.yml Requirement



docker-compose should:



\* define one service: `rag-api`

\* expose `8000:8000`

\* load `.env`

\* mount local `artifacts/` into container

\* mount local `data/` into container if needed

\* restart policy can be omitted for demo



Example behavior:



```bash

docker compose up --build

```



Then:



```bash

curl http://localhost:8000/health

```



\## 9. Artifact Handling



The API must not crash if artifacts are missing.



If artifacts are missing, `/health` should return `degraded`.



If user calls `/query` while artifacts are missing, API should return a clear error:



```json

{

&#x20; "error": "ARTIFACTS\_NOT\_FOUND",

&#x20; "message": "No retrieval artifacts found under artifacts/index. Please run indexing/build\_data\_v2.py first."

}

```



\## 10. RAG Behavior Requirements



The `/query` flow should follow this order:



```text

1\. Validate request

2\. Load config

3\. Check artifacts

4\. Run gate / scope check if implemented

5\. Run retrieval

6\. Generate answer by mock or LLM mode

7\. Return structured response with citations and debug info

```



The system must preserve evidence-oriented behavior:



\* Do not answer as if evidence exists when no relevant chunks are retrieved.

\* If query is out of scope, return `refused: true`.

\* Always expose citations when giving AML red flag assessment.

\* If LLM fails, fallback to mock response or structured error.

\* Do not hide retrieval failure.



\## 11. Mock Mode Requirements



Mock mode is required.



When `LLM\_MODE=mock`, the system should not call external APIs.



Mock mode should still return:



\* answer

\* assessment

\* identified flags if inferable from retrieved chunks

\* citations

\* debug info



Purpose: demo can run without API keys.



\## 12. README Quick Start Requirement



README must include two local execution paths.



\## 12.1 Native Python



```bash

pip install -r requirements.txt

uvicorn api.main:app --reload

```



Then open:



```text

http://localhost:8000/health

```



\## 12.2 Docker Compose



```bash

cp .env.example .env

docker compose up --build

```



Then open:



```text

http://localhost:8000/health

```



\## 13. Acceptance Criteria



The demo is considered successful when all criteria pass:



1\. `python -m compileall .` passes.

2\. `uvicorn api.main:app --reload` starts successfully.

3\. `docker compose up --build` starts successfully.

4\. `GET /health` returns service status.

5\. `POST /query` returns structured JSON.

6\. Query response includes answer, citations, assessment, and debug info.

7\. Missing artifacts produce a clear error instead of crash.

8\. Mock mode works without external API keys.

9\. README contains native and Docker quick start.

10\. Existing retrieval logic is not silently rewritten.



\## 14. Codex Implementation Constraints



When using Codex, follow these constraints:



1\. Do not rewrite the whole project at once.

2\. Do not introduce heavy frameworks.

3\. Do not modify retrieval math unless explicitly requested.

4\. Prefer wrapping existing functions over inventing new logic.

5\. Keep API response schema stable.

6\. If notebook code is messy, first extract functions into `rag\_core/`, then connect API.

7\. Every step must be runnable before moving to the next step.

8\. Do not claim a feature is implemented unless there is runnable code.



\## 15. Future Roadmap



After this FastAPI + Docker Compose demo is stable, future extensions may include:



1\. Streamlit UI client

2\. `/ingest` API

3\. document upload

4\. index versioning

5\. query logs

6\. eval automation

7\. regression tests

8\. role-based access control mock

9\. source freshness metadata

10\. multi-turn session state

\## 16. Addendum: Structured Conversation Memory (Implemented)

Roadmap item 10 (multi-turn session state) is now implemented in the service as
an **opt-in, local, in-process, bounded** layer on top of the single-turn spec
above. It does not alter the single-turn contract: requests without
`session_id` / `use_memory` behave exactly as specified in this document, and
the top-level `/query` response keys are unchanged.

Additions:

\* `POST /query` accepts additive optional fields: `session_id`, `use_memory`,
  `memory_mode` (`off` | `structured`), and `reset_memory`.

\* A deterministic, rule-based intent router produces one of three high-level
  outcomes (`route_family`): **retrieve**, **refuse**, or
  **no_retrieval_response**. Internally these expand to five fine-grained
  `intent_route` labels (`retrieve`, `retrieve_with_memory`,
  `answer_from_history`, `ask_clarifying_question`, `refuse`) for debug and
  tests. Routing never depends on a live LLM.

\* Per-session structured memory is bounded (recent turns, citations, flags, and
  retrieved chunk IDs are all capped; excerpts and summaries are truncated). It
  is never an unlimited raw transcript and is not persisted.

\* New debug fields (inside `debug`): `intent_route`, `route_family`,
  `route_reason`, `memory_used`, `memory_available`, `memory_updated`,
  `memory_turn_count`, `session_id`, `referenced_previous_answer`,
  `referenced_previous_evidence`, `active_flags`, `active_citation_count`.

\* New inspection endpoints (demo/debug only, no auth):
  `GET /sessions/{session_id}/memory` and
  `DELETE /sessions/{session_id}/memory`.

\* No LangChain / LlamaIndex / Redis / SQL / vector DB / background workers /
  external memory services were introduced, and retrieval math is unchanged.

See `docs/conversation_memory.md` for the full schema, bounds, and routing
policy.
