# 專案面試說明地圖

本文協助你理解並在面試中解釋這個 repo 的核心內容。

---

## 這個專案一句話是什麼

一套以 FastAPI 實作的 RAG demo 服務，使用 AML 紅旗辨識作為場景，展示 AI 工程中
常見的問題：如何根據用戶提問從知識庫中檢索相關依據、篩選後生成有引用的回答、
管理多輪對話狀態，以及確保系統在外部依賴失敗時能有意義地降級。

---

## 三個層次的說明

### 1. 對 HR 說

> 「這是一個 AI 工程 Portfolio 專案。我用 FastAPI 建了一個可以回答洗錢相關問題的
> 查詢服務，它會從知識庫中找到相關規則、判斷是否有紅旗、然後給出有來源引用的
> 答案。我也加了多輪對話、自動測試、以及本地 LLM 的整合，主要目的是展示我對
> RAG 系統工程的理解。」

### 2. 對技術面試官說

> 「這是一個 FastAPI RAG 服務，整合 BM25、Dense FAISS、Hybrid RRF 三種檢索模式，
> 並在進入 LLM 前設了一層規則型 Gate 做範圍過濾。生成端支援 Mock、Groq、Gemini、
> Ollama 多條路徑，各自有降級機制。我另外加了 opt-in 的結構化多輪對話記憶，搭配
> 確定性 Intent Routing 判斷每一輪要做什麼。測試有 121 個，另有多輪評估腳本驗證
> 4 個固定對話情境的路由與記憶行為。」

### 3. 實作層次說明

> 「架構上，FastAPI 的 `/query` endpoint 收到請求後，先過 Pre-LLM Gate 做範圍過濾。
> 如果啟用了 `use_memory`，則先跑 Intent Router 決定這一輪要走哪條路徑（新查詢、
> 從記憶回覆、要求澄清、或拒答）。通過 Gate 的請求進入 BM25 / Dense / Hybrid RRF
> 檢索，再把 retrieved chunks 傳給生成端。生成端可以是 Mock（確定性、不需 Key）、
> Groq、Gemini/Gemma、或本地 Ollama，任何一個失敗都 fallback 到 Mock。最後回傳
> assessment、identified_flags、citations 和 debug 欄位。」

---

## 系統流程（白話版）

```
用戶輸入查詢
    ↓
[是否啟用 Memory？]
    ├─ 否 → 直接進 Gate
    └─ 是 → 先跑 Intent Router
             ├─ retrieve：走正常查詢流程
             ├─ retrieve_with_memory：帶記憶做查詢
             ├─ answer_from_history：從結構化記憶回答（不做新檢索）
             ├─ ask_clarifying_question：問題太模糊，要求澄清
             └─ refuse：範圍外，拒答
                  ↓
[Pre-LLM Gate]（規則型範圍過濾）
    ├─ 範圍外 → 拒答，不進檢索
    └─ 範圍內
         ↓
[檢索：BM25 / Dense FAISS / Hybrid RRF]
    ↓
[生成：Mock / Groq / Gemini / Ollama]（任何失敗 → fallback Mock）
    ↓
回傳：assessment + flags + citations + debug
    ↓
[若啟用 Memory] → 更新結構化記憶狀態
```

---

## 各主要功能是做什麼的

| 功能 | 說明 |
|---|---|
| **FastAPI API** | 對外的 HTTP 介面，提供 `/health`、`/query`、`/sources`；設計為可從外部工具或 UI 調用 |
| **BM25 檢索** | 關鍵字型稀疏檢索，在 startup 從 chunks.json 建立，輕量、不需 embedding 模型 |
| **Dense FAISS 檢索** | 語意型向量檢索，使用多語言 embedding 模型；不可用時自動降級回 BM25 |
| **Hybrid RRF** | 結合 BM25 和 Dense 排名，用 `1/(60+rank)` 公式融合，再加文件層級加權 |
| **Pre-LLM Gate** | 在進入 LLM 前過濾超出範圍的問題（如制裁、稅務逃漏），直接拒答，節省 LLM 成本並防止模型發散 |
| **Mock 生成** | 確定性回傳，不需 API Key，不需網路，是預設模式；讓測試和 demo 可重現 |
| **Live 生成（Groq/Gemini）** | 接真實 LLM API，用於實際語意生成驗證；失敗自動 fallback mock |
| **Ollama 本地模式** | 接本地 Ollama HTTP server，讓開發者在本機驗證本地 LLM 的接入；不是品質 benchmark |
| **結構化對話記憶** | 記錄活躍情境、紅旗、引用、prior answer summary 等結構化狀態；有界限、不是無限 transcript |
| **Intent Routing** | 確定性規則判斷每一輪的路徑；不依賴 LLM，可重現，易測試 |
| **Citations** | 每個回答附上 chunk-level 的來源引用，讓審查員能追溯依據 |
| **Debug 欄位** | 記錄 fallback 原因、路由結果、記憶狀態等，協助開發者診斷問題 |
| **pytest（121 tests）** | API contract tests，驗證每個端點的行為、拒答邏輯、schema、降級行為 |
| **Multi-turn eval** | 驅動 4 個固定對話情境，驗證路由與記憶行為是否符合規格 |
| **CQC / Failure Diagnostics** | 跨查詢一致性評估 + 可觀察失敗分類，幫助找出系統中的不穩定點 |

---

## 如何解釋多輪評估結果

```
multi-turn eval: 4 / 4 sessions passed
```

**白話說法：**

> 「我設計了 4 個固定的對話情境，分別是：建立 AML 情境後要求回溯紅旗、模糊提問
> 觸發澄清要求、範圍外問題觸發拒答、以及要求回溯先前引用。這 4 個情境全部按預期
> 完成，路由選擇和記憶內容都符合期待值。」

不要說：「我的 AI 通過了 4/4 的評估。」
要說：「這個評估腳本驗證了 4 個固定情境下的路由與記憶行為是否符合規格。」

---

## 常見面試問題與建議回答

### 「這個專案在做什麼？」

> 「這是一個 RAG demo 服務，用 AML 紅旗辨識作為場景。用戶輸入一個問題，系統先
> 判斷問題是否在範圍內，然後從知識庫中檢索相關規則，最後生成一個有引用依據的
> 判斷。我也加了多輪對話記憶和自動評估，主要用來展示 RAG 系統的工程設計。」

### 「為什麼選 AML？」

> 「因為 AML 紅旗辨識這個場景對 RAG 設計來說很有趣：它有明確的範圍邊界（某些
> 問題應該拒答）、需要引用依據（不能憑空回答）、有中英文混合的查詢需求。這些
> 特點讓我可以展示 Gate、混合檢索、以及引用設計。這不是因為我是 AML 專家。」

### 「你做的是 RAG 還是 Agent？」

> 「這是 RAG 為主的系統，不是 Agent。它不會主動規劃多步行動，也不會自己決定要
> 用哪個工具。Intent Router 是確定性規則，不是 LLM 自主決策。多輪記憶是有結構
> 的狀態管理，不是 Agent 的 tool loop。」

### 「多輪對話怎麼做？」

> 「用結構化記憶，不是無限的 transcript。每個 session 有一個有界限的記憶物件，
> 記錄活躍情境摘要、去重複後的紅旗清單、引用、prior answer summary 等欄位。
> 每一輪先跑 Intent Router 決定要做什麼：新查詢就做正常 RAG；問到之前的答案
> 就從 prior_answer_summary 回覆；問題太模糊就問澄清；範圍外就拒答。這樣的設計
> 讓每一輪的行為可預期、可測試。」

### 「Ollama 在這裡扮演什麼角色？」

> 「Ollama 是一條可選的本地生成路徑。它讓開發者在本機接一個本地 LLM，驗證接入
> 是否正確。這不是模型品質的比較，也不是 benchmark。Mock 仍然是預設模式，測試
> 也不依賴 Ollama。」

### 「你怎麼驗證？」

> 「有三層：第一是 121 個 pytest，驗證 API contract、schema、降級行為、拒答邏輯。
> 第二是 multi-turn eval，驅動 4 個固定情境驗證路由和記憶行為。第三是可選的
> CQC 跨查詢一致性評估和 Failure Diagnostics，用來找出不穩定點。」

### 「這離 production 還差什麼？」

> 「差很多，而且我沒有打算做成 production 系統。目前缺的包括：認證授權、稽核
> 日誌、記憶的跨 worker 持久化、真實的 AML 法規語料覆蓋，以及嚴格的模型品質
> 評估。這是一個 demo 和工程能力展示，不是 production AML 合規平台。」

### 「你本人主要貢獻是什麼？」

> 「整個 RAG pipeline 的設計與實作：從 FastAPI 架構、多種檢索模式的整合與降級、
> Pre-LLM Gate，到多 LLM 後端的接入。結構化多輪記憶和 Intent Router 是後來疊加
> 的功能，是在單輪版本穩定之後加上去的。評估腳本和測試也是我設計的。」

---

## 可以安全說的 vs. 應該避免說的

### 可以說的

- 這是一個 RAG 工程 demo，展示 BM25 / Dense / Hybrid 檢索、Gate、生成、引用等功能
- 這個專案通過了 121 個 pytest cases
- multi-turn eval 驗證了 4 個固定情境的路由與記憶行為
- 我設計了結構化的多輪記憶，有界限、採用確定性路由
- 這個系統在外部依賴失敗時會 fallback 並記錄原因

### 應該避免說的

- ~~「這是一個 AML 合規系統」~~ — 這是 demo，不是合規工具
- ~~「通過了模型品質評估」~~ — 我們做的是行為與路由測試，不是品質 benchmark
- ~~「可以直接 production 部署」~~ — 缺少認證、稽核等 production 要件
- ~~「AI 做了路由決策」~~ — 路由是確定性規則，不是 LLM
- ~~「達到了業界水準的 AML 覆蓋率」~~ — 語料是示範用小型資料集
