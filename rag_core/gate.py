"""Pre-LLM scope gate, ported from experiment_rag_v4 (balanced v3.1).

Source (migration_staging/experiment_rag_v4_source.py):
KnowledgeManifest (L423), TopicDetector (L456), GateDecision/GateResult
(L541/546), SemanticScopeClassifier (L570), pre_llm_gate (L638).

Design philosophy (unchanged from the notebook):
- the gate only handles "catastrophic" refusals (clearly out-of-scope);
- grey-area scenarios are always allowed through, with risk markers;
- insufficient evidence yields warnings, not a veto.

The rule-based path has zero ML dependencies. SemanticScopeClassifier
requires the dense backend and is experimental (off by default,
ENABLE_SEMANTIC_GATE=true to activate).
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set


class KnowledgeManifest:
    """Defines what the system's knowledge base covers."""

    def __init__(self):
        self.covered_topics: Dict[str, str] = {
            "virtual_assets": "虛擬資產/加密貨幣相關紅旗",
            "cash_structuring": "現金拆分/門檻規避",
            "rapid_movement": "快速流轉/過路帳戶",
            "third_party": "第三人代辦/人頭帳戶",
            "cross_border": "跨境高風險地區",
            "identity_mismatch": "與身分/商業模式不符",
            "shell_company": "空殼公司/受益人不透明",
        }
        self.not_covered_topics: Dict[str, str] = {
            "TBML": "貿易型洗錢（Trade-Based Money Laundering）",
            "sanctions": "制裁名單篩選",
            "tax_evasion": "稅務逃漏",
        }
        self.required_evidence: Dict[str, List[str]] = {
            "TBML": ["invoice", "customs", "shipping", "goods_flow"],
        }

    def is_topic_covered(self, topic: str) -> bool:
        return topic.lower() in [t.lower() for t in self.covered_topics]

    def is_topic_explicitly_not_covered(self, topic: str) -> bool:
        return topic.upper() in [t.upper() for t in self.not_covered_topics]

    def get_required_evidence(self, topic: str) -> List[str]:
        return self.required_evidence.get(topic.upper(), [])


class TopicDetector:
    """Rule-based keyword matching for topics and evidence fields."""

    def __init__(self):
        self.topic_keywords: Dict[str, List[str]] = {
            "TBML": [
                "貿易型洗錢", "TBML", "報關", "報關單", "發票",
                "貨物", "貨物流向", "進出口", "國際貿易",
                "信用狀", "L/C", "提單", "invoice",
                "trade-based", "trade based", "customs", "shipping",
            ],
            "sanctions": [
                "制裁", "制裁名單", "sanction", "sanctions",
            ],
            "tax_evasion": [
                "逃漏稅", "稅務逃漏", "tax evasion", "tax fraud",
            ],
            "virtual_assets": [
                "虛擬資產", "加密貨幣", "比特幣", "以太幣",
                "混幣", "錢包", "交易所", "OTC", "私下交易",
                "非託管", "隱私幣", "virtual asset", "cryptocurrency",
                "bitcoin", "ethereum", "crypto wallet", "exchange",
            ],
            "cash_structuring": [
                "拆分", "門檻", "小額", "多筆", "申報",
                "structuring", "smurfing", "reporting threshold",
            ],
            "rapid_movement": [
                "快速流轉", "入帳即轉", "過路帳戶", "多對手方",
                "很快轉出", "立即轉出", "轉出", "進出頻繁",
                "rapid movement", "rapidly transferred", "pass-through",
            ],
            "third_party": [
                "第三人", "代辦", "人頭", "代為操作",
                "third party", "nominee",
            ],
            "cross_border": [
                "跨境", "境外", "高風險地區", "匯往", "匯款",
                "cross-border", "high-risk jurisdiction",
            ],
            "identity_mismatch": [
                "與身分不符", "與職業不符", "與業務不符",
                "業務無關", "不符", "無關", "無法說明",
                "profile mismatch", "inconsistent with occupation",
            ],
            "shell_company": [
                "空殼公司", "受益人不明", "股權結構", "不透明",
                "shell company", "opaque ownership", "beneficial owner",
            ],
        }

    def detect_topics(self, text: str) -> Set[str]:
        detected = set()
        text_lower = text.lower()
        for topic, keywords in self.topic_keywords.items():
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    detected.add(topic)
                    break
        return detected

    def detect_evidence_fields(self, text: str) -> Set[str]:
        evidence_keywords = {
            "invoice": ["發票", "invoice", "單據"],
            "customs": ["報關單", "報關", "customs", "海關"],
            "shipping": ["提單", "運送", "shipping", "物流"],
            "goods_flow": ["貨物流向", "貨物", "商品", "goods"],
        }
        detected = set()
        text_lower = text.lower()
        for field_name, keywords in evidence_keywords.items():
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    detected.add(field_name)
                    break
        return detected

    def detect_explicit_knowledge_gap(self, text: str) -> bool:
        gap_patterns = [
            r"教材沒有",
            r"沒有.*章節",
            r"沒有.*資料",
            r"手上沒有",
            r"缺乏.*文件",
            r"\bno .*materials?\b",
            r"\black(?:s|ing)? .*documentation\b",
        ]
        return any(re.search(p, text, flags=re.IGNORECASE) for p in gap_patterns)


class GateDecision(Enum):
    ALLOW = "ALLOW"
    REFUSE = "REFUSE"


@dataclass
class GateResult:
    decision: GateDecision
    reason_code: str = ""
    reason_message: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def allowed(self) -> bool:
        return self.decision == GateDecision.ALLOW

    @property
    def decision_label(self) -> str:
        """Lowercase label for API serialization (spec: "allow"/"refuse")."""
        return self.decision.value.lower()


class SemanticScopeClassifier:
    """Embedding-distance scope check. EXPERIMENTAL — requires the dense
    backend; threshold 0.35 was tuned only on the notebook corpus."""

    def __init__(self, embedding_model, threshold: float = 0.35):
        self.embedding_model = embedding_model
        self.threshold = threshold
        self.anchor_sentences: List[str] = [
            "客戶使用加密貨幣進行大額交易",
            "虛擬資產交易所發現可疑的混幣行為",
            "Suspicious cryptocurrency transactions involving mixing services",
            "客戶分批存入現金，每次金額剛好低於申報門檻",
            "Multiple cash deposits structured to avoid reporting thresholds",
            "資金入帳後立即轉出到其他帳戶",
            "帳戶收到大額款項後迅速流轉至多個對手方",
            "第三人代為開戶並操作帳戶交易",
            "帳戶由他人代辦，本人無法說明交易目的",
            "資金頻繁匯往高風險國家或地區",
            "Cross-border wire transfers to jurisdictions with weak AML controls",
            "交易模式與客戶申報的職業和收入明顯不符",
            "帳戶交易量遠超其商業規模所能解釋的範圍",
            "公司受益人結構不透明，無法確認最終控制人",
            "Shell company with no apparent business operations receiving large transfers",
        ]
        self._anchor_embeddings = None

    def _ensure_anchors_encoded(self):
        if self._anchor_embeddings is None:
            self._anchor_embeddings = self.embedding_model.encode(
                self.anchor_sentences,
                normalize_embeddings=True,
                show_progress_bar=False,
            )

    def compute_scope_similarity(self, query: str):
        import numpy as np

        self._ensure_anchors_encoded()
        query_embedding = self.embedding_model.encode(
            [query], normalize_embeddings=True, show_progress_bar=False
        )
        similarities = np.dot(self._anchor_embeddings, query_embedding.T).flatten()
        max_idx = int(np.argmax(similarities))
        return float(similarities[max_idx]), self.anchor_sentences[max_idx]

    def is_in_scope(self, query: str) -> bool:
        max_sim, _ = self.compute_scope_similarity(query)
        return max_sim > self.threshold


def pre_llm_gate(
    scenario: str,
    manifest: Optional[KnowledgeManifest] = None,
    detector: Optional[TopicDetector] = None,
    scope_classifier: Optional[SemanticScopeClassifier] = None,
) -> GateResult:
    """Pre-LLM gate (balanced v3.1), ported verbatim from L638."""
    if manifest is None:
        manifest = KnowledgeManifest()
    if detector is None:
        detector = TopicDetector()

    detected_topics = detector.detect_topics(scenario)

    # Rule 1: scenario explicitly states a knowledge gap
    if detector.detect_explicit_knowledge_gap(scenario):
        uncovered = [
            t for t in detected_topics
            if manifest.is_topic_explicitly_not_covered(t)
        ]
        if uncovered:
            return GateResult(
                decision=GateDecision.REFUSE,
                reason_code="EXPLICIT_KNOWLEDGE_GAP",
                reason_message=(
                    f"情境明確表示缺乏 {', '.join(uncovered)} 相關教材，無法進行分析"
                ),
                metadata={
                    "detected_topics": sorted(detected_topics),
                    "uncovered_topics": sorted(uncovered),
                },
            )

    covered_topics = [t for t in detected_topics if manifest.is_topic_covered(t)]
    explicitly_not_covered = [
        t for t in detected_topics if manifest.is_topic_explicitly_not_covered(t)
    ]

    # Rule 2: topic entirely outside system scope
    if explicitly_not_covered and not covered_topics:
        return GateResult(
            decision=GateDecision.REFUSE,
            reason_code="TOPIC_NOT_COVERED",
            reason_message=(
                f"本系統未包含 {', '.join(explicitly_not_covered)} 相關教材"
            ),
            metadata={
                "detected_topics": sorted(detected_topics),
                "explicitly_not_covered": sorted(explicitly_not_covered),
            },
        )

    # Rule 3: insufficient evidence → warnings, not a veto
    evidence_warnings = []
    present_evidence = detector.detect_evidence_fields(scenario)
    for topic in detected_topics:
        required = manifest.get_required_evidence(topic)
        if not required:
            continue
        missing = set(required) - present_evidence
        missing_ratio = len(missing) / len(required)
        if missing_ratio > 0:
            evidence_warnings.append({
                "topic": topic,
                "missing_evidence": sorted(missing),
                "missing_ratio": round(missing_ratio, 2),
                "severity": (
                    "high" if missing_ratio > 0.5
                    else "medium" if missing_ratio > 0.25
                    else "low"
                ),
            })

    # Rule 4 (experimental): semantic out-of-scope detection
    if not detected_topics and scope_classifier is not None:
        if not scope_classifier.is_in_scope(scenario):
            sim, nearest = scope_classifier.compute_scope_similarity(scenario)
            return GateResult(
                decision=GateDecision.REFUSE,
                reason_code="SEMANTIC_OUT_OF_SCOPE",
                reason_message="此問題不在本系統的知識範圍內（AML 紅旗偵測）",
                metadata={
                    "detected_topics": [],
                    "semantic_similarity": round(sim, 4),
                    "nearest_anchor": nearest[:80],
                    "threshold": scope_classifier.threshold,
                    "gate_mode": "balanced_v3.1_semantic",
                },
            )

    return GateResult(
        decision=GateDecision.ALLOW,
        metadata={
            "detected_topics": sorted(detected_topics),
            "covered_topics": sorted(covered_topics),
            "explicitly_not_covered": sorted(explicitly_not_covered),
            "evidence_warnings": evidence_warnings,
            "gate_mode": "balanced_v3",
        },
    )


def check_scope(
    query: str,
    retrieved_chunks: Optional[List[Dict[str, Any]]] = None,
    scope_classifier: Optional[SemanticScopeClassifier] = None,
) -> GateResult:
    """Public gate interface for the pipeline.

    Conservative baseline: allow AML-related queries; refuse only when a
    query is clearly outside the knowledge scope; when uncertain, allow.
    ``retrieved_chunks`` is accepted for future evidence-aware checks
    (currently unused — the notebook gate is purely query-based).
    """
    return pre_llm_gate(query, scope_classifier=scope_classifier)
