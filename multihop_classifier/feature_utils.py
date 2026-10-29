import re

import jieba.posseg as pseg

COMPARISON_PATTERN = re.compile(
    r"相比|更|区别|差异|对比|区别是什么|versus|vs\.?|compared with|difference",
    re.IGNORECASE,
)
LOGIC_PATTERN = re.compile(
    r"首先|然后|接着|最后|之前|之后|如果|则|因为|所以|由于|导致|因此|"
    r"先.+再|在.+之后|first|then|before|after|because|therefore",
    re.IGNORECASE,
)
BRIDGE_PATTERN = re.compile(r"通过|根据|对应|涉及|其中|基于|所在|属于|related to|which", re.IGNORECASE)


def extract_features(question: str) -> dict:
    entities: list[str] = []
    for word, flag in pseg.cut(question):
        term = word.strip()
        if not term:
            continue
        if flag.startswith(("nr", "ns", "nt", "nz", "eng")) or (len(term) >= 2 and flag.startswith("n")):
            if term not in entities:
                entities.append(term)
    comparison = bool(COMPARISON_PATTERN.search(question))
    logic = bool(LOGIC_PATTERN.search(question))
    bridge_count = max(0, len(entities) - 1) if comparison or logic or BRIDGE_PATTERN.search(question) else 0
    return {
        "comparison_flag": comparison,
        "logic_connector_flag": logic,
        "bridge_entity_count": bridge_count,
        "entity_count": len(entities),
        "token_length": len(re.findall(r"[\u3400-\u9fff]|[A-Za-z0-9_]+|[^\s]", question)),
    }


def feature_vector(features: dict) -> list[float]:
    return [
        float(bool(features.get("comparison_flag"))),
        float(bool(features.get("logic_connector_flag"))),
        float(features.get("bridge_entity_count", 0)),
        float(features.get("entity_count", 0)),
        float(features.get("token_length", 0)) / 100.0,
    ]
