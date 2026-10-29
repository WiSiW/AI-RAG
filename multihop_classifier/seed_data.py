import json
from pathlib import Path

single_templates = [
    "什么是{entity}？",
    "{entity}的定义是什么？",
    "{entity}有哪些参数？",
]
multi_templates = [
    "{entity1}与{entity2}相比有什么区别？",
    "先找到{entity1}，再根据它查询{entity2}。",
    "如果{entity1}发生变化，{entity2}会受到什么影响？",
    "通过{entity1}的配置如何定位{entity2}？",
]
entities = ["设备A", "设备B", "安装流程", "错误码500", "电源模块", "监控指标", "知识库", "检索服务"]


def generate(output: Path, repeats: int = 20):
    rows = []
    for repeat in range(repeats):
        for index, entity in enumerate(entities):
            rows.append({"question": single_templates[repeat % len(single_templates)].format(entity=entity), "label": "single"})
            other = entities[(index + repeat + 1) % len(entities)]
            rows.append({"question": multi_templates[repeat % len(multi_templates)].format(entity1=entity, entity2=other), "label": "multi"})
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
    print(f"wrote {len(rows)} seed rows to {output}; these require manual review")


if __name__ == "__main__":
    generate(Path("data/multihop_seed.jsonl"))
