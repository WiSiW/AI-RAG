from types import SimpleNamespace

from app.chunking.multigranularity import (
    SectionSummary,
    build_l1_specs,
    build_l3_specs,
    build_section_units,
)


def l2_row(
    chunk_id: str,
    content: str,
    *,
    section: str = "第一章",
    index: int = 0,
) -> SimpleNamespace:
    return SimpleNamespace(
        id=chunk_id,
        chunk_index=index,
        content=content,
        section_path=section,
        summary=f"摘要-{chunk_id}",
        keywords=["糖尿病", "参数"],
        metadata_json={"pages": [2]},
        granularity_level="L2",
    )


def test_section_units_and_l1_summary_keep_parent_links():
    rows = [
        l2_row("l2-a", "第一段内容", index=0),
        l2_row("l2-b", "第二段内容", index=1),
    ]

    units = build_section_units(rows)
    specs = build_l1_specs(
        units,
        {"第一章": SectionSummary("第一章的完整摘要")},
    )

    assert len(units) == 1
    assert specs[0].granularity_level == "L1"
    assert specs[0].granule_type == "section_summary"
    assert specs[0].content.startswith("第一章")
    assert specs[0].metadata["parent_chunk_ids"] == ["l2-a", "l2-b"]
    assert specs[0].metadata["pages"] == [2]


def test_l3_extracts_table_error_and_key_sentence_with_parent_metadata():
    row = l2_row(
        "l2-a",
        "| 参数 | 值 |\nERROR 500 服务异常\n糖尿病需要监测血糖和饮食。",
        index=3,
    )

    specs = build_l3_specs([row])
    contents = [spec.content for spec in specs]
    types = {spec.granule_type for spec in specs}

    assert "| 参数 | 值 |" in contents
    assert any("ERROR 500" in content for content in contents)
    assert any("糖尿病" in content for content in contents)
    assert {"table_row", "error_line"}.issubset(types)
    assert all(spec.granularity_level == "L3" for spec in specs)
    assert all(spec.parent_chunk_id == "l2-a" for spec in specs)
    assert all(spec.metadata["pages"] == [2] for spec in specs)
