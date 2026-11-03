import type { Node } from "unist";
import type { Plugin } from "unified";
import type { Chunk } from "../types";

export interface SourceLocation {
  start: number;
  end: number;
  kind: "chunk" | "parent" | "section";
  label: string;
}

interface MatchSpan {
  start: number;
  end: number;
  normalizedStart: number;
}

interface SourceMatcher {
  normalizedLength: number;
  find: (value: string) => MatchSpan[];
}

function comparisonText(value: string): { text: string; positions: number[] } {
  const chars: string[] = [];
  const positions: number[] = [];
  for (let index = 0; index < value.length; index += 1) {
    const char = value[index];
    if (/\s/u.test(char)) continue;
    for (const lowered of char.toLowerCase()) {
      chars.push(lowered);
      positions.push(index);
    }
  }
  return { text: chars.join(""), positions };
}

function createMatcher(source: string): SourceMatcher {
  const normalized = comparisonText(source);
  return {
    normalizedLength: normalized.text.length,
    find(value: string): MatchSpan[] {
      const needle = comparisonText(value).text;
      if (!needle) return [];
      const matches: MatchSpan[] = [];
      let normalizedStart = normalized.text.indexOf(needle);
      while (normalizedStart >= 0) {
        const sourceStart = normalized.positions[normalizedStart];
        const sourceEnd = normalized.positions[normalizedStart + needle.length - 1];
        if (sourceStart !== undefined && sourceEnd !== undefined) {
          matches.push({ start: sourceStart, end: sourceEnd + 1, normalizedStart });
        }
        normalizedStart = normalized.text.indexOf(needle, normalizedStart + needle.length);
      }
      return matches;
    },
  };
}

function isL2(chunk: Chunk): boolean {
  return chunk.granularity_level === "L2";
}

function orderedL2(chunks: Chunk[]): Chunk[] {
  return chunks.filter(isL2).sort((left, right) => left.chunk_index - right.chunk_index);
}

function nearestMatch(
  matches: MatchSpan[],
  normalizedLength: number,
  expectedRatio: number,
): MatchSpan | null {
  if (!matches.length) return null;
  if (matches.length === 1 || normalizedLength <= 0) return matches[0];
  const expected = normalizedLength * expectedRatio;
  return matches.reduce((best, current) => (
    Math.abs(current.normalizedStart - expected) < Math.abs(best.normalizedStart - expected)
      ? current
      : best
  ));
}

function matchForChunk(
  matcher: SourceMatcher,
  chunk: Chunk,
  l2Chunks: Chunk[],
): MatchSpan | null {
  const matches = matcher.find(chunk.content);
  if (!matches.length) return null;
  const index = l2Chunks.findIndex((item) => item.id === chunk.id);
  const expectedRatio = index >= 0 ? (index + 0.5) / l2Chunks.length : 0.5;
  return nearestMatch(matches, matcher.normalizedLength, expectedRatio);
}

function parentIds(chunk: Chunk): string[] {
  const ids: string[] = [];
  const raw = chunk.metadata_json?.parent_chunk_ids;
  if (Array.isArray(raw)) {
    for (const value of raw) {
      if (typeof value === "string" && !ids.includes(value)) ids.push(value);
    }
  }
  if (chunk.parent_chunk_id && !ids.includes(chunk.parent_chunk_id)) {
    ids.push(chunk.parent_chunk_id);
  }
  return ids;
}

function sourceRange(spans: MatchSpan[]): { start: number; end: number } | null {
  if (!spans.length) return null;
  return {
    start: Math.min(...spans.map((span) => span.start)),
    end: Math.max(...spans.map((span) => span.end)),
  };
}

function parentLocation(
  matcher: SourceMatcher,
  chunk: Chunk,
  chunks: Chunk[],
  l2Chunks: Chunk[],
  fallbackLabel: string,
): SourceLocation | null {
  const parents = parentIds(chunk)
    .map((id) => chunks.find((item) => item.id === id))
    .filter((item): item is Chunk => !!item);
  const spans = parents
    .map((parent) => matchForChunk(matcher, parent, l2Chunks))
    .filter((span): span is MatchSpan => !!span);
  const range = sourceRange(spans);
  return range ? { ...range, kind: "parent", label: fallbackLabel } : null;
}

function sectionLocation(
  matcher: SourceMatcher,
  chunk: Chunk,
  l2Chunks: Chunk[],
): SourceLocation | null {
  const peers = l2Chunks.filter((item) => item.section_path === chunk.section_path);
  if (!peers.length) return null;
  const spans = peers
    .map((peer) => matchForChunk(matcher, peer, l2Chunks))
    .filter((span): span is MatchSpan => !!span);
  const range = sourceRange(spans);
  return range ? { ...range, kind: "section", label: "当前分块所属章节范围" } : null;
}

export function locateChunkInSource(
  source: string,
  selected: Chunk | null,
  chunks: Chunk[],
): SourceLocation | null {
  if (!source || !selected) return null;
  const matcher = createMatcher(source);
  const l2Chunks = orderedL2(chunks);

  if (selected.granularity_level === "L1") {
    const section = parentLocation(
      matcher,
      selected,
      chunks,
      l2Chunks,
      "L1 章节摘要关联的原文范围",
    ) ?? sectionLocation(matcher, selected, l2Chunks);
    if (section) return section;
  }

  const direct = matchForChunk(matcher, selected, l2Chunks);
  const parentId = selected.parent_chunk_id;
  const parent = parentId ? chunks.find((item) => item.id === parentId) : null;
  const parentSpan = parent ? matchForChunk(matcher, parent, l2Chunks) : null;

  if (direct) {
    if (parentSpan && selected.granularity_level === "L3") {
      const insideParent = matcher
        .find(selected.content)
        .find((span) => span.start < parentSpan.end && span.end > parentSpan.start);
      if (insideParent) {
        return { ...insideParent, kind: "chunk", label: "当前原子块原文" };
      }
    }
    return {
      ...direct,
      kind: "chunk",
      label: selected.granularity_level === "L3" ? "当前原子块原文" : "当前分块原文",
    };
  }

  if (parentSpan) {
    return { ...parentSpan, kind: "parent", label: "当前分块对应的父段落范围" };
  }

  return parentLocation(
    matcher,
    selected,
    chunks,
    l2Chunks,
    "当前分块关联的父级原文",
  ) ?? sectionLocation(matcher, selected, l2Chunks);
}

interface MarkdownNode {
  type: string;
  value?: string;
  children?: MarkdownNode[];
  position?: {
    start?: { offset?: number };
    end?: { offset?: number };
  };
  data?: {
    hName?: string;
    hProperties?: Record<string, unknown>;
  };
}

function markText(value: string): MarkdownNode {
  return {
    type: "emphasis",
    data: {
      hName: "mark",
      hProperties: { "data-current-chunk": "true" },
    },
    children: [{ type: "text", value }],
  };
}

function markNode(node: MarkdownNode): MarkdownNode {
  return {
    ...node,
    data: {
      ...node.data,
      hProperties: {
        ...node.data?.hProperties,
        "data-current-chunk": "true",
      },
    },
  };
}

function highlightNode(node: MarkdownNode, start: number, end: number): MarkdownNode[] {
  if (node.children) {
    node.children = node.children.flatMap((child) => highlightNode(child, start, end));
  }

  const nodeStart = node.position?.start?.offset;
  const nodeEnd = node.position?.end?.offset;
  if (nodeStart === undefined || nodeEnd === undefined || nodeEnd <= start || nodeStart >= end) {
    return [node];
  }

  if (node.type === "code" || node.type === "inlineCode") {
    return [markNode(node)];
  }
  if (node.type !== "text" || !node.value) return [node];

  const localStart = Math.max(0, start - nodeStart);
  const localEnd = Math.min(node.value.length, end - nodeStart);
  if (localEnd <= localStart) return [node];

  const result: MarkdownNode[] = [];
  if (localStart > 0) result.push({ type: "text", value: node.value.slice(0, localStart) });
  result.push(markText(node.value.slice(localStart, localEnd)));
  if (localEnd < node.value.length) result.push({ type: "text", value: node.value.slice(localEnd) });
  return result;
}

export function remarkSourceHighlight(start: number, end: number): Plugin<[], Node> {
  return () => (tree) => {
    const root = tree as unknown as MarkdownNode;
    highlightNode(root, start, end);
  };
}