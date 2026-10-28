import { ChangeEvent, DragEvent, FormEvent, useEffect, useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { CheckCircle2, FileArchive, FileText, LoaderCircle, Server, Trash2, UploadCloud, XCircle } from "lucide-react";
import { Link, useSearchParams } from "react-router-dom";
import { api } from "../api/client";
import { ErrorState, LoadingState, PageHeader, formatBytes } from "../components/UI";
import type { BatchUploadItem, BatchUploadResponse } from "../types";

const accepted = ".pdf,.docx,.md,.markdown,.epub,.zip,.tar,.tgz,.tar.gz";

function isArchive(file: File): boolean {
  return /\.(zip|tar|tgz|tar\.gz)$/i.test(file.name);
}

export default function UploadPage() {
  const [searchParams] = useSearchParams();
  const queryClient = useQueryClient();
  const kbQuery = useQuery({ queryKey: ["knowledge-bases"], queryFn: api.listKnowledgeBases });
  const capabilitiesQuery = useQuery({ queryKey: ["capabilities"], queryFn: api.getCapabilities });
  const [kbId, setKbId] = useState(searchParams.get("kb") ?? "");
  const [files, setFiles] = useState<File[]>([]);
  const [chunkMode, setChunkMode] = useState("structure");
  const [chunkSize, setChunkSize] = useState("512");
  const [chunkOverlap, setChunkOverlap] = useState("64");
  const [tags, setTags] = useState("");
  const [serverPath, setServerPath] = useState("");
  const [result, setResult] = useState<BatchUploadResponse | null>(null);

  const knowledgeBases = kbQuery.data ?? [];
  const capabilities = capabilitiesQuery.data;
  const selectedKb = useMemo(() => knowledgeBases.find((kb) => kb.id === kbId), [knowledgeBases, kbId]);

  useEffect(() => {
    if (!kbId && knowledgeBases.length) setKbId(knowledgeBases[0].id);
  }, [kbId, knowledgeBases]);

  useEffect(() => {
    if (!selectedKb) return;
    setChunkMode(String(selectedKb.settings.chunk_mode ?? "structure"));
    setChunkSize(String(selectedKb.settings.chunk_size ?? 512));
    setChunkOverlap(String(selectedKb.settings.chunk_overlap ?? 64));
  }, [selectedKb]);

  const options = {
    chunkMode,
    chunkSize: Number(chunkSize),
    chunkOverlap: Number(chunkOverlap),
    tags,
  };

  const uploadMutation = useMutation({
    mutationFn: async () => {
      const regular = files.filter((file) => !isArchive(file));
      const archives = files.filter(isArchive);
      const items: BatchUploadItem[] = [];
      if (regular.length) {
        const response = await api.uploadDocuments(kbId, regular, options);
        items.push(...response.items);
      }
      for (const archive of archives) {
        try {
          const response = await api.uploadArchive(kbId, archive, options);
          items.push(...response.items);
        } catch (error) {
          items.push({
            filename: archive.name,
            ok: false,
            document: null,
            job: null,
            error: (error as Error).message,
          });
        }
      }
      const accepted = items.filter((item) => item.ok).length;
      return { items, accepted, failed: items.length - accepted };
    },
    onSuccess: async (data) => {
      setResult(data);
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: ["documents", kbId] }),
        queryClient.invalidateQueries({ queryKey: ["knowledge-bases"] }),
      ]);
    },
  });

  const serverArchiveMutation = useMutation({
    mutationFn: () => api.importServerArchive(kbId, {
      path: serverPath.trim(),
      chunk_mode: chunkMode,
      chunk_size: Number(chunkSize),
      chunk_overlap: Number(chunkOverlap),
      tags,
    }),
    onSuccess: async (data) => {
      setResult(data);
      await queryClient.invalidateQueries({ queryKey: ["documents", kbId] });
    },
  });

  function addFiles(nextFiles: File[]) {
    setResult(null);
    setFiles((current) => {
      const known = new Set(current.map((file) => `${file.name}-${file.size}`));
      return [...current, ...nextFiles.filter((file) => !known.has(`${file.name}-${file.size}`))];
    });
  }

  function chooseFiles(event: ChangeEvent<HTMLInputElement>) {
    addFiles(Array.from(event.target.files ?? []));
    event.target.value = "";
  }

  function dropFiles(event: DragEvent<HTMLLabelElement>) {
    event.preventDefault();
    addFiles(Array.from(event.dataTransfer.files ?? []));
  }

  function submit(event: FormEvent) {
    event.preventDefault();
    if (files.length && kbId) uploadMutation.mutate();
  }

  function submitServerArchive(event: FormEvent) {
    event.preventDefault();
    if (serverPath.trim() && kbId) serverArchiveMutation.mutate();
  }

  if (kbQuery.isLoading) return <LoadingState />;
  if (kbQuery.error) return <ErrorState message={kbQuery.error.message} />;

  const results = result?.items ?? [];
  const busy = uploadMutation.isPending || serverArchiveMutation.isPending;

  return (
    <div className="mx-auto max-w-6xl">
      <PageHeader title="批量上传文档" description="支持多文件、ZIP/TAR 压缩包页面导入，以及服务器导入目录中的压缩包。大文件采用流式落盘，避免整体读入内存。" />
      <div className="grid gap-5 lg:grid-cols-[1fr_340px]">
        <div className="space-y-5">
          <form onSubmit={submit} className="panel p-5 sm:p-6">
            <div className="grid gap-4 sm:grid-cols-2">
              <div>
                <label className="label">目标知识库</label>
                <select className="input" value={kbId} onChange={(event) => setKbId(event.target.value)} required>
                  {knowledgeBases.map((kb) => <option key={kb.id} value={kb.id}>{kb.name}</option>)}
                </select>
              </div>
              <div>
                <label className="label">分块模式</label>
                <select className="input" value={chunkMode} onChange={(event) => setChunkMode(event.target.value)}>
                  {(capabilities?.chunk_modes ?? [{ value: "structure", label: "结构感知", description: "" }]).map((mode) => (
                    <option key={mode.value} value={mode.value}>{mode.label}</option>
                  ))}
                </select>
                <p className="mt-1 text-xs text-slate-400">{capabilities?.chunk_modes.find((mode) => mode.value === chunkMode)?.description}</p>
              </div>
            </div>

            <label
              className="group mt-5 flex min-h-56 cursor-pointer flex-col items-center justify-center rounded-2xl border-2 border-dashed border-slate-200 bg-slate-50/70 px-6 text-center transition hover:border-brand-300 hover:bg-brand-50/50"
              onDragOver={(event) => event.preventDefault()}
              onDrop={dropFiles}
            >
              <UploadCloud className="h-10 w-10 text-slate-400 group-hover:text-brand-500" />
              <div className="mt-3 font-medium text-slate-700">选择或拖入多个文件 / 压缩包</div>
              <div className="mt-1 text-xs text-slate-500">
                PDF / DOCX / Markdown / EPUB；支持 ZIP、TAR、TAR.GZ；单文件最大 {capabilities?.upload_max_mb ?? 2048} MB
              </div>
              <input type="file" multiple accept={accepted} onChange={chooseFiles} className="hidden" />
            </label>

            {!!files.length && (
              <div className="mt-4 space-y-2">
                {files.map((file, index) => (
                  <div key={`${file.name}-${file.size}`} className="flex items-center gap-3 rounded-xl border border-slate-200 bg-white p-3">
                    {isArchive(file) ? <FileArchive className="h-5 w-5 text-amber-600" /> : <FileText className="h-5 w-5 text-brand-600" />}
                    <div className="min-w-0 flex-1">
                      <div className="truncate text-sm font-medium text-slate-800">{file.name}</div>
                      <div className="text-xs text-slate-400">{formatBytes(file.size)} · {isArchive(file) ? "服务端解包" : "文档"}</div>
                    </div>
                    <button type="button" onClick={() => setFiles((current) => current.filter((_, i) => i !== index))} className="rounded-lg p-2 text-slate-400 hover:bg-red-50 hover:text-red-600" aria-label="移除文件">
                      <Trash2 className="h-4 w-4" />
                    </button>
                  </div>
                ))}
              </div>
            )}

            <div className="mt-5 grid gap-4 sm:grid-cols-3">
              <div><label className="label">分块大小（tokens）</label><input className="input" type="number" min={32} max={8192} value={chunkSize} onChange={(event) => setChunkSize(event.target.value)} /></div>
              <div><label className="label">重叠大小（tokens）</label><input className="input" type="number" min={0} value={chunkOverlap} onChange={(event) => setChunkOverlap(event.target.value)} /></div>
              <div><label className="label">标签</label><input className="input" value={tags} onChange={(event) => setTags(event.target.value)} placeholder="制度, 研发, 2026" /></div>
            </div>

            {uploadMutation.error && <div className="mt-5"><ErrorState message={uploadMutation.error.message} /></div>}
            <button type="submit" disabled={!files.length || !kbId || busy} className="btn-primary mt-6 w-full py-3">
              {uploadMutation.isPending ? <LoaderCircle className="h-4 w-4 animate-spin" /> : <UploadCloud className="h-4 w-4" />}
              {uploadMutation.isPending ? "正在上传并创建任务" : `上传 ${files.length || ""} 个文件并建立索引`}
            </button>
          </form>

          <form onSubmit={submitServerArchive} className="panel p-5 sm:p-6">
            <div className="flex items-center gap-2 font-semibold text-slate-900"><Server className="h-4 w-4 text-brand-600" />服务器压缩包导入</div>
            <p className="mt-2 text-sm leading-6 text-slate-500">
              将压缩包放到后端 <code className="rounded bg-slate-100 px-1">ZHIYUAN_IMPORT_ROOT</code> 目录，再填写相对路径或目录内绝对路径。路径越界会被拒绝。
            </p>
            <div className="mt-4 flex flex-col gap-3 sm:flex-row">
              <input className="input flex-1" value={serverPath} onChange={(event) => setServerPath(event.target.value)} placeholder="例如：2026/政策制度.zip" />
              <button type="submit" className="btn-secondary" disabled={!serverPath.trim() || !kbId || busy}>
                {serverArchiveMutation.isPending ? <LoaderCircle className="h-4 w-4 animate-spin" /> : <FileArchive className="h-4 w-4" />}
                导入服务器压缩包
              </button>
            </div>
            {serverArchiveMutation.error && <div className="mt-4"><ErrorState message={serverArchiveMutation.error.message} /></div>}
          </form>

          {!!results.length && (
            <div className="panel p-5">
              <div className="flex items-center justify-between">
                <h3 className="font-semibold text-slate-900">导入结果</h3>
                <div className="flex gap-2 text-xs">
                  <span className="chip bg-emerald-50 text-emerald-700">成功 {result?.accepted ?? 0}</span>
                  <span className="chip bg-red-50 text-red-700">失败 {result?.failed ?? 0}</span>
                </div>
              </div>
              <div className="mt-4 space-y-2">
                {results.map((item, index) => (
                  <div key={`${item.filename}-${index}`} className={`rounded-xl border p-3 text-sm ${item.ok ? "border-emerald-100 bg-emerald-50/60" : "border-red-100 bg-red-50/60"}`}>
                    <div className="flex items-center gap-2">
                      {item.ok ? <CheckCircle2 className="h-4 w-4 text-emerald-600" /> : <XCircle className="h-4 w-4 text-red-600" />}
                      <span className="truncate font-medium text-slate-800">{item.filename}</span>
                      {item.document && <Link to={`/documents/${item.document.id}`} className="ml-auto text-xs text-brand-700 hover:underline">查看预览</Link>}
                    </div>
                    {item.error && <div className="mt-1 text-xs text-red-700">{item.error}</div>}
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        <aside className="space-y-4">
          <div className="panel p-5">
            <h3 className="font-semibold text-slate-900">支持的导入方式</h3>
            <ul className="mt-3 space-y-3 text-sm leading-6 text-slate-500">
              <li>· 多文件页面批量上传，逐个创建异步入库任务。</li>
              <li>· ZIP/TAR 包自动安全解包，跳过非法路径与不支持格式。</li>
              <li>· 服务器目录压缩包导入，适合离线批量迁移。</li>
              <li>· 单文件流式落盘并流式上传 MinIO，支持 GB 级文件。</li>
              <li>· 可统一指定分块模式、大小、重叠与标签。</li>
            </ul>
          </div>
          <div className="rounded-2xl border border-amber-200 bg-amber-50 p-4 text-xs leading-6 text-amber-800">
            压缩包解压上限：{capabilities?.archive_max_mb ?? 4096} MB；单文件上限：{capabilities?.upload_max_mb ?? 2048} MB。Nginx 已同步提高请求体上限。
          </div>
        </aside>
      </div>
    </div>
  );
}
