# BGE Reranker v2 M3 本地部署方案

> 适用环境：Windows + Ollama 0.32.5+ + `dengcao/bge-reranker-v2-m3:latest`
>
> 已验证显卡：NVIDIA GeForce GTX 1060 6GB
>
> 已验证驱动：582.78 / CUDA 13.0

## 1. 问题现象

执行：

```powershell
ollama run dengcao/bge-reranker-v2-m3:latest
```

出现：

```text
Error: 500 Internal Server Error: llama-server process has terminated:
exit status 0xc0000409
```

Ollama 服务日志中的完整根因是：

```text
GGML_ASSERT(n_outputs_max <= cparams.n_outputs_max) failed
```

`0xc0000409` 只是 Windows 在进程触发断言崩溃后返回的通用错误码，不代表恶意程序或系统损坏。

## 2. 根因说明

`bge-reranker-v2-m3` 是 **Cross-Encoder 重排模型**，不是文本生成模型。

- 输入：查询 + 候选文档
- 输出：每个候选文档的相关性分数
- 模型架构：BERT
- 正确调用方式：重排接口 `/rerank`
- 错误调用方式：`ollama run`、`/api/generate`、`/api/chat`

`ollama run` 会将该模型按生成模型加载，并尝试进行文本生成 warmup，最终触发：

```text
n_outputs_max <= cparams.n_outputs_max
```

即输出数量与模型实际输出结构不匹配。

该问题在以下情况下均可复现：

- GPU 模式
- CPU 模式
- 较小 `num_ctx`
- 较小 `num_batch`

因此不是 NVIDIA 驱动、CUDA 或显存问题。

当前 Ollama 0.32.5 的标准 API 也没有提供：

```text
/api/rerank
/v1/rerank
```

## 3. 推荐方案

使用 Ollama 自带的 `llama-server.exe` 启动重排服务。

Ollama 安装目录中的 `llama-server.exe` 已支持：

```text
--rerank
--pooling {none,mean,cls,last,rank}
```

该方案使用同一份 GGUF 模型文件，不需要重新下载模型。

## 4. 前台启动

打开 PowerShell，执行：

```powershell
$lib = "$env:LOCALAPPDATA\Programs\Ollama\lib\ollama"

$blob = (
    ollama show dengcao/bge-reranker-v2-m3:latest --modelfile |
    Select-String '^FROM ' |
    Select-Object -Last 1
).Line.Substring(5).Trim()

$env:PATH = "$lib;$lib\cuda_v12;$env:PATH"

& "$lib\llama-server.exe" `
    --model $blob `
    --host 0.0.0.0 `
    --port 11436 `
    --rerank `
    --pooling rank `
    --ctx-size 8192 `
    --alias bge-reranker-v2-m3 `
    --no-webui `
    --offline
```

正常情况下会看到：

```text
model loaded
listening on http://0.0.0.0:11436
```

不要关闭该 PowerShell 窗口。

## 5. 测试重排接口

新开一个 PowerShell 窗口：

```powershell
$body = @{
    model = "bge-reranker-v2-m3"
    query = "测试查询"
    documents = @(
        "这是一段相关测试文本",
        "这是一段完全无关的内容"
    )
    top_n = 2
} | ConvertTo-Json -Depth 5

Invoke-RestMethod `
    -Uri "http://127.0.0.1:11436/rerank" `
    -Method Post `
    -ContentType "application/json" `
    -Body $body
```

预期响应：

```json
{
  "model": "bge-reranker-v2-m3",
  "object": "list",
  "usage": {
    "prompt_tokens": 25,
    "total_tokens": 25
  },
  "results": [
    {
      "index": 0,
      "relevance_score": -2.4809
    },
    {
      "index": 1,
      "relevance_score": -8.1039
    }
  ]
}
```

分数越高表示越相关。

`/v1/rerank` 也可以使用：

```text
POST http://127.0.0.1:11436/v1/rerank
```

## 6. 配置到知源项目

进入“模型配置”，新建或编辑 Rerank 模型。

### 后端运行在 Docker 中

| 配置项 | 值 |
| --- | --- |
| 类型 | Rerank |
| 显示名称 | BGE Reranker v2 M3 |
| Base URL | `http://host.docker.internal:11436` |
| 模型名称 | `bge-reranker-v2-m3` |
| API Key | 留空 |
| 高级配置 JSON | `{"rerank_path":"rerank"}` |

### 后端直接运行在 Windows 中

| 配置项 | 值 |
| --- | --- |
| Base URL | `http://127.0.0.1:11436` |
| 模型名称 | `bge-reranker-v2-m3` |
| 高级配置 JSON | `{"rerank_path":"rerank"}` |

保存后点击“测试”，应显示连接成功。

## 7. 后台启动

如果不想一直保留 PowerShell 窗口，可以使用以下方式启动隐藏进程。

```powershell
$lib = "$env:LOCALAPPDATA\Programs\Ollama\lib\ollama"

$blob = (
    ollama show dengcao/bge-reranker-v2-m3:latest --modelfile |
    Select-String '^FROM ' |
    Select-Object -Last 1
).Line.Substring(5).Trim()

$env:PATH = "$lib;$lib\cuda_v12;$env:PATH"

$logDir = "$env:LOCALAPPDATA\Ollama\logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$arguments = @(
    "--model", $blob,
    "--host", "0.0.0.0",
    "--port", "11436",
    "--rerank",
    "--pooling", "rank",
    "--ctx-size", "8192",
    "--alias", "bge-reranker-v2-m3",
    "--no-webui",
    "--offline"
)

$process = Start-Process `
    -FilePath "$lib\llama-server.exe" `
    -ArgumentList $arguments `
    -WindowStyle Hidden `
    -RedirectStandardOutput "$logDir\reranker.stdout.log" `
    -RedirectStandardError "$logDir\reranker.stderr.log" `
    -PassThru

$process.Id | Set-Content "$logDir\reranker.pid"
Write-Host "Reranker started. PID=$($process.Id)"
```

查看日志：

```powershell
Get-Content "$env:LOCALAPPDATA\Ollama\logs\reranker.stderr.log" -Tail 100
```

停止服务：

```powershell
$pidFile = "$env:LOCALAPPDATA\Ollama\logs\reranker.pid"
if (Test-Path $pidFile) {
    $rerankPid = [int](Get-Content $pidFile)
    Stop-Process -Id $rerankPid -Force -ErrorAction SilentlyContinue
    Remove-Item $pidFile -Force
}
```

## 8. Docker 访问与防火墙

如果知源后端运行在 Docker 中，必须使用：

```text
http://host.docker.internal:11436
```

不能使用容器内部的 `127.0.0.1:11436`。

如果仍然无法访问，可检查 Windows 防火墙。管理员 PowerShell 中可选执行：

```powershell
New-NetFirewallRule `
    -DisplayName "BGE Reranker 11436" `
    -Direction Inbound `
    -Protocol TCP `
    -LocalPort 11436 `
    -Action Allow
```

## 9. 验证端口

```powershell
Get-NetTCPConnection -LocalPort 11436 -State Listen
```

应看到监听记录。

如果监听地址只有 `127.0.0.1`，Docker 容器无法访问。启动时必须使用：

```text
--host 0.0.0.0
```

## 10. 常见误区

### 误区一：使用 `ollama run`

错误：

```powershell
ollama run dengcao/bge-reranker-v2-m3:latest
```

该命令适用于生成式 LLM，不适用于 Cross-Encoder Reranker。

### 误区二：使用 `/api/embed`

错误：

```text
POST /api/embed
```

Reranker 不是 Embedding 模型，`/api/embed` 返回的是向量，不是查询与文档的相关性分数。

### 误区三：把它配置成 LLM

错误：

```text
模型类型：LLM
```

该模型不能生成自然语言回答，只能对候选文档进行相关性排序。

### 误区四：Base URL 填成 `/v1`

不推荐：

```text
http://host.docker.internal:11436/v1
```

推荐：

```text
http://host.docker.internal:11436
```

然后在高级配置中设置：

```json
{"rerank_path":"rerank"}
```

## 11. 最终结论

- `ollama run` 无法运行该 Cross-Encoder 重排模型。
- 不是 CUDA、驱动或显存问题。
- 使用 Ollama 自带 `llama-server.exe --rerank` 可以正常运行。
- 已验证 `/rerank` 和 `/v1/rerank` 均可用。
- 知源项目应将该服务配置为 Rerank 模型，而不是 LLM 或 Embedding 模型。
