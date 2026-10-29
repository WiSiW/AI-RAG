import { FormEvent, useEffect, useState } from "react";
import { Boxes, LoaderCircle, LockKeyhole, UserRound } from "lucide-react";
import { useNavigate } from "react-router-dom";
import { getToken, login } from "../api/client";

export default function LoginPage() {
  const navigate = useNavigate();
  const [username, setUsername] = useState("admin");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (getToken()) navigate("/knowledge-bases", { replace: true });
  }, [navigate]);

  async function submit(event: FormEvent) {
    event.preventDefault();
    setLoading(true);
    setError("");
    try {
      await login(username.trim(), password);
      navigate("/knowledge-bases", { replace: true });
    } catch (err) {
      setError(err instanceof Error ? err.message : "登录失败");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="grid min-h-screen bg-ink-950 lg:grid-cols-[1.05fr_0.95fr]">
      <div className="relative hidden overflow-hidden p-12 lg:flex lg:flex-col lg:justify-between">
        <div className="absolute -left-28 top-1/4 h-96 w-96 rounded-full bg-brand-500/20 blur-3xl" />
        <div className="absolute -right-20 bottom-0 h-80 w-80 rounded-full bg-cyan-400/10 blur-3xl" />
        <div className="relative flex items-center gap-3 text-white">
          <div className="grid h-11 w-11 place-items-center rounded-2xl bg-gradient-to-br from-brand-400 to-brand-700 shadow-xl shadow-brand-900/50">
            <Boxes className="h-6 w-6" />
          </div>
          <div>
            <div className="text-xl font-semibold tracking-wide">智库</div>
            <div className="text-xs text-slate-400">企业知识中枢</div>
          </div>
        </div>
        <div className="relative max-w-xl">
          <p className="mb-4 text-sm font-medium uppercase tracking-[0.24em] text-brand-300">Knowledge Intelligence</p>
          <h1 className="text-4xl font-semibold leading-tight text-white xl:text-5xl">
            把分散文档，
            <br />
            变成可追溯的知识。
          </h1>
          <p className="mt-6 max-w-lg text-base leading-8 text-slate-400">
            文档解析、混合检索、语义重排与流式问答一体化。每个答案都能回到具体文档与章节。
          </p>
          <div className="mt-9 grid grid-cols-3 gap-3 text-center">
            {[
              ["4", "文档格式"],
              ["Hybrid", "混合检索"],
              ["API", "模型接入"],
            ].map(([value, label]) => (
              <div key={label} className="rounded-2xl border border-white/10 bg-white/5 p-4">
                <div className="text-lg font-semibold text-white">{value}</div>
                <div className="mt-1 text-xs text-slate-400">{label}</div>
              </div>
            ))}
          </div>
        </div>
        <div className="relative text-xs text-slate-500">单管理员版 · Docker Compose 单机部署</div>
      </div>

      <div className="flex items-center justify-center bg-slate-50 p-5 sm:p-10">
        <div className="w-full max-w-md">
          <div className="mb-8 flex items-center gap-3 lg:hidden">
            <div className="grid h-10 w-10 place-items-center rounded-xl bg-ink-950 text-white">
              <Boxes className="h-5 w-5" />
            </div>
            <div className="text-lg font-semibold text-slate-900">智库</div>
          </div>
          <h2 className="text-2xl font-semibold text-slate-950">管理员登录</h2>
          <p className="mt-2 text-sm text-slate-500">使用环境变量中配置的管理员账号登录。</p>

          <form onSubmit={submit} className="mt-8 space-y-5">
            <div>
              <label className="label" htmlFor="username">用户名</label>
              <div className="relative">
                <UserRound className="pointer-events-none absolute left-3.5 top-3 h-4 w-4 text-slate-400" />
                <input
                  id="username"
                  className="input pl-10"
                  value={username}
                  onChange={(event) => setUsername(event.target.value)}
                  autoComplete="username"
                  required
                />
              </div>
            </div>
            <div>
              <label className="label" htmlFor="password">密码</label>
              <div className="relative">
                <LockKeyhole className="pointer-events-none absolute left-3.5 top-3 h-4 w-4 text-slate-400" />
                <input
                  id="password"
                  className="input pl-10"
                  type="password"
                  value={password}
                  onChange={(event) => setPassword(event.target.value)}
                  autoComplete="current-password"
                  required
                />
              </div>
            </div>
            {error && (
              <div className="rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
                {error}
              </div>
            )}
            <button type="submit" disabled={loading} className="btn-primary w-full py-3">
              {loading && <LoaderCircle className="h-4 w-4 animate-spin" />}
              {loading ? "正在登录" : "进入知识中枢"}
            </button>
          </form>
          <p className="mt-6 text-center text-xs leading-5 text-slate-400">
            首次部署请复制 .env.example 为 .env，并修改管理员密码与 SECRET_KEY。
          </p>
        </div>
      </div>
    </div>
  );
}
