import {
  Bot,
  Boxes,
  Database,
  History,
  Code2,
  Search,
  LogOut,
  Plus,
  Settings2,
  UploadCloud,
} from "lucide-react";
import { NavLink, Outlet, useLocation, useNavigate } from "react-router-dom";
import { clearToken } from "../api/client";

const navItems = [
  { to: "/knowledge-bases", label: "知识库", icon: Database },
  { to: "/upload", label: "上传文档", icon: UploadCloud },
  { to: "/chat", label: "智能问答", icon: Bot },
  { to: "/history", label: "历史问答", icon: History },
  { to: "/models", label: "模型配置", icon: Settings2 },
  { to: "/search-quality", label: "检索质检", icon: Search },
  { to: "/api-publishing", label: "API 发布", icon: Code2 },
];

const pageTitles: Record<string, string> = {
  "/knowledge-bases": "知识库",
  "/upload": "上传文档",
  "/chat": "智能问答",
  "/history": "历史问答",
  "/models": "模型配置",
  "/search-quality": "检索质检",
  "/api-publishing": "API 发布",
};

export default function AppLayout() {
  const navigate = useNavigate();
  const location = useLocation();
  const title =
    Object.entries(pageTitles).find(([path]) => location.pathname.startsWith(path))?.[1] ??
    "智库";

  function logout() {
    clearToken();
    navigate("/login", { replace: true });
  }

  return (
    <div className="flex min-h-screen bg-transparent">
      <aside className="fixed inset-y-0 left-0 z-30 hidden w-64 flex-col bg-ink-950 px-4 py-5 text-white lg:flex">
        <div className="mb-8 flex items-center gap-3 px-2">
          <div className="grid h-10 w-10 place-items-center rounded-xl bg-gradient-to-br from-brand-400 to-brand-700 shadow-lg shadow-brand-900/40">
            <Boxes className="h-5 w-5" />
          </div>
          <div>
            <div className="text-lg font-semibold tracking-wide">智库</div>
            <div className="text-xs text-slate-400">企业知识中枢</div>
          </div>
        </div>

        <NavLink
          to="/knowledge-bases"
          className="mb-5 flex items-center justify-center gap-2 rounded-xl bg-white/10 px-3 py-2.5 text-sm font-medium text-slate-200 transition hover:bg-white/15"
        >
          <Plus className="h-4 w-4" />
          新建知识库
        </NavLink>

        <nav className="space-y-1">
          {navItems.map((item) => {
            const Icon = item.icon;
            return (
              <NavLink
                key={item.to}
                to={item.to}
                className={({ isActive }) =>
                  `flex items-center gap-3 rounded-xl px-3 py-2.5 text-sm transition ${
                    isActive
                      ? "bg-brand-500/20 font-semibold text-brand-200"
                      : "text-slate-400 hover:bg-white/5 hover:text-white"
                  }`
                }
              >
                <Icon className="h-4 w-4" />
                {item.label}
              </NavLink>
            );
          })}
        </nav>

        <div className="mt-auto rounded-2xl border border-white/10 bg-white/5 p-3.5">
          <div className="text-xs text-slate-400">当前版本</div>
          <div className="mt-1 text-sm font-medium text-slate-200">单管理员 · API 模型</div>
          <button
            type="button"
            onClick={logout}
            className="mt-4 flex w-full items-center gap-2 rounded-lg px-2 py-2 text-sm text-slate-400 transition hover:bg-white/5 hover:text-white"
          >
            <LogOut className="h-4 w-4" />
            退出登录
          </button>
        </div>
      </aside>

      <div className="min-w-0 flex-1 lg:pl-64">
        <header className="sticky top-0 z-20 flex h-16 items-center justify-between border-b border-slate-200/80 bg-white/80 px-5 backdrop-blur-xl lg:px-8">
          <div>
            <div className="text-lg font-semibold text-slate-900">{title}</div>
            <div className="text-xs text-slate-500">基于混合检索与引用溯源的知识问答</div>
          </div>
          <div className="flex items-center gap-2">
            <div className="hidden items-center gap-2 rounded-full border border-emerald-200 bg-emerald-50 px-3 py-1.5 text-xs font-medium text-emerald-700 sm:flex">
              <span className="h-1.5 w-1.5 rounded-full bg-emerald-500" />
              API 模式
            </div>
            <button
              type="button"
              onClick={logout}
              className="grid h-9 w-9 place-items-center rounded-full bg-slate-100 text-slate-600 lg:hidden"
              aria-label="退出登录"
            >
              <LogOut className="h-4 w-4" />
            </button>
          </div>
        </header>

        <div className="mx-auto max-w-[1500px] p-4 sm:p-6 lg:p-8">
          <Outlet />
        </div>
      </div>

      <nav className="fixed inset-x-3 bottom-3 z-40 flex justify-around rounded-2xl border border-white/70 bg-ink-950/95 p-2 text-slate-400 shadow-2xl backdrop-blur lg:hidden">
        {navItems.map((item) => {
          const Icon = item.icon;
          return (
            <NavLink
              key={item.to}
              to={item.to}
              className={({ isActive }) =>
                `flex flex-1 flex-col items-center gap-1 rounded-xl py-2 text-[11px] ${
                  isActive ? "bg-white/10 text-brand-200" : ""
                }`
              }
            >
              <Icon className="h-4 w-4" />
              {item.label}
            </NavLink>
          );
        })}
      </nav>
    </div>
  );
}

