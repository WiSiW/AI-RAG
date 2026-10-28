import { Link } from "react-router-dom";

export default function NotFoundPage() {
  return (
    <div className="grid min-h-screen place-items-center bg-slate-50 p-6">
      <div className="text-center">
        <div className="text-7xl font-bold text-slate-200">404</div>
        <h1 className="mt-4 text-xl font-semibold text-slate-900">页面不存在</h1>
        <p className="mt-2 text-sm text-slate-500">链接可能已失效或内容已被删除。</p>
        <Link to="/knowledge-bases" className="btn-primary mt-6">返回知识库</Link>
      </div>
    </div>
  );
}
