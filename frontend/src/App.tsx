import { Navigate, Route, Routes } from "react-router-dom";
import { getToken } from "./api/client";
import AppLayout from "./components/AppLayout";
import ApiPublishingPage from "./pages/ApiPublishingPage";
import ChatPage from "./pages/ChatPage";
import ChunkEvaluationPage from "./pages/ChunkEvaluationPage";
import DocumentPreviewPage from "./pages/DocumentPreviewPage";
import HistoryPage from "./pages/HistoryPage";
import KnowledgeBaseDetailPage from "./pages/KnowledgeBaseDetailPage";
import KnowledgeBasesPage from "./pages/KnowledgeBasesPage";
import LoginPage from "./pages/LoginPage";
import ModelsPage from "./pages/ModelsPage";
import SearchQualityPage from "./pages/SearchQualityPage";
import NotFoundPage from "./pages/NotFoundPage";
import UploadPage from "./pages/UploadPage";

function RequireAuth() {
  return getToken() ? <AppLayout /> : <Navigate to="/login" replace />;
}

export default function App() {
  return (
    <Routes>
      <Route path="/login" element={<LoginPage />} />
      <Route element={<RequireAuth />}>
        <Route index element={<Navigate to="/knowledge-bases" replace />} />
        <Route path="/knowledge-bases" element={<KnowledgeBasesPage />} />
        <Route path="/knowledge-bases/:kbId" element={<KnowledgeBaseDetailPage />} />
        <Route path="/upload" element={<UploadPage />} />
        <Route path="/documents/:documentId" element={<DocumentPreviewPage />} />
        <Route path="/models" element={<ModelsPage />} />
        <Route path="/search-quality" element={<SearchQualityPage />} />
        <Route path="/chunk-evaluations" element={<ChunkEvaluationPage />} />
        <Route path="/api-publishing" element={<ApiPublishingPage />} />
        <Route path="/chat" element={<ChatPage />} />
        <Route path="/history" element={<HistoryPage />} />
      </Route>
      <Route path="*" element={<NotFoundPage />} />
    </Routes>
  );
}

