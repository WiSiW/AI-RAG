import { PropsWithChildren, useEffect } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { chatRuntimeStore } from "./runtimeStore";

export default function ChatRuntimeProvider({ children }: PropsWithChildren) {
  const queryClient = useQueryClient();

  useEffect(() => {
    chatRuntimeStore.setTerminalHandler((conversationId) => {
      if (!conversationId) return;
      void queryClient.invalidateQueries({ queryKey: ["conversations"] });
      void queryClient.invalidateQueries({ queryKey: ["conversation", conversationId] });
    });
    return () => chatRuntimeStore.setTerminalHandler(null);
  }, [queryClient]);

  return children;
}
