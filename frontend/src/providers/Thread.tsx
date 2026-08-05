import { validate } from "uuid";
import { Thread } from "@langchain/langgraph-sdk";
import { useQueryState } from "nuqs";
import {
  createContext,
  useContext,
  ReactNode,
  useCallback,
  useState,
  Dispatch,
  SetStateAction,
} from "react";
import { createClient } from "./client";
import { useAuth, GATEWAY_URL } from "./Auth";
import { DEFAULT_AGENT } from "@/components/agent-switcher";
import { DEFAULT_API_URL } from "./Stream";

interface ThreadContextType {
  getThreads: () => Promise<Thread[]>;
  threads: Thread[];
  setThreads: Dispatch<SetStateAction<Thread[]>>;
  threadsLoading: boolean;
  setThreadsLoading: Dispatch<SetStateAction<boolean>>;
  ensureTitle: (
    threadId: string,
    messages: { role: string; content: string }[],
  ) => Promise<void>;
}

/** A conversation's display name, stored on the thread by `ensureTitle`. */
export function threadTitle(thread: Thread): string | undefined {
  const title = (thread.metadata as { title?: unknown } | undefined)?.title;
  return typeof title === "string" && title.trim() ? title : undefined;
}

const ThreadContext = createContext<ThreadContextType | undefined>(undefined);

function getThreadSearchMetadata(
  assistantId: string,
): { graph_id: string } | { assistant_id: string } {
  if (validate(assistantId)) {
    return { assistant_id: assistantId };
  } else {
    return { graph_id: assistantId };
  }
}

export function ThreadProvider({ children }: { children: ReactNode }) {
  const { token } = useAuth();
  const apiUrl = process.env.NEXT_PUBLIC_API_URL || DEFAULT_API_URL;
  const [assistantId] = useQueryState("assistantId");
  const [threads, setThreads] = useState<Thread[]>([]);
  const [threadsLoading, setThreadsLoading] = useState(false);

  const getThreads = useCallback(async (): Promise<Thread[]> => {
    const resolvedAssistantId =
      assistantId || process.env.NEXT_PUBLIC_ASSISTANT_ID || DEFAULT_AGENT;
    if (!apiUrl || !token) return [];
    const client = createClient(apiUrl, token);

    const threads = await client.threads.search({
      metadata: {
        ...getThreadSearchMetadata(resolvedAssistantId),
      },
      limit: 100,
    });

    return threads;
  }, [apiUrl, assistantId, token]);

  /**
   * Give a thread a real title, once. The chat list used to show the first
   * message, so every conversation that opened with "hello" was called "hello".
   *
   * Titles live in thread metadata rather than being derived at render time:
   * one LLM call per conversation instead of one per list paint, and the name
   * stays stable as the conversation grows.
   */
  const ensureTitle = useCallback(
    async (threadId: string, messages: { role: string; content: string }[]) => {
      if (!apiUrl || !token || !threadId || messages.length === 0) return;
      const client = createClient(apiUrl, token);

      const existing = await client.threads.get(threadId).catch(() => null);
      if (!existing || threadTitle(existing)) return;

      const response = await fetch(`${GATEWAY_URL}/threads/title`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({ messages }),
      });
      if (!response.ok) return;

      const { title } = (await response.json()) as { title?: string };
      if (!title) return;

      // Merge, never replace: graph_id/assistant_id live here too and the
      // thread list is searched by them.
      await client.threads.update(threadId, {
        metadata: { ...(existing.metadata ?? {}), title },
      });
    },
    [apiUrl, token],
  );

  const value = {
    getThreads,
    threads,
    setThreads,
    threadsLoading,
    setThreadsLoading,
    ensureTitle,
  };

  return (
    <ThreadContext.Provider value={value}>{children}</ThreadContext.Provider>
  );
}

export function useThreads() {
  const context = useContext(ThreadContext);
  if (context === undefined) {
    throw new Error("useThreads must be used within a ThreadProvider");
  }
  return context;
}
