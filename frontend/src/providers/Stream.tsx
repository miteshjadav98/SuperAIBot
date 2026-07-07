import React, {
  createContext,
  useContext,
  ReactNode,
  useEffect,
  useRef,
} from "react";
import { useStream } from "@langchain/langgraph-sdk/react";
import { type Message } from "@langchain/langgraph-sdk";
import {
  uiMessageReducer,
  isUIMessage,
  isRemoveUIMessage,
  type UIMessage,
  type RemoveUIMessage,
} from "@langchain/langgraph-sdk/react-ui";
import { useQueryState } from "nuqs";
import { useThreads } from "./Thread";
import { useAuth } from "./Auth";
import { toast } from "sonner";
import { DEFAULT_AGENT } from "@/components/agent-switcher";

export type StateType = { messages: Message[]; ui?: UIMessage[] };

const useTypedStream = useStream<
  StateType,
  {
    UpdateType: {
      messages?: Message[] | Message | string;
      ui?: (UIMessage | RemoveUIMessage)[] | UIMessage | RemoveUIMessage;
      context?: Record<string, unknown>;
    };
    CustomEventType: UIMessage | RemoveUIMessage;
  }
>;

type StreamContextType = ReturnType<typeof useTypedStream>;
const StreamContext = createContext<StreamContextType | undefined>(undefined);

async function checkServerStatus(
  apiUrl: string,
  token: string | null,
): Promise<boolean> {
  try {
    const res = await fetch(`${apiUrl}/info`, {
      headers: token ? { Authorization: `Bearer ${token}` } : undefined,
    });
    return res.ok;
  } catch (e) {
    console.error(e);
    return false;
  }
}

export const DEFAULT_API_URL = "http://localhost:2024";

const StreamSession = ({
  children,
  apiUrl,
  assistantId,
  token,
}: {
  children: ReactNode;
  apiUrl: string;
  assistantId: string;
  token: string | null;
}) => {
  const [threadId, setThreadId] = useQueryState("threadId");
  const { getThreads, setThreads } = useThreads();
  // Thread id of an in-flight run. Writing it to the URL immediately would
  // rerender with a changed `threadId` prop and abort the active stream (the
  // chat would go blank until a reload), so we hold it here and only sync the
  // URL once the stream settles.
  const pendingThreadIdRef = useRef<string | null>(null);
  const streamValue = useTypedStream({
    apiUrl,
    assistantId,
    defaultHeaders: token ? { Authorization: `Bearer ${token}` } : undefined,
    threadId: threadId ?? null,
    fetchStateHistory: true,
    onCustomEvent: (event, options) => {
      if (isUIMessage(event) || isRemoveUIMessage(event)) {
        options.mutate((prev) => {
          const ui = uiMessageReducer(prev.ui ?? [], event);
          return { ...prev, ui };
        });
      }
    },
    onThreadId: (id) => {
      pendingThreadIdRef.current = id;
    },
  });

  const isStreaming = streamValue.isLoading;
  useEffect(() => {
    if (isStreaming || !pendingThreadIdRef.current) return;
    const id = pendingThreadIdRef.current;
    pendingThreadIdRef.current = null;
    setThreadId(id);
    // The run is finished, so the new thread (with its messages) is now
    // searchable — refresh the Chat History list.
    getThreads().then(setThreads).catch(console.error);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isStreaming]);

  useEffect(() => {
    checkServerStatus(apiUrl, token).then((ok) => {
      if (!ok) {
        toast.error("Failed to connect to the SuperBot server", {
          description: (
            <p>
              Please make sure the backend is running at <code>{apiUrl}</code>{" "}
              and that you are signed in.
            </p>
          ),
          duration: 10000,
          richColors: true,
          closeButton: true,
        });
      }
    });
  }, [apiUrl, token]);

  return (
    <StreamContext.Provider value={streamValue}>
      {children}
    </StreamContext.Provider>
  );
};

export const StreamProvider: React.FC<{ children: ReactNode }> = ({
  children,
}) => {
  const { token } = useAuth();
  const apiUrl = process.env.NEXT_PUBLIC_API_URL || DEFAULT_API_URL;

  // The agent dropdown writes this query param; fall back to env, then superbot.
  const [assistantId] = useQueryState("assistantId");
  const finalAssistantId =
    assistantId || process.env.NEXT_PUBLIC_ASSISTANT_ID || DEFAULT_AGENT;

  return (
    <StreamSession
      key={finalAssistantId}
      apiUrl={apiUrl}
      assistantId={finalAssistantId}
      token={token}
    >
      {children}
    </StreamSession>
  );
};

// Create a custom hook to use the context
export const useStreamContext = (): StreamContextType => {
  const context = useContext(StreamContext);
  if (context === undefined) {
    throw new Error("useStreamContext must be used within a StreamProvider");
  }
  return context;
};

export default StreamContext;
