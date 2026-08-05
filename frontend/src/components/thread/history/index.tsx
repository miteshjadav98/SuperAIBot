import { Button } from "@/components/ui/button";
import { useThreads, threadTitle } from "@/providers/Thread";
import { Thread } from "@langchain/langgraph-sdk";
import { useEffect } from "react";

import { getContentString } from "../utils";
import { useQueryState, parseAsBoolean } from "nuqs";
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet";
import { Skeleton } from "@/components/ui/skeleton";
import { PanelRightOpen, PanelRightClose, LogOut, SquarePen } from "lucide-react";
import { useMediaQuery } from "@/hooks/useMediaQuery";
import { useAuth } from "@/providers/Auth";
import { TooltipIconButton } from "@/components/thread/tooltip-icon-button";

function ThreadList({
  threads,
  onThreadClick,
}: {
  threads: Thread[];
  onThreadClick?: (threadId: string) => void;
}) {
  const [threadId, setThreadId] = useQueryState("threadId");

  return (
    <div className="flex h-full w-full flex-col items-start justify-start gap-2 overflow-y-scroll [&::-webkit-scrollbar]:w-1.5 [&::-webkit-scrollbar-thumb]:rounded-full [&::-webkit-scrollbar-thumb]:bg-muted [&::-webkit-scrollbar-track]:bg-transparent">
      {threads.map((t) => {
        // The generated title when there is one; the first message is only the
        // fallback, because on its own it labels every briefing thread "hello".
        let itemText = threadTitle(t) ?? t.thread_id;
        if (
          !threadTitle(t) &&
          typeof t.values === "object" &&
          t.values &&
          "messages" in t.values &&
          Array.isArray(t.values.messages) &&
          t.values.messages?.length > 0
        ) {
          const firstMessage = t.values.messages[0];
          itemText = getContentString(firstMessage.content);
        }
        return (
          <div
            key={t.thread_id}
            className="w-full px-1"
          >
            <Button
              variant="ghost"
              className="w-[280px] items-start justify-start text-left font-normal"
              onClick={(e) => {
                e.preventDefault();
                onThreadClick?.(t.thread_id);
                if (t.thread_id === threadId) return;
                setThreadId(t.thread_id);
              }}
            >
              <p className="truncate text-ellipsis">{itemText}</p>
            </Button>
          </div>
        );
      })}
    </div>
  );
}

/** Starts a fresh conversation. Clearing `threadId` is all a new chat is. */
function NewChatButton({ onClick }: { onClick?: () => void }) {
  const [, setThreadId] = useQueryState("threadId");

  return (
    <div className="w-full px-3">
      <Button
        variant="outline"
        className="w-full justify-start gap-2 font-normal"
        onClick={() => {
          setThreadId(null);
          onClick?.();
        }}
      >
        <SquarePen className="size-4" />
        New chat
      </Button>
    </div>
  );
}

function AccountFooter() {
  const { user, logout } = useAuth();
  if (!user) return null;
  return (
    <div className="flex w-full items-center justify-between gap-2 border-t px-4 py-3">
      <span
        className="text-muted-foreground truncate text-sm"
        title={user.email}
      >
        {user.email}
      </span>
      <TooltipIconButton
        tooltip="Sign out"
        variant="ghost"
        onClick={logout}
      >
        <LogOut className="size-4" />
      </TooltipIconButton>
    </div>
  );
}

function ThreadHistoryLoading() {
  return (
    <div className="flex h-full w-full flex-col items-start justify-start gap-2 overflow-y-scroll [&::-webkit-scrollbar]:w-1.5 [&::-webkit-scrollbar-thumb]:rounded-full [&::-webkit-scrollbar-thumb]:bg-muted [&::-webkit-scrollbar-track]:bg-transparent">
      {Array.from({ length: 30 }).map((_, i) => (
        <Skeleton
          key={`skeleton-${i}`}
          className="h-10 w-[280px]"
        />
      ))}
    </div>
  );
}

export default function ThreadHistory() {
  const isLargeScreen = useMediaQuery("(min-width: 1024px)");
  const [chatHistoryOpen, setChatHistoryOpen] = useQueryState(
    "chatHistoryOpen",
    parseAsBoolean.withDefault(false),
  );

  const { getThreads, threads, setThreads, threadsLoading, setThreadsLoading } =
    useThreads();

  useEffect(() => {
    if (typeof window === "undefined") return;
    setThreadsLoading(true);
    getThreads()
      .then(setThreads)
      .catch(console.error)
      .finally(() => setThreadsLoading(false));
  }, []);

  return (
    <>
      <div className="shadow-inner-right hidden h-screen w-[300px] shrink-0 flex-col items-start justify-start gap-6 border-r-[1px] border-slate-300 lg:flex">
        <div className="flex w-full items-center justify-between px-4 pt-1.5">
          <Button
            className="hover:bg-muted"
            variant="ghost"
            onClick={() => setChatHistoryOpen((p) => !p)}
          >
            {chatHistoryOpen ? (
              <PanelRightOpen className="size-5" />
            ) : (
              <PanelRightClose className="size-5" />
            )}
          </Button>
          <h1 className="text-xl font-semibold tracking-tight">
            Chat History
          </h1>
        </div>
        <div className="flex min-h-0 w-full flex-1 flex-col gap-3">
          <NewChatButton />
          {threadsLoading ? (
            <ThreadHistoryLoading />
          ) : (
            <ThreadList threads={threads} />
          )}
        </div>
        <AccountFooter />
      </div>
      <div className="lg:hidden">
        <Sheet
          open={!!chatHistoryOpen && !isLargeScreen}
          onOpenChange={(open) => {
            if (isLargeScreen) return;
            setChatHistoryOpen(open);
          }}
        >
          <SheetContent
            side="left"
            className="flex w-[280px] flex-col gap-0 p-0 lg:hidden"
          >
            <SheetHeader className="px-4 pt-4 pb-2">
              <SheetTitle>Chat History</SheetTitle>
            </SheetHeader>
            {/* min-h-0 lets the list scroll within the sheet so the footer
                (with logout) stays pinned and reachable on tall phones. */}
            <div className="flex min-h-0 w-full flex-1 flex-col gap-3 overflow-hidden px-2">
              <NewChatButton onClick={() => setChatHistoryOpen(false)} />
              <ThreadList
                threads={threads}
                onThreadClick={() => setChatHistoryOpen((o) => !o)}
              />
            </div>
            <AccountFooter />
          </SheetContent>
        </Sheet>
      </div>
    </>
  );
}
