import { v4 as uuidv4 } from "uuid";
import { ReactNode, useEffect, useRef } from "react";
import { cn } from "@/lib/utils";
import { useStreamContext } from "@/providers/Stream";
import { useState, FormEvent } from "react";
import { Button } from "../ui/button";
import { Checkpoint, Message } from "@langchain/langgraph-sdk";
import { AssistantMessage, AssistantMessageLoading } from "./messages/ai";
import { HumanMessage } from "./messages/human";
import {
  DO_NOT_RENDER_ID_PREFIX,
  ensureToolCallsHaveResponses,
} from "@/lib/ensure-tool-responses";
import { SuperBotLogoSVG } from "../icons/superbot";
import { TooltipIconButton } from "./tooltip-icon-button";
import {
  ArrowDown,
  LoaderCircle,
  PanelRightOpen,
  PanelRightClose,
  SquarePen,
  XIcon,
  Plus,
} from "lucide-react";
import { useQueryState, parseAsBoolean } from "nuqs";
import { StickToBottom, useStickToBottomContext } from "use-stick-to-bottom";
import ThreadHistory from "./history";
import { AgentSwitcher } from "@/components/agent-switcher";
import { ThemeToggle } from "@/components/theme-toggle";
import { toast } from "sonner";
import { useMediaQuery } from "@/hooks/useMediaQuery";
import { Label } from "../ui/label";
import { useFileUpload } from "@/hooks/use-file-upload";
import { ExecutionGraph } from "./execution-graph";
import { ContentBlocksPreview } from "./ContentBlocksPreview";
import { ingestPdfBlock, isPdfBlock } from "@/lib/multimodal-utils";
import { useAuth } from "@/providers/Auth";
import {
  useArtifactOpen,
  ArtifactContent,
  ArtifactTitle,
  useArtifactContext,
} from "./artifact";

function StickyToBottomContent(props: {
  content: ReactNode;
  footer?: ReactNode;
  className?: string;
  contentClassName?: string;
}) {
  const context = useStickToBottomContext();
  return (
    <div
      ref={context.scrollRef}
      style={{ width: "100%", height: "100%" }}
      className={props.className}
    >
      <div
        ref={context.contentRef}
        className={props.contentClassName}
      >
        {props.content}
      </div>

      {props.footer}
    </div>
  );
}

function ScrollToBottom(props: { className?: string }) {
  const { isAtBottom, scrollToBottom } = useStickToBottomContext();

  if (isAtBottom) return null;
  return (
    <Button
      variant="outline"
      className={props.className}
      onClick={() => scrollToBottom()}
    >
      <ArrowDown className="h-4 w-4" />
      <span>Scroll to bottom</span>
    </Button>
  );
}

export function Thread() {
  const [artifactContext, setArtifactContext] = useArtifactContext();
  const [artifactOpen, closeArtifact] = useArtifactOpen();

  const [threadId, _setThreadId] = useQueryState("threadId");
  const [assistantId] = useQueryState("assistantId");
  const { token } = useAuth();
  const [chatHistoryOpen, setChatHistoryOpen] = useQueryState(
    "chatHistoryOpen",
    parseAsBoolean.withDefault(false),
  );
  const [input, setInput] = useState("");
  const {
    contentBlocks,
    setContentBlocks,
    handleFileUpload,
    dropRef,
    removeBlock,
    resetBlocks: _resetBlocks,
    dragOver,
    handlePaste,
  } = useFileUpload();
  const [firstTokenReceived, setFirstTokenReceived] = useState(false);
  const isLargeScreen = useMediaQuery("(min-width: 1024px)");

  const stream = useStreamContext();
  const messages = stream.messages;
  const isLoading = stream.isLoading;

  const lastError = useRef<string | undefined>(undefined);

  const setThreadId = (id: string | null) => {
    _setThreadId(id);

    // close artifact and reset artifact context
    closeArtifact();
    setArtifactContext({});
  };

  useEffect(() => {
    if (!stream.error) {
      lastError.current = undefined;
      return;
    }
    try {
      const message = (stream.error as any).message;
      if (!message || lastError.current === message) {
        // Message has already been logged. do not modify ref, return early.
        return;
      }

      // Message is defined, and it has not been logged yet. Save it, and send the error
      lastError.current = message;
      toast.error("An error occurred. Please try again.", {
        description: (
          <p>
            <strong>Error:</strong> <code>{message}</code>
          </p>
        ),
        richColors: true,
        closeButton: true,
      });
    } catch {
      // no-op
    }
  }, [stream.error]);

  // TODO: this should be part of the useStream hook
  const prevMessageLength = useRef(0);
  useEffect(() => {
    if (
      messages.length !== prevMessageLength.current &&
      messages?.length &&
      messages[messages.length - 1].type === "ai"
    ) {
      setFirstTokenReceived(true);
    }

    prevMessageLength.current = messages.length;
  }, [messages]);

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    if ((input.trim().length === 0 && contentBlocks.length === 0) || isLoading)
      return;
    setFirstTokenReceived(false);

    const isPdfAgent = assistantId === "pdf_chatbot";
    const pdfBlocks = contentBlocks.filter(isPdfBlock);
    // Inline PDFs are NEVER sent to the model: Azure rasterizes a PDF to one
    // image per page and hard-caps a request at 50 images, so a long PDF used to
    // crash the run ("Too many images: 51"). The PDF chatbot answers from the
    // RAG index instead; other agents have no use for a PDF at all. Images/text
    // attachments still go inline (vision models use them).
    const inlineBlocks = contentBlocks.filter((b) => !isPdfBlock(b));

    if (pdfBlocks.length > 0 && !isPdfAgent) {
      toast.info(
        "PDFs are only used by the PDF Chatbot — switch to that agent to ask about a document. Sending your message without the PDF.",
      );
    }

    // Keep the message non-empty when a PDF is attached with no question typed.
    const pdfNote =
      isPdfAgent &&
      pdfBlocks.length > 0 &&
      input.trim().length === 0 &&
      inlineBlocks.length === 0
        ? `Uploaded ${pdfBlocks.length === 1 ? "a PDF" : `${pdfBlocks.length} PDFs`}. Please answer questions from it.`
        : "";
    const textToSend = input.trim().length > 0 ? input.trim() : pdfNote;

    if (textToSend.length === 0 && inlineBlocks.length === 0) {
      // Nothing left to send (e.g. only a PDF attached on a non-PDF agent).
      setContentBlocks([]);
      return;
    }

    const newHumanMessage: Message = {
      id: uuidv4(),
      type: "human",
      content: [
        ...(textToSend.length > 0 ? [{ type: "text", text: textToSend }] : []),
        ...inlineBlocks,
      ] as Message["content"],
    };

    // For the PDF chatbot, push any attached PDFs through the FastAPI /upload
    // endpoint so they're chunked + embedded into the Atlas vector store, making
    // them retrievable by the `ask_pdf_knowledge_base` RAG tool. We await
    // indexing before streaming the message so the tool can already find the
    // chunks on this first turn.
    if (isPdfAgent && pdfBlocks.length > 0) {
      const label = pdfBlocks.length === 1 ? "PDF" : `${pdfBlocks.length} PDFs`;
      const toastId = toast.loading(`Indexing ${label} for search…`);
      try {
        const results = await Promise.all(
          pdfBlocks.map((block) => ingestPdfBlock(block, token)),
        );
        const totalChunks = results.reduce((sum, r) => sum + r.chunks, 0);
        const imagesTotal = results.reduce((sum, r) => sum + r.imagesTotal, 0);
        const imagesDescribed = results.reduce(
          (sum, r) => sum + r.imagesDescribed,
          0,
        );
        const ocrUsed = results.some(
          (r) => r.textEngine === "document_intelligence",
        );
        const cap = results[0]?.imageCap ?? 0;

        let description = `${totalChunks} searchable chunk${totalChunks === 1 ? "" : "s"}${ocrUsed ? " · OCR" : ""}.`;
        if (imagesTotal > imagesDescribed) {
          description += ` Described the first ${imagesDescribed} of ${imagesTotal} images (cap ${cap}); all text was still indexed.`;
        } else if (imagesDescribed > 0) {
          description += ` Described ${imagesDescribed} image${imagesDescribed === 1 ? "" : "s"}.`;
        }
        toast.success(`Indexed ${label}.`, { id: toastId, description });
      } catch (err) {
        toast.error(`Could not index ${label}.`, {
          id: toastId,
          description: (
            <p>
              <code>{err instanceof Error ? err.message : String(err)}</code>
            </p>
          ),
        });
        // Indexing failed — abort (leaving the input + attachment in place)
        // so the user can retry rather than asking against a knowledge base
        // that doesn't have the document.
        return;
      }
    }

    const toolMessages = ensureToolCallsHaveResponses(stream.messages);

    const context =
      Object.keys(artifactContext).length > 0 ? artifactContext : undefined;

    stream.submit(
      { messages: [...toolMessages, newHumanMessage], context },
      {
        streamMode: ["values", "messages"],
        streamSubgraphs: true,
        streamResumable: true,
        optimisticValues: (prev) => ({
          ...prev,
          context,
          messages: [
            ...(prev.messages ?? []),
            ...toolMessages,
            newHumanMessage,
          ],
        }),
      },
    );

    setInput("");
    setContentBlocks([]);
  };

  const handleRegenerate = (
    parentCheckpoint: Checkpoint | null | undefined,
  ) => {
    // Do this so the loading state is correct
    prevMessageLength.current = prevMessageLength.current - 1;
    setFirstTokenReceived(false);
    stream.submit(undefined, {
      checkpoint: parentCheckpoint,
      streamMode: ["values", "messages"],
      streamSubgraphs: true,
      streamResumable: true,
    });
  };

  const chatStarted = !!threadId || !!messages.length;
  const hasNoAIOrToolMessages = !messages.find(
    (m) => m.type === "ai" || m.type === "tool",
  );

  return (
    <div className="flex h-screen w-full overflow-hidden">
      <div
        className="hidden h-full overflow-hidden border-r bg-background transition-[width] duration-300 ease-out lg:block"
        style={{ width: isLargeScreen && chatHistoryOpen ? 300 : 0 }}
      >
        <div
          className="h-full"
          style={{ width: 300 }}
        >
          <ThreadHistory />
        </div>
      </div>

      <div
        className={cn(
          "grid min-w-0 flex-1 grid-cols-[1fr_0fr] transition-all duration-500",
          artifactOpen && "grid-cols-[3fr_2fr]",
        )}
      >
        <div
          className={cn(
            "relative flex min-w-0 flex-1 flex-col overflow-hidden",
            !chatStarted && "grid-rows-[1fr]",
          )}
        >
          {!chatStarted && (
            <div className="absolute top-0 left-0 z-10 flex w-full items-center justify-between gap-3 p-2 pl-4">
              <div>
                {(!chatHistoryOpen || !isLargeScreen) && (
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
                )}
              </div>
              <div className="absolute top-2 right-4 flex items-center gap-4">
                <AgentSwitcher />
                <ThemeToggle />
              </div>
            </div>
          )}
          {chatStarted && (
            <div className="relative z-10 flex items-center justify-between gap-3 p-2">
              <div className="relative flex items-center justify-start gap-2">
                <div className="absolute left-0 z-10">
                  {(!chatHistoryOpen || !isLargeScreen) && (
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
                  )}
                </div>
                <button
                  className={cn(
                    "flex cursor-pointer items-center gap-2 transition-[margin] duration-300",
                    !chatHistoryOpen && "ml-12",
                  )}
                  onClick={() => setThreadId(null)}
                >
                  <SuperBotLogoSVG
                    width={32}
                    height={32}
                  />
                  <span className="hidden text-xl font-semibold tracking-tight sm:inline">
                    SuperBot
                  </span>
                </button>
              </div>

              <div className="flex items-center gap-2 sm:gap-4">
                <AgentSwitcher />
                <ThemeToggle />
                <TooltipIconButton
                  size="lg"
                  className="p-4"
                  tooltip="New thread"
                  variant="ghost"
                  onClick={() => setThreadId(null)}
                >
                  <SquarePen className="size-5" />
                </TooltipIconButton>
              </div>

              <div className="from-background to-background/0 absolute inset-x-0 top-full h-5 bg-gradient-to-b" />
            </div>
          )}

          <StickToBottom className="relative flex-1 overflow-hidden">
            <StickyToBottomContent
              className={cn(
                "absolute inset-0 overflow-y-scroll px-4 [&::-webkit-scrollbar]:w-1.5 [&::-webkit-scrollbar-thumb]:rounded-full [&::-webkit-scrollbar-thumb]:bg-muted [&::-webkit-scrollbar-track]:bg-transparent",
                !chatStarted && "mt-[25vh] flex flex-col items-stretch",
                chatStarted && "grid grid-rows-[1fr_auto]",
              )}
              contentClassName="pt-8 pb-16 max-w-3xl mx-auto flex flex-col gap-4 w-full"
              content={
                <>
                  {messages
                    .filter((m) => !m.id?.startsWith(DO_NOT_RENDER_ID_PREFIX))
                    .map((message, index) =>
                      message.type === "human" ? (
                        <HumanMessage
                          key={message.id || `${message.type}-${index}`}
                          message={message}
                          isLoading={isLoading}
                        />
                      ) : (
                        <AssistantMessage
                          key={message.id || `${message.type}-${index}`}
                          message={message}
                          isLoading={isLoading}
                          handleRegenerate={handleRegenerate}
                        />
                      ),
                    )}
                  {/* Special rendering case where there are no AI/tool messages, but there is an interrupt.
                    We need to render it outside of the messages list, since there are no messages to render */}
                  {hasNoAIOrToolMessages && !!stream.interrupt && (
                    <AssistantMessage
                      key="interrupt-msg"
                      message={undefined}
                      isLoading={isLoading}
                      handleRegenerate={handleRegenerate}
                    />
                  )}
                  {/* What the Super Bot actually did: which agents ran, what
                      was parallel, what failed. Replaces raw tool-call noise —
                      the full trace lives in LangSmith. */}
                  <ExecutionGraph
                    plan={stream.values?.plan}
                    results={stream.values?.results}
                    isLoading={isLoading}
                  />
                  {isLoading && !firstTokenReceived && (
                    <AssistantMessageLoading />
                  )}
                </>
              }
              footer={
                <div className="sticky bottom-0 flex flex-col items-center gap-8 bg-background">
                  {!chatStarted && (
                    <div className="flex items-center gap-3">
                      <SuperBotLogoSVG className="h-8 w-8 flex-shrink-0" />
                      <h1 className="text-2xl font-semibold tracking-tight">
                        SuperBot
                      </h1>
                    </div>
                  )}

                  <ScrollToBottom className="animate-in fade-in-0 zoom-in-95 absolute bottom-full left-1/2 mb-4 -translate-x-1/2" />

                  <div
                    ref={dropRef}
                    className={cn(
                      "bg-muted relative z-10 mx-auto mb-8 w-full max-w-3xl rounded-2xl shadow-xs transition-all",
                      dragOver
                        ? "border-primary border-2 border-dotted"
                        : "border border-solid",
                    )}
                  >
                    <form
                      onSubmit={handleSubmit}
                      className="mx-auto grid max-w-3xl grid-rows-[1fr_auto] gap-2"
                    >
                      <ContentBlocksPreview
                        blocks={contentBlocks}
                        onRemove={removeBlock}
                      />
                      <textarea
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        onPaste={handlePaste}
                        onKeyDown={(e) => {
                          if (
                            e.key === "Enter" &&
                            !e.shiftKey &&
                            !e.metaKey &&
                            !e.nativeEvent.isComposing
                          ) {
                            e.preventDefault();
                            const el = e.target as HTMLElement | undefined;
                            const form = el?.closest("form");
                            form?.requestSubmit();
                          }
                        }}
                        placeholder="Type your message..."
                        className="field-sizing-content resize-none border-none bg-transparent p-3.5 pb-0 shadow-none ring-0 outline-none focus:ring-0 focus:outline-none"
                      />

                      <div className="flex flex-wrap items-center gap-x-4 gap-y-2 p-2 pt-4 sm:gap-6">
                        <Label
                          htmlFor="file-input"
                          className="flex cursor-pointer items-center gap-2"
                        >
                          <Plus className="size-5 text-muted-foreground" />
                          <span className="hidden text-sm text-muted-foreground sm:inline">
                            Upload PDF or Image
                          </span>
                          <span className="text-sm text-muted-foreground sm:hidden">
                            Attach
                          </span>
                        </Label>
                        <input
                          id="file-input"
                          type="file"
                          onChange={handleFileUpload}
                          multiple
                          accept="image/jpeg,image/png,image/gif,image/webp,application/pdf"
                          className="hidden"
                        />
                        {stream.isLoading ? (
                          <Button
                            key="stop"
                            onClick={() => stream.stop()}
                            className="ml-auto"
                          >
                            <LoaderCircle className="h-4 w-4 animate-spin" />
                            Cancel
                          </Button>
                        ) : (
                          <Button
                            type="submit"
                            className="ml-auto shadow-md transition-all"
                            disabled={
                              isLoading ||
                              (!input.trim() && contentBlocks.length === 0)
                            }
                          >
                            Send
                          </Button>
                        )}
                      </div>
                    </form>
                  </div>
                </div>
              }
            />
          </StickToBottom>
        </div>
        <div className="relative flex flex-col border-l">
          <div className="absolute inset-0 flex min-w-[30vw] flex-col">
            <div className="grid grid-cols-[1fr_auto] border-b p-4">
              <ArtifactTitle className="truncate overflow-hidden" />
              <button
                onClick={closeArtifact}
                className="cursor-pointer"
              >
                <XIcon className="size-5" />
              </button>
            </div>
            <ArtifactContent className="relative flex-grow" />
          </div>
        </div>
      </div>
    </div>
  );
}
