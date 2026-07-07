"use client";

import { useQueryState } from "nuqs";

/**
 * The agents exposed by the mega-app LangGraph server (see
 * backend/langgraph.json). Each `id` must match a graph id registered there.
 */
export const AGENTS = [
  { id: "superbot", label: "Super Bot (Auto)", emoji: "🤖" },
  { id: "personal_chef", label: "Personal Chef", emoji: "🍳" },
  { id: "email_agent", label: "Email Agent", emoji: "✉️" },
  { id: "wedding_planner", label: "Wedding Planner", emoji: "💍" },
  { id: "pdf_chatbot", label: "PDF Chatbot", emoji: "📄" },
  { id: "movie_recommender", label: "Movie Recommender", emoji: "🎬" },
] as const;

export const DEFAULT_AGENT = AGENTS[0].id;

/**
 * Dropdown that switches which graph the chat talks to. It writes the
 * `assistantId` URL query param (read by StreamProvider) and clears `threadId`
 * so switching agents starts a fresh conversation.
 */
export function AgentSwitcher() {
  const [assistantId, setAssistantId] = useQueryState("assistantId");
  const [, setThreadId] = useQueryState("threadId");

  const current = assistantId ?? DEFAULT_AGENT;

  return (
    <label className="flex items-center gap-2 text-sm">
      <span className="text-muted-foreground hidden sm:inline">Agent</span>
      <select
        value={current}
        onChange={(e) => {
          setThreadId(null);
          setAssistantId(e.target.value);
        }}
        className="bg-background hover:bg-muted focus:ring-ring cursor-pointer rounded-md border px-3 py-1.5 font-medium shadow-xs transition-colors focus:ring-2 focus:outline-none"
      >
        {AGENTS.map((a) => (
          <option
            key={a.id}
            value={a.id}
          >
            {a.emoji} {a.label}
          </option>
        ))}
      </select>
    </label>
  );
}
