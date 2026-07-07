"use client";

import { useEffect, useState } from "react";
import { useQueryState } from "nuqs";
import { GATEWAY_URL } from "@/providers/Auth";

export interface AgentOption {
  id: string;
  label: string;
  emoji: string;
}

/**
 * The Super Bot router entry is always shown first; the rest of the list is
 * discovered from the backend registry (GET /agents on the FastAPI gateway),
 * so adding a new agent to the platform needs no frontend change. The static
 * list below is only the fallback while loading / if the gateway is down.
 */
const SUPERBOT: AgentOption = { id: "superbot", label: "Super Bot (Auto)", emoji: "🤖" };

const FALLBACK_AGENTS: AgentOption[] = [
  SUPERBOT,
  { id: "personal_chef", label: "Personal Chef", emoji: "🍳" },
  { id: "email_agent", label: "Email Agent", emoji: "✉️" },
  { id: "wedding_planner", label: "Wedding Planner", emoji: "💍" },
  { id: "pdf_chatbot", label: "PDF Chatbot", emoji: "📄" },
  { id: "movie_recommender", label: "Movie Recommender", emoji: "🎬" },
];

export const DEFAULT_AGENT = SUPERBOT.id;

/**
 * Dropdown that switches which agent the chat talks to. It writes the
 * `assistantId` URL query param (read by StreamProvider) and clears `threadId`
 * so switching agents starts a fresh conversation.
 */
export function AgentSwitcher() {
  const [assistantId, setAssistantId] = useQueryState("assistantId");
  const [, setThreadId] = useQueryState("threadId");
  const [agents, setAgents] = useState<AgentOption[]>(FALLBACK_AGENTS);

  useEffect(() => {
    fetch(`${GATEWAY_URL}/agents`)
      .then((res) => (res.ok ? res.json() : Promise.reject(res.status)))
      .then((data: { agents: AgentOption[] }) => {
        if (Array.isArray(data.agents) && data.agents.length > 0) {
          setAgents([SUPERBOT, ...data.agents]);
        }
      })
      .catch(() => {
        // Gateway unreachable — keep the fallback list.
      });
  }, []);

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
        {agents.map((a) => (
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
