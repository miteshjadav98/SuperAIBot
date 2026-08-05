"use client";

import { useEffect, useState } from "react";
import { GATEWAY_URL } from "@/providers/Auth";

export interface AgentOption {
  id: string;
  label: string;
  emoji: string;
  capabilities?: string[];
}

/** The Super Bot entry is local — it is the orchestrator, not a registry agent. */
export const SUPERBOT: AgentOption = {
  id: "superbot",
  label: "Super Bot (Auto)",
  emoji: "🤖",
};

export const DEFAULT_AGENT = SUPERBOT.id;

/**
 * Only used while the gateway is loading or unreachable. The real list comes
 * from the backend registry, so adding an agent needs no frontend change.
 */
const FALLBACK_AGENTS: AgentOption[] = [
  SUPERBOT,
  { id: "personal_chef", label: "Personal Chef", emoji: "🍳" },
  { id: "email_agent", label: "Email Agent", emoji: "✉️" },
  { id: "wedding_planner", label: "Wedding Planner", emoji: "💍" },
  { id: "pdf_chatbot", label: "PDF Chatbot", emoji: "📄" },
  { id: "movie_recommender", label: "Movie Recommender", emoji: "🎬" },
];

/**
 * Agents discovered from `GET /agents`, shared by the switcher and the
 * execution graph so both name an agent the same way.
 */
export function useAgents() {
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

  const lookup = (id: string): AgentOption =>
    agents.find((a) => a.id === id) ?? {
      id,
      label: id.replace(/_/g, " "),
      emoji: "🔧",
    };

  return { agents, lookup };
}
