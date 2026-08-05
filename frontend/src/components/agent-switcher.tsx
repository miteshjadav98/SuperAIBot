"use client";

import { useQueryState } from "nuqs";
import { useAgents, DEFAULT_AGENT } from "@/hooks/useAgents";

export type { AgentOption } from "@/hooks/useAgents";
export { DEFAULT_AGENT } from "@/hooks/useAgents";

/**
 * Dropdown that switches which agent the chat talks to. It writes the
 * `assistantId` URL query param (read by StreamProvider) and clears `threadId`
 * so switching agents starts a fresh conversation. The agent list itself is
 * discovered from the backend registry (see `useAgents`).
 */
export function AgentSwitcher() {
  const [assistantId, setAssistantId] = useQueryState("assistantId");
  const [, setThreadId] = useQueryState("threadId");
  const { agents } = useAgents();

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
