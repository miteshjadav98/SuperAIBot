"use client";

import { useMemo, useState } from "react";
import {
  AlertCircle,
  CheckCircle2,
  ChevronDown,
  ChevronRight,
  Circle,
  Loader2,
  RotateCw,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { useAgents } from "@/hooks/useAgents";

/** Mirrors `superbot/state.py`. */
export interface Task {
  id: string;
  agent_id: string;
  instruction: string;
  depends_on: string[];
  max_attempts: number;
}

export interface TaskResult {
  task_id: string;
  agent_id: string;
  ok: boolean;
  output: string;
}

type Status = "done" | "failed" | "running" | "retrying" | "pending";

/**
 * Group tasks into execution layers, the same way `executor._ready` does:
 * everything whose dependencies are already satisfied runs together. A layer
 * with more than one task is work that happened *in parallel*, which is the
 * whole point of showing this.
 */
function toLayers(plan: Task[]): Task[][] {
  const satisfied = new Set<string>();
  const layers: Task[][] = [];
  let remaining = [...plan];

  while (remaining.length > 0) {
    const ready = remaining.filter((t) =>
      (t.depends_on ?? []).every((d) => satisfied.has(d)),
    );
    // Defensive: a plan the backend would have rejected. Show the rest flat
    // rather than looping forever.
    if (ready.length === 0) {
      layers.push(remaining);
      break;
    }
    layers.push(ready);
    ready.forEach((t) => satisfied.add(t.id));
    remaining = remaining.filter((t) => !ready.includes(t));
  }
  return layers;
}

function statusOf(
  task: Task,
  results: TaskResult[],
  isLoading: boolean,
  succeeded: Set<string>,
): Status {
  const attempts = results.filter((r) => r.task_id === task.id);
  if (attempts.some((r) => r.ok)) return "done";
  if (attempts.length >= (task.max_attempts ?? 1)) return "failed";

  const unblocked = (task.depends_on ?? []).every((d) => succeeded.has(d));
  if (!unblocked) return "pending";
  if (!isLoading) return attempts.length > 0 ? "failed" : "pending";
  return attempts.length > 0 ? "retrying" : "running";
}

const STATUS_STYLES: Record<Status, { icon: React.ReactNode; ring: string }> = {
  done: {
    icon: <CheckCircle2 className="h-3.5 w-3.5 text-emerald-600 dark:text-emerald-400" />,
    ring: "border-emerald-500/40 bg-emerald-500/5",
  },
  failed: {
    icon: <AlertCircle className="h-3.5 w-3.5 text-red-600 dark:text-red-400" />,
    ring: "border-red-500/40 bg-red-500/5",
  },
  running: {
    icon: <Loader2 className="h-3.5 w-3.5 animate-spin text-blue-600 dark:text-blue-400" />,
    ring: "border-blue-500/50 bg-blue-500/5",
  },
  retrying: {
    icon: <RotateCw className="h-3.5 w-3.5 animate-spin text-amber-600 dark:text-amber-400" />,
    ring: "border-amber-500/50 bg-amber-500/5",
  },
  pending: {
    icon: <Circle className="h-3.5 w-3.5 text-muted-foreground/50" />,
    ring: "border-border bg-muted/40",
  },
};

/**
 * Renders the Super Bot's task DAG and its live progress.
 *
 * The backend already plans, fans out and merges; without this the most
 * interesting part of a run is invisible unless you read the logs. Layers are
 * stacked vertically, tasks within a layer sit side by side — so "these two ran
 * at the same time" is something you can see rather than something you have to
 * be told.
 */
export function ExecutionGraph({
  plan,
  results,
  isLoading,
}: {
  plan?: Task[] | null;
  results?: TaskResult[] | null;
  isLoading: boolean;
}) {
  const { lookup } = useAgents();
  const [expanded, setExpanded] = useState<string | null>(null);

  const taskResults = useMemo(() => results ?? [], [results]);
  const layers = useMemo(() => toLayers(plan ?? []), [plan]);
  const succeeded = useMemo(
    () => new Set(taskResults.filter((r) => r.ok).map((r) => r.task_id)),
    [taskResults],
  );

  // A one-task plan is just "the router picked an agent" — the DAG view adds
  // nothing over the answer itself.
  if (!plan || plan.length < 2) return null;

  const doneCount = succeeded.size;
  const parallelWidth = Math.max(...layers.map((l) => l.length));

  return (
    <div className="mx-auto w-full rounded-xl border bg-muted/30 p-3 text-sm">
      <div className="mb-3 flex items-center justify-between gap-2">
        <div className="flex items-center gap-2 font-medium">
          {isLoading ? (
            <Loader2 className="h-4 w-4 animate-spin text-blue-600 dark:text-blue-400" />
          ) : (
            <CheckCircle2 className="h-4 w-4 text-emerald-600 dark:text-emerald-400" />
          )}
          <span>Execution plan</span>
        </div>
        <span className="text-muted-foreground text-xs">
          {doneCount}/{plan.length} tasks
          {parallelWidth > 1 && ` · ${parallelWidth} in parallel`}
          {layers.length > 1 && ` · ${layers.length} steps`}
        </span>
      </div>

      <div className="flex flex-col gap-1">
        {layers.map((layer, layerIndex) => (
          <div key={layerIndex}>
            {layerIndex > 0 && (
              <div className="flex items-center gap-2 py-1 pl-3">
                <div className="bg-border h-4 w-px" />
                <span className="text-muted-foreground text-[10px] uppercase tracking-wide">
                  then
                </span>
              </div>
            )}

            <div className="flex flex-wrap gap-2">
              {layer.map((task) => {
                const status = statusOf(task, taskResults, isLoading, succeeded);
                const style = STATUS_STYLES[status];
                const agent = lookup(task.agent_id);
                const attempts = taskResults.filter((r) => r.task_id === task.id);
                const latest = attempts[attempts.length - 1];
                const isOpen = expanded === task.id;

                return (
                  <button
                    key={task.id}
                    type="button"
                    onClick={() => setExpanded(isOpen ? null : task.id)}
                    className={cn(
                      "flex min-w-[220px] flex-1 flex-col gap-1 rounded-lg border p-2 text-left transition-colors",
                      style.ring,
                    )}
                  >
                    <div className="flex items-center gap-1.5">
                      {style.icon}
                      <span aria-hidden>{agent.emoji}</span>
                      <span className="truncate font-medium">{agent.label}</span>
                      {attempts.length > 1 && (
                        <span className="text-amber-600 dark:text-amber-400 text-[10px]">
                          retry {attempts.length}/{task.max_attempts}
                        </span>
                      )}
                      {isOpen ? (
                        <ChevronDown className="text-muted-foreground ml-auto h-3 w-3 shrink-0" />
                      ) : (
                        <ChevronRight className="text-muted-foreground ml-auto h-3 w-3 shrink-0" />
                      )}
                    </div>

                    <p
                      className={cn(
                        "text-muted-foreground text-xs",
                        !isOpen && "line-clamp-2",
                      )}
                    >
                      {task.instruction}
                    </p>

                    {isOpen && latest && (
                      <p
                        className={cn(
                          "mt-1 max-h-40 overflow-y-auto whitespace-pre-wrap rounded border bg-background/60 p-1.5 text-xs",
                          !latest.ok && "text-red-600 dark:text-red-400",
                        )}
                      >
                        {latest.output}
                      </p>
                    )}
                  </button>
                );
              })}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
