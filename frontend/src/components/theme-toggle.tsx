"use client";

import { useEffect, useState } from "react";
import { useTheme } from "next-themes";
import { Moon, Sun } from "lucide-react";
import { TooltipIconButton } from "@/components/thread/tooltip-icon-button";

/**
 * Light/dark toggle. Flips between the two resolved themes. Renders a stable
 * placeholder until mounted so server and client markup match (next-themes
 * can't know the theme during SSR).
 */
export function ThemeToggle() {
  const { resolvedTheme, setTheme } = useTheme();
  const [mounted, setMounted] = useState(false);

  useEffect(() => setMounted(true), []);

  const isDark = resolvedTheme === "dark";

  return (
    <TooltipIconButton
      tooltip={isDark ? "Light mode" : "Dark mode"}
      variant="ghost"
      onClick={() => setTheme(isDark ? "light" : "dark")}
      aria-label="Toggle theme"
    >
      {!mounted ? (
        <Sun className="size-5" />
      ) : isDark ? (
        <Moon className="size-5" />
      ) : (
        <Sun className="size-5" />
      )}
    </TooltipIconButton>
  );
}
