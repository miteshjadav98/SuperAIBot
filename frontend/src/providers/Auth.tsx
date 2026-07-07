"use client";

import React, {
  createContext,
  useContext,
  useCallback,
  useEffect,
  useState,
  ReactNode,
} from "react";

export interface AuthUser {
  id: string;
  email: string;
  phone?: string | null;
}

interface AuthContextType {
  user: AuthUser | null;
  token: string | null;
  isLoading: boolean;
  login: (email: string, password: string) => Promise<void>;
  register: (email: string, password: string, phone?: string) => Promise<void>;
  logout: () => void;
}

const TOKEN_KEY = "superbot:token";
const USER_KEY = "superbot:user";

export const GATEWAY_URL =
  process.env.NEXT_PUBLIC_GATEWAY_URL || "http://localhost:8000";

const AuthContext = createContext<AuthContextType | undefined>(undefined);

async function authRequest(
  path: string,
  body: Record<string, unknown>,
): Promise<{ token: string; user: AuthUser }> {
  let res: Response;
  try {
    res = await fetch(`${GATEWAY_URL}${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
  } catch {
    throw new Error(
      "Could not reach the SuperBot server. Is the backend running?",
    );
  }
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const detail =
      typeof data.detail === "string"
        ? data.detail
        : "Something went wrong. Please try again.";
    throw new Error(detail);
  }
  return data;
}

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<AuthUser | null>(null);
  const [token, setToken] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  // Restore the session, then confirm the token is still valid.
  useEffect(() => {
    const storedToken = window.localStorage.getItem(TOKEN_KEY);
    const storedUser = window.localStorage.getItem(USER_KEY);
    if (!storedToken || !storedUser) {
      setIsLoading(false);
      return;
    }
    try {
      setUser(JSON.parse(storedUser));
      setToken(storedToken);
    } catch {
      window.localStorage.removeItem(TOKEN_KEY);
      window.localStorage.removeItem(USER_KEY);
      setIsLoading(false);
      return;
    }

    fetch(`${GATEWAY_URL}/auth/me`, {
      headers: { Authorization: `Bearer ${storedToken}` },
    })
      .then((res) => {
        if (res.status === 401) {
          window.localStorage.removeItem(TOKEN_KEY);
          window.localStorage.removeItem(USER_KEY);
          setUser(null);
          setToken(null);
        }
      })
      .catch(() => {
        // Gateway unreachable — keep the cached session; LangGraph server
        // still validates the token on every request.
      })
      .finally(() => setIsLoading(false));
  }, []);

  const persist = useCallback((nextToken: string, nextUser: AuthUser) => {
    window.localStorage.setItem(TOKEN_KEY, nextToken);
    window.localStorage.setItem(USER_KEY, JSON.stringify(nextUser));
    setToken(nextToken);
    setUser(nextUser);
  }, []);

  const login = useCallback(
    async (email: string, password: string) => {
      const data = await authRequest("/auth/login", { email, password });
      persist(data.token, data.user);
    },
    [persist],
  );

  const register = useCallback(
    async (email: string, password: string, phone?: string) => {
      const data = await authRequest("/auth/register", {
        email,
        password,
        phone: phone || null,
      });
      persist(data.token, data.user);
    },
    [persist],
  );

  const logout = useCallback(() => {
    window.localStorage.removeItem(TOKEN_KEY);
    window.localStorage.removeItem(USER_KEY);
    setToken(null);
    setUser(null);
  }, []);

  return (
    <AuthContext.Provider
      value={{ user, token, isLoading, login, register, logout }}
    >
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth(): AuthContextType {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error("useAuth must be used within an AuthProvider");
  }
  return context;
}

/** Read the stored token outside React (e.g. provider factories). */
export function getStoredToken(): string | null {
  if (typeof window === "undefined") return null;
  return window.localStorage.getItem(TOKEN_KEY);
}
