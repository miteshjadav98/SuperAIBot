"use client";

import { FormEvent, useState } from "react";
import { useAuth } from "@/providers/Auth";
import { SuperBotLogoSVG } from "@/components/icons/superbot";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { PasswordInput } from "@/components/ui/password-input";
import { LoaderCircle } from "lucide-react";

export function AuthScreen() {
  const { login, register } = useAuth();
  const [mode, setMode] = useState<"login" | "register">("login");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [phone, setPhone] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  const isLogin = mode === "login";

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    setError(null);
    setSubmitting(true);
    try {
      if (isLogin) {
        await login(email, password);
      } else {
        await register(email, password, phone.trim() || undefined);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong.");
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="flex min-h-screen w-full items-center justify-center p-4">
      <div className="animate-in fade-in-0 zoom-in-95 bg-background flex w-full max-w-md flex-col rounded-lg border shadow-lg">
        <div className="flex flex-col items-center gap-3 border-b p-8 pb-6">
          <SuperBotLogoSVG
            width={48}
            height={48}
          />
          <h1 className="text-2xl font-semibold tracking-tight">SuperBot</h1>
          <p className="text-muted-foreground text-center text-sm">
            {isLogin
              ? "Welcome back! Sign in to continue your conversations."
              : "Create an account to start chatting with SuperBot."}
          </p>
        </div>

        <form
          onSubmit={handleSubmit}
          className="bg-muted/50 flex flex-col gap-4 rounded-b-lg p-8 pt-6"
        >
          <div className="flex flex-col gap-2">
            <Label htmlFor="email">Email</Label>
            <Input
              id="email"
              type="email"
              autoComplete="email"
              placeholder="you@example.com"
              className="bg-background"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
            />
          </div>

          <div className="flex flex-col gap-2">
            <Label htmlFor="password">Password</Label>
            <PasswordInput
              id="password"
              autoComplete={isLogin ? "current-password" : "new-password"}
              placeholder={isLogin ? "Your password" : "At least 8 characters"}
              className="bg-background"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
              minLength={isLogin ? undefined : 8}
            />
          </div>

          {!isLogin && (
            <div className="flex flex-col gap-2">
              <Label htmlFor="phone">
                Phone <span className="text-muted-foreground">(optional)</span>
              </Label>
              <Input
                id="phone"
                type="tel"
                autoComplete="tel"
                placeholder="+91 98765 43210"
                className="bg-background"
                value={phone}
                onChange={(e) => setPhone(e.target.value)}
              />
            </div>
          )}

          {error && (
            <p className="text-sm text-rose-500" role="alert">
              {error}
            </p>
          )}

          <Button
            type="submit"
            size="lg"
            className="mt-2"
            disabled={submitting}
          >
            {submitting && <LoaderCircle className="size-4 animate-spin" />}
            {isLogin ? "Sign in" : "Create account"}
          </Button>

          <p className="text-muted-foreground text-center text-sm">
            {isLogin ? "New to SuperBot?" : "Already have an account?"}{" "}
            <button
              type="button"
              className="text-foreground cursor-pointer font-medium underline underline-offset-4"
              onClick={() => {
                setError(null);
                setMode(isLogin ? "register" : "login");
              }}
            >
              {isLogin ? "Create an account" : "Sign in"}
            </button>
          </p>
        </form>
      </div>
    </div>
  );
}
