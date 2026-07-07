"use client";

import { Thread } from "@/components/thread";
import { StreamProvider } from "@/providers/Stream";
import { ThreadProvider } from "@/providers/Thread";
import { AuthProvider, useAuth } from "@/providers/Auth";
import { ArtifactProvider } from "@/components/thread/artifact";
import { AuthScreen } from "@/components/auth/auth-screen";
import { Toaster } from "@/components/ui/sonner";
import { LoaderCircle } from "lucide-react";
import React from "react";

function AppContent(): React.ReactNode {
  const { user, isLoading } = useAuth();

  if (isLoading) {
    return (
      <div className="flex h-screen w-full items-center justify-center">
        <LoaderCircle className="text-muted-foreground size-6 animate-spin" />
      </div>
    );
  }

  if (!user) {
    return <AuthScreen />;
  }

  return (
    <ThreadProvider>
      <StreamProvider>
        <ArtifactProvider>
          <Thread />
        </ArtifactProvider>
      </StreamProvider>
    </ThreadProvider>
  );
}

export default function HomePage(): React.ReactNode {
  return (
    <React.Suspense fallback={<div>Loading...</div>}>
      <Toaster />
      <AuthProvider>
        <AppContent />
      </AuthProvider>
    </React.Suspense>
  );
}
