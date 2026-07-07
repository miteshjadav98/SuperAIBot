import { Client } from "@langchain/langgraph-sdk";

export function createClient(apiUrl: string, token: string | null) {
  return new Client({
    apiUrl,
    ...(token && {
      defaultHeaders: {
        Authorization: `Bearer ${token}`,
      },
    }),
  });
}
