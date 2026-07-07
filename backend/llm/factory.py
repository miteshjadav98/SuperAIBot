"""LLM provider abstraction.

Agents call ``get_chat_model()`` instead of constructing a provider client
inline. This is the single place that knows how to build a chat model, so
switching providers (or models per-agent) never touches agent code.

Azure is the default and its SDK is imported eagerly (every agent uses it).
Other providers lazy-import so their optional SDKs aren't required to boot.
"""

from typing import Any

from core.settings import settings


def get_chat_model(provider: str | None = None, **overrides: Any):
    """Return a LangChain chat model for ``provider`` (default from settings).

    ``overrides`` are passed straight to the underlying constructor, e.g.
    ``get_chat_model(temperature=0)`` or ``get_chat_model("openai", model="gpt-4o")``.
    """
    provider = (provider or settings.default_llm_provider).lower()

    if provider == "azure":
        from langchain_openai import AzureChatOpenAI

        params = dict(
            azure_deployment=settings.azure_openai_deployment,
            azure_endpoint=settings.azure_openai_endpoint,
            api_key=settings.azure_openai_api_key,
            api_version=settings.azure_openai_api_version,
        )
        params.update(overrides)
        return AzureChatOpenAI(**params)

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        params = dict(model="gpt-4o", api_key=settings.openai_api_key)
        params.update(overrides)
        return ChatOpenAI(**params)

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic  # lazy: optional dep

        params = dict(model="claude-opus-4-8", api_key=settings.anthropic_api_key)
        params.update(overrides)
        return ChatAnthropic(**params)

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI  # lazy: optional dep

        params = dict(model="gemini-1.5-pro", google_api_key=settings.google_api_key)
        params.update(overrides)
        return ChatGoogleGenerativeAI(**params)

    if provider == "ollama":
        from langchain_ollama import ChatOllama  # lazy: optional dep

        params = dict(model="llama3.1", base_url=settings.ollama_base_url)
        params.update(overrides)
        return ChatOllama(**params)

    raise ValueError(
        f"Unknown LLM provider {provider!r}. "
        "Supported: azure, openai, anthropic, gemini, ollama."
    )


def get_embeddings(**overrides: Any):
    """Return the Azure embeddings model used by the RAG agent."""
    from langchain_openai import AzureOpenAIEmbeddings

    params = dict(
        azure_deployment=settings.azure_openai_embedding_deployment,
        azure_endpoint=settings.azure_openai_endpoint,
        api_key=settings.azure_openai_api_key,
        api_version=settings.azure_openai_api_version,
    )
    params.update(overrides)
    return AzureOpenAIEmbeddings(**params)
