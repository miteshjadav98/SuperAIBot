"""Domain exceptions. The API layer maps these to HTTP status codes so the
service layer never imports FastAPI."""


class PromptServiceError(Exception):
    """Base class for all domain errors."""


class PromptNotFoundError(PromptServiceError):
    def __init__(self, prompt_id: str):
        self.prompt_id = prompt_id
        super().__init__(f"Prompt '{prompt_id}' does not exist")


class VersionNotFoundError(PromptServiceError):
    def __init__(self, prompt_id: str, version: int):
        self.prompt_id = prompt_id
        self.version = version
        super().__init__(f"Prompt '{prompt_id}' has no version {version}")


class PromptAlreadyExistsError(PromptServiceError):
    def __init__(self, prompt_id: str):
        self.prompt_id = prompt_id
        super().__init__(f"Prompt '{prompt_id}' already exists")


class VersionConflictError(PromptServiceError):
    """Raised when concurrent writers exhaust retries for a version number."""

    def __init__(self, prompt_id: str):
        self.prompt_id = prompt_id
        super().__init__(
            f"Could not allocate a new version for '{prompt_id}' due to "
            "concurrent updates; please retry"
        )
