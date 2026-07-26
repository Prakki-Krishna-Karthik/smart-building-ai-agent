"""Exception hierarchy for Ollama integration failures."""


class OllamaError(RuntimeError):
    """Base class for expected Ollama client errors."""


class OllamaConnectionError(OllamaError):
    """Raised when the Ollama service cannot be reached."""


class OllamaHTTPError(OllamaError):
    """Raised when Ollama returns an unsuccessful HTTP response."""


class OllamaModelError(OllamaError):
    """Raised when a requested model is unavailable or cannot be pulled."""


class OllamaResponseError(OllamaError):
    """Raised when Ollama returns malformed or schema-invalid data."""

