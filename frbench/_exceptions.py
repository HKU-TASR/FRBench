"""FRBench exception hierarchy."""


class FRBenchError(Exception):
    """Base class for FRBench errors."""


class FRBenchDownloadError(FRBenchError):
    """Raised when an asset cannot be downloaded or verified."""


class FRBenchAssetNotFoundError(FRBenchDownloadError):
    """Raised when a requested asset key is missing from the manifest."""


class FRBenchConfigError(FRBenchError):
    """Raised when model or detector configuration is invalid."""
