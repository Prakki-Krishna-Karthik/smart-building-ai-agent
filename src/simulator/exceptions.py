"""Exception hierarchy for EnergyPlus integration failures."""


class EnergyPlusError(RuntimeError):
    """Base class for all expected EnergyPlus adapter errors."""


class EnergyPlusInstallationError(EnergyPlusError):
    """Raised when a usable EnergyPlus executable cannot be located or verified."""


class EnergyPlusInputError(EnergyPlusError):
    """Raised when simulation input paths are missing or invalid."""


class EnergyPlusSimulationError(EnergyPlusError):
    """Raised when an EnergyPlus process fails or cannot be started."""


class EnergyPlusSimulationTimeout(EnergyPlusSimulationError):
    """Raised when a simulation exceeds the configured execution timeout."""


class EnergyPlusOutputError(EnergyPlusError):
    """Raised when simulation output files cannot be collected."""

