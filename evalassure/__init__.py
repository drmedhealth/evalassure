"""evalassure — open toolkit to test, evaluate, audit (compliance), troubleshoot and auto-fix
AI/ML models, LLM applications and AI automations."""
__version__ = "1.0.0"

from .core.types import Report, Severity  # noqa: E402


def run_scenario(path, autofix=None, progress=None):
    """Run a scenario YAML file and return a Report."""
    from .core.runner import run_scenario as _run

    return _run(path, autofix=autofix, progress=progress)


__all__ = ["run_scenario", "Report", "Severity", "__version__"]
