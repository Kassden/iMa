"""Offline consumer checks before a paid planner or durable request is created."""
from importlib import import_module
from importlib.metadata import version
from packaging.version import Version


def require_planner_transport():
    """Validate the declared research transport in the interpreter actually used."""
    try:
        httpx = import_module("httpx")
        installed = Version(version("httpx"))
        if not Version("0.28") <= installed < Version("1"):
            raise ValueError(f"unsupported httpx version: {installed}")
        if not callable(getattr(httpx, "AsyncClient", None)):
            raise ValueError("httpx.AsyncClient is unavailable")
    except Exception as error:
        raise RuntimeError(
            "Planner runtime is incomplete: qualify the declared httpx>=0.28,<1 "
            "research dependency in this interpreter before starting paid planning"
        ) from error
    return {"module": "httpx", "version": str(installed),
            "module_path": getattr(httpx, "__file__", None), "network_request_sent": False}
