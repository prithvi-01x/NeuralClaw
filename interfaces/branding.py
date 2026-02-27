"""
interfaces/branding.py — Shared NeuralClaw Branding Assets

Constants used across CLI, onboard wizard, and the neuralclaw CLI tool.
Extracted to avoid duplication of the ASCII banner and colour-safe print wrapper.
"""

from __future__ import annotations

import re

NEURALCLAW_BANNER = """
███╗   ██╗███████╗██╗   ██╗██████╗  █████╗ ██╗      ██████╗██╗      █████╗ ██╗    ██╗
████╗  ██║██╔════╝██║   ██║██╔══██╗██╔══██╗██║     ██╔════╝██║     ██╔══██╗██║    ██║
██╔██╗ ██║█████╗  ██║   ██║██████╔╝███████║██║     ██║     ██║     ███████║██║ █╗ ██║
██║╚██╗██║██╔══╝  ██║   ██║██╔══██╗██╔══██║██║     ██║     ██║     ██╔══██║██║███╗██║
██║ ╚████║███████╗╚██████╔╝██║  ██║██║  ██║███████╗╚██████╗███████╗██║  ██║╚███╔███╔╝
╚═╝  ╚═══╝╚══════╝ ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝ ╚═════╝╚══════╝╚═╝  ╚═╝ ╚══╝╚══╝
"""


def safe_print(*args: object, **kwargs: object) -> None:
    """Print with Rich colour tags stripped — safe for terminals without Rich."""
    text = " ".join(str(x) for x in args)
    text = re.sub(r"\[/?[a-zA-Z_ ]+\]", "", text)
    print(text)
