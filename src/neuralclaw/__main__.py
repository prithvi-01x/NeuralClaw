"""Allow `python -m neuralclaw` to launch the agent."""

import asyncio
import sys

from neuralclaw.main import main

sys.exit(asyncio.run(main()))
