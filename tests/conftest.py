import os
import sys

from hypothesis import settings
from loguru import logger

# Pipecat logs every frame at DEBUG; keep test output readable.
logger.remove()
logger.add(sys.stderr, level="WARNING")

# CI replays a fixed set of property-test examples so a build can't flip on a lucky draw; local runs
# keep searching randomly (and a counterexample found locally becomes a fixed regression example).
settings.register_profile("ci", derandomize=True, database=None)
if os.environ.get("CI"):
    settings.load_profile("ci")
