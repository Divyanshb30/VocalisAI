import sys

from loguru import logger

# Pipecat logs every frame at DEBUG; keep test output readable.
logger.remove()
logger.add(sys.stderr, level="WARNING")
