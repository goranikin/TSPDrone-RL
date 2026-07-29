"""Backward-compatible entrypoint. Prefer:

    uv run python -m src.experiments.run
"""

from src.experiments.run import main

if __name__ == "__main__":
    main()
