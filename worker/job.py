from __future__ import annotations

import signal
import sys
from pathlib import Path

from pipeline import run_pipeline


INPUT_DIR = Path("/workspace/job/input")
# Everything the job prints also goes here, so the app can show it again
# after reconnecting.
JOB_LOG = Path("/workspace/job/job.log")


class _Tee:
    """Write to the app's SSH session and to the job log; keep going once the app is gone."""

    def __init__(self, stream, log):
        self.stream = stream
        self.log = log

    def write(self, text: str) -> int:
        self.log.write(text)
        if self.stream is not None:
            try:
                self.stream.write(text)
            except (OSError, ValueError):
                self.stream = None  # the app disconnected; the log carries on
        return len(text)

    def flush(self) -> None:
        self.log.flush()
        if self.stream is not None:
            try:
                self.stream.flush()
            except (OSError, ValueError):
                self.stream = None

    def isatty(self) -> bool:
        return False


def main() -> None:
    if not INPUT_DIR.is_dir():
        raise RuntimeError("Input folder was not uploaded.")
    # Closing the app ends its SSH session; the job must outlive it.
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    log = JOB_LOG.open("a", encoding="utf-8", buffering=1)
    sys.stdout = _Tee(sys.stdout, log)
    sys.stderr = _Tee(sys.stderr, log)
    print("[worker] starting local upload job", flush=True)
    run_pipeline()
    print("[worker] job completed", flush=True)


if __name__ == "__main__":
    main()
