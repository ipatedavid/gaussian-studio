from __future__ import annotations

import queue
import signal
import sys
import threading
import traceback
from pathlib import Path

from pipeline import run_pipeline


INPUT_DIR = Path("/workspace/job/input")
# Everything the job prints also goes here, so the app can show it again
# after reconnecting.
JOB_LOG = Path("/workspace/job/job.log")


class _Tee:
    """Write to the job log and to the app's SSH session; keep going once the app is gone.

    The session is written from a background thread through a queue. A session the
    app left without closing (its computer slept, its network dropped) stops taking
    output without failing, and a direct write would then hang the whole job. When
    the queue fills up, the session is dropped; the log carries on, and the app
    reads the log when it reconnects.
    """

    QUEUE_SIZE = 5000

    def __init__(self, stream, log):
        self.stream = stream
        self.log = log
        self._queue: queue.Queue[str | None] = queue.Queue(self.QUEUE_SIZE)
        self._writer = threading.Thread(target=self._send, daemon=True)
        self._writer.start()

    def _send(self) -> None:
        while (text := self._queue.get()) is not None:
            stream = self.stream
            if stream is None:
                continue
            try:
                stream.write(text)
                stream.flush()
            except (OSError, ValueError):
                self.stream = None  # the app disconnected; the log carries on

    def write(self, text: str) -> int:
        self.log.write(text)
        if self.stream is not None:
            try:
                self._queue.put_nowait(text)
            except queue.Full:
                self.stream = None  # the session stopped taking output
        return len(text)

    def flush(self) -> None:
        self.log.flush()

    def close(self, timeout: float = 30) -> None:
        """Let the app see the last lines, unless the session is stuck."""
        try:
            self._queue.put(None, timeout=timeout)
        except queue.Full:
            return
        self._writer.join(timeout)

    def isatty(self) -> bool:
        return False


def main() -> None:
    if not INPUT_DIR.is_dir():
        raise RuntimeError("Input folder was not uploaded.")
    # Closing the app ends its SSH session; the job must outlive it.
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    log = JOB_LOG.open("a", encoding="utf-8", buffering=1)
    sys.stdout = out = _Tee(sys.stdout, log)
    sys.stderr = err = _Tee(sys.stderr, log)
    try:
        print("[worker] starting local upload job", flush=True)
        run_pipeline()
        print("[worker] job completed", flush=True)
    except BaseException:
        # Print the error while the app's session is still being fed: Python's own
        # report would come after close() and reach only the log on the pod.
        traceback.print_exc()
        raise SystemExit(1)
    finally:
        out.close()
        err.close()


if __name__ == "__main__":
    main()
