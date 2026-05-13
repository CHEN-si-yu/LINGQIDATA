import os
import sys
import time
import threading
from datetime import datetime as _dt
from collections import defaultdict

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
API_KEY_FILE = os.path.join(PROJECT_DIR, "APIKey.txt")
BASE_URL = "https://data.diemeng.chat/api"
DATA_DIR = os.path.join(PROJECT_DIR, "data")
LOG_DIR = os.path.join(DATA_DIR, "logs")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

_log_lock = threading.Lock()
_log_path = os.path.join(LOG_DIR, f"fetch_{_dt.now().strftime('%Y%m%d_%H%M%S')}.log")


def log_print(*args, **kwargs):
    """Print with ISO-8601 timestamp, also write to log file. Thread-safe."""
    ts = _dt.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] " + " ".join(str(a) for a in args)
    # Console
    print(line, **{k: v for k, v in kwargs.items() if k == 'file'})
    # File
    with _log_lock:
        with open(_log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def log_path():
    """Return the current session log file path."""
    return _log_path


def load_api_key():
    try:
        with open(API_KEY_FILE, "r", encoding="utf-8") as f:
            return f.read().strip().split("\t")[0]
    except FileNotFoundError:
        print(f"Error: APIKey.txt not found at {API_KEY_FILE}")
        sys.exit(1)


class RateLimiter:
    """Thread-safe token-bucket rate limiter for the entire account.

    Controls total request frequency across all endpoints to stay under
    ``max_rpm`` requests per minute.  Also tracks per-endpoint call counts
    so the controller can monitor hot endpoints.
    """

    def __init__(self, max_rpm=280):
        self.max_rpm = max_rpm
        self._refill_rate = max_rpm / 60.0          # tokens per second
        self._max_tokens = float(max_rpm)
        self._tokens = 0.0
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()
        self._endpoint_counts = defaultdict(int)    # per-endpoint call counter

    def _refill(self):
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(self._max_tokens, self._tokens + elapsed * self._refill_rate)
        self._last_refill = now

    def acquire(self, endpoint="unknown"):
        """Block until a token is available, then consume it.

        Returns the wait time in seconds (0 if no wait).
        """
        with self._lock:
            self._refill()
            if self._tokens >= 1.0:
                self._tokens -= 1.0
                self._endpoint_counts[endpoint] += 1
                return 0.0

            # How long until one token is available
            wait = (1.0 - self._tokens) / self._refill_rate
            self._tokens = 0.0

        time.sleep(wait)

        with self._lock:
            self._refill()
            self._tokens -= 1.0
            self._endpoint_counts[endpoint] += 1
        return wait

    @property
    def stats(self):
        """Return a snapshot of current usage."""
        with self._lock:
            self._refill()
            return {
                "tokens_available": round(self._tokens, 1),
                "max_rpm": self.max_rpm,
                "endpoint_counts": dict(self._endpoint_counts),
            }

    def reset_counts(self):
        with self._lock:
            self._endpoint_counts.clear()


# Global singleton — one rate limiter for the entire process
_limiter = RateLimiter(max_rpm=280)


def rate_limiter():
    return _limiter
