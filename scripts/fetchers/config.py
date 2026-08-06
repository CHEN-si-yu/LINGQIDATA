import os
import sys
import time
import threading
from datetime import datetime as _dt
from collections import defaultdict

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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
            for line in f:
                line = line.strip()
                if line:
                    return line
        print(f"Error: APIKey.txt is empty")
        sys.exit(1)
    except FileNotFoundError:
        print(f"Error: APIKey.txt not found at {API_KEY_FILE}")
        sys.exit(1)


def load_api_keys():
    """Return all non-empty API keys from APIKey.txt as a list."""
    try:
        with open(API_KEY_FILE, "r", encoding="utf-8") as f:
            keys = [line.strip() for line in f if line.strip()]
        if not keys:
            print(f"Error: APIKey.txt is empty")
            sys.exit(1)
        return keys
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
        # 从 0 起步而不是一次性注满 max_rpm: 初始 280 token 会让多 worker
        # 在启动瞬间爆发请求, 触发服务器 429 (请求过于频繁)。
        # 匀速 280/min 放行后, 任意窗口内的请求数都被平滑约束。
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

        Returns the total wait time in seconds (0 if no wait).

        注意: 睡眠醒来后必须重新检查 token —— 多个线程会同时醒来,
        若醒来后直接扣减, 所有线程会一次通过(整批爆发), 导致请求按
        批次突刺触发服务器 429 (2026-08-01 全量重拉踩坑的根因)。
        """
        wait_total = 0.0
        while True:
            with self._lock:
                self._refill()
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    self._endpoint_counts[endpoint] += 1
                    return wait_total

                # How long until one token is available.
                # 注意: 不能把 _tokens 清零 —— 部分 token 要保留累积,
                # 否则多线程争抢时会互相重置 _last_refill, 累积的 token
                # 永远凑不满 1.0, 线程陷入小睡空转 (实测 73 次迭代才拿到)。
                wait = (1.0 - self._tokens) / self._refill_rate
                wait_total += wait

            time.sleep(wait)

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
