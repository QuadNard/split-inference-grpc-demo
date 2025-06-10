import logging
import os
import random
import signal
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import grpc

from service.proto import bitrate_pb2, bitrate_pb2_grpc

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("client_chain.log"),
    ],
)
logger = logging.getLogger(__name__)


@dataclass
class ClientConfig:
    """Configuration for the client chain."""

    prefill_host: str = "localhost"
    prefill_port: int = 50052
    decode_host: str = "localhost"
    decode_port: int = 50053
    min_throughput: float = 500.0
    max_throughput: float = 3000.0
    request_interval: float = 1.0
    timeout_seconds: float = 10.0
    max_retries: int = 3
    retry_delay: float = 1.0
    enable_metrics: bool = True


class MetricsCollector:
    """Collects and tracks client metrics."""

    def __init__(self):
        self.lock = threading.Lock()
        self.reset_metrics()

    def reset_metrics(self):
        """Reset all metrics."""
        with self.lock:
            self.total_requests = 0
            self.successful_requests = 0
            self.failed_requests = 0
            self.total_latency = 0.0
            self.min_latency = float("inf")
            self.max_latency = 0.0
            self.prefill_latencies = []
            self.decode_latencies = []
            self.total_latencies = []
            self.error_counts = {}

    def record_success(
        self,
        total_latency: float,
        prefill_latency: float = 0.0,
        decode_latency: float = 0.0,
    ):
        """Record a successful request."""
        with self.lock:
            self.total_requests += 1
            self.successful_requests += 1
            self.total_latency += total_latency
            self.min_latency = min(self.min_latency, total_latency)
            self.max_latency = max(self.max_latency, total_latency)
            self.total_latencies.append(total_latency)
            self.prefill_latencies.append(prefill_latency)
            self.decode_latencies.append(decode_latency)

            # Keep only last 1000 samples to prevent memory growth
            if len(self.total_latencies) > 1000:
                self.total_latencies.pop(0)
                self.prefill_latencies.pop(0)
                self.decode_latencies.pop(0)

    def record_failure(self, error_type: str):
        """Record a failed request."""
        with self.lock:
            self.total_requests += 1
            self.failed_requests += 1
            self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1

    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics."""
        with self.lock:
            if self.successful_requests == 0:
                return {
                    "total_requests": self.total_requests,
                    "successful_requests": 0,
                    "failed_requests": self.failed_requests,
                    "success_rate": 0.0,
                    "avg_latency": 0.0,
                    "min_latency": 0.0,
                    "max_latency": 0.0,
                    "error_counts": self.error_counts.copy(),
                }

            avg_latency = self.total_latency / self.successful_requests
            success_rate = (self.successful_requests / self.total_requests) * 100

            return {
                "total_requests": self.total_requests,
                "successful_requests": self.successful_requests,
                "failed_requests": self.failed_requests,
                "success_rate": success_rate,
                "avg_latency": avg_latency,
                "min_latency": self.min_latency
                if self.min_latency != float("inf")
                else 0.0,
                "max_latency": self.max_latency,
                "error_counts": self.error_counts.copy(),
            }


class BitrateClient:
    """Enhanced client for bitrate prediction chain."""

    def __init__(self, config: ClientConfig):
        self.config = config
        self.metrics = MetricsCollector()
        self.running = False
        self.prefill_channel: Optional[grpc.Channel] = None
        self.decode_channel: Optional[grpc.Channel] = None
        self.prefill_stub: Optional[bitrate_pb2_grpc.BitrateServiceStub] = None
        self.decode_stub: Optional[bitrate_pb2_grpc.BitrateServiceStub] = None

        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, shutting down...")
        self.stop()

    @contextmanager
    def _grpc_error_handler(self, operation: str):
        """Context manager for handling gRPC errors."""
        try:
            yield
        except grpc.RpcError as e:
            error_code = e.code()
            error_details = e.details()
            error_type = f"{operation}_{error_code.name}"

            logger.error(f"{operation} failed: {error_code.name} - {error_details}")
            self.metrics.record_failure(error_type)
            raise
        except Exception as e:
            error_type = f"{operation}_UNKNOWN"
            logger.error(f"{operation} failed with unexpected error: {e}")
            self.metrics.record_failure(error_type)
            raise

    def _create_channels(self):
        """Create gRPC channels with proper configuration."""
        # Channel options for reliability
        channel_options = [
            ("grpc.keepalive_time_ms", 30000),
            ("grpc.keepalive_timeout_ms", 5000),
            ("grpc.keepalive_permit_without_calls", True),
            ("grpc.http2.max_pings_without_data", 0),
            ("grpc.http2.min_time_between_pings_ms", 10000),
            ("grpc.http2.min_ping_interval_without_data_ms", 300000),
            ("grpc.max_receive_message_length", 4 * 1024 * 1024),
            ("grpc.max_send_message_length", 4 * 1024 * 1024),
        ]

        # Create channels
        prefill_addr = f"{self.config.prefill_host}:{self.config.prefill_port}"
        decode_addr = f"{self.config.decode_host}:{self.config.decode_port}"

        self.prefill_channel = grpc.insecure_channel(
            prefill_addr, options=channel_options
        )
        self.decode_channel = grpc.insecure_channel(
            decode_addr, options=channel_options
        )

        # Create stubs
        self.prefill_stub = bitrate_pb2_grpc.BitrateServiceStub(self.prefill_channel)
        self.decode_stub = bitrate_pb2_grpc.BitrateServiceStub(self.decode_channel)

        logger.info(f"Connected to prefill server: {prefill_addr}")
        logger.info(f"Connected to decode server: {decode_addr}")

    def _close_channels(self):
        """Close gRPC channels."""
        if self.prefill_channel:
            self.prefill_channel.close()
            self.prefill_channel = None

        if self.decode_channel:
            self.decode_channel.close()
            self.decode_channel = None

        self.prefill_stub = None
        self.decode_stub = None

    def _wait_for_ready(self):
        """Wait for servers to be ready."""
        logger.info("Waiting for servers to be ready...")

        try:
            # Wait for prefill server
            grpc.channel_ready_future(self.prefill_channel).result(
                timeout=self.config.timeout_seconds
            )
            logger.info("Prefill server ready")

            # Wait for decode server
            grpc.channel_ready_future(self.decode_channel).result(
                timeout=self.config.timeout_seconds
            )
            logger.info("Decode server ready")

        except grpc.FutureTimeoutError:
            raise ConnectionError("Timeout waiting for servers to be ready")

    def _generate_throughput(self) -> float:
        """Generate random throughput value."""
        return random.uniform(self.config.min_throughput, self.config.max_throughput)

    def _make_request_with_retry(self, throughput: float) -> Tuple[float, float, float]:
        """
        Make a request with retry logic.

        Returns:
            Tuple of (total_latency, prefill_latency, decode_latency)
        """
        last_exception = None

        for attempt in range(self.config.max_retries):
            try:
                return self._make_single_request(throughput)
            except Exception as e:
                last_exception = e
                if attempt < self.config.max_retries - 1:
                    logger.warning(
                        f"Request attempt {attempt + 1} failed: {e}, retrying..."
                    )
                    time.sleep(self.config.retry_delay)
                else:
                    logger.error(f"All {self.config.max_retries} attempts failed")

        raise last_exception

    def _make_single_request(self, throughput: float) -> Tuple[float, float, float]:
        """Make a single request through the chain."""
        total_start = time.perf_counter()

        # Prefill request
        prefill_start = time.perf_counter()
        with self._grpc_error_handler("PREFILL"):
            prefill_request = bitrate_pb2.Throughput(kbps=throughput)
            embedding = self.prefill_stub.Prefill(
                prefill_request, timeout=self.config.timeout_seconds
            )
        prefill_latency = (time.perf_counter() - prefill_start) * 1000

        # Decode request
        decode_start = time.perf_counter()
        with self._grpc_error_handler("DECODE"):
            result = self.decode_stub.Decode(
                embedding, timeout=self.config.timeout_seconds
            )
        decode_latency = (time.perf_counter() - decode_start) * 1000

        total_latency = (time.perf_counter() - total_start) * 1000

        return total_latency, prefill_latency, decode_latency

    def start(self):
        """Start the client chain."""
        logger.info("Starting bitrate client chain...")
        logger.info(f"Configuration: {self.config}")

        try:
            # Create connections
            self._create_channels()
            self._wait_for_ready()

            self.running = True
            request_count = 0

            logger.info("Starting request loop...")

            while self.running:
                try:
                    # Generate random throughput
                    throughput = self._generate_throughput()

                    # Make request with retry
                    total_latency, prefill_latency, decode_latency = (
                        self._make_request_with_retry(throughput)
                    )

                    # Record success
                    self.metrics.record_success(
                        total_latency, prefill_latency, decode_latency
                    )

                    # Log result
                    logger.info(
                        f"[{request_count:04d}] throughput={throughput:.0f}kbps → "
                        f"total={total_latency:.1f}ms "
                        f"(prefill={prefill_latency:.1f}ms, decode={decode_latency:.1f}ms)"
                    )

                    request_count += 1

                    # Print periodic stats
                    if self.config.enable_metrics and request_count % 10 == 0:
                        self._print_stats()

                    # Wait before next request
                    if self.running:
                        time.sleep(self.config.request_interval)

                except KeyboardInterrupt:
                    logger.info("Interrupted by user")
                    break
                except Exception as e:
                    logger.error(f"Request failed: {e}")
                    if self.running:
                        time.sleep(self.config.request_interval)

        finally:
            self.stop()

    def stop(self):
        """Stop the client chain."""
        if self.running:
            logger.info("Stopping client chain...")
            self.running = False
            self._print_final_stats()
            self._close_channels()
            logger.info("Client chain stopped")

    def _print_stats(self):
        """Print current statistics."""
        stats = self.metrics.get_stats()
        logger.info(
            f"STATS: {stats['successful_requests']}/{stats['total_requests']} "
            f"({stats['success_rate']:.1f}% success), "
            f"avg={stats['avg_latency']:.1f}ms, "
            f"min={stats['min_latency']:.1f}ms, "
            f"max={stats['max_latency']:.1f}ms"
        )

        if stats["error_counts"]:
            logger.info(f"ERRORS: {stats['error_counts']}")

    def _print_final_stats(self):
        """Print final statistics."""
        logger.info("=" * 50)
        logger.info("FINAL STATISTICS")
        logger.info("=" * 50)

        stats = self.metrics.get_stats()
        for key, value in stats.items():
            if key == "success_rate":
                logger.info(f"{key}: {value:.2f}%")
            elif "latency" in key:
                logger.info(f"{key}: {value:.2f}ms")
            else:
                logger.info(f"{key}: {value}")


def load_config_from_env() -> ClientConfig:
    """Load configuration from environment variables."""
    return ClientConfig(
        prefill_host=os.getenv("PREFILL_HOST", "localhost"),
        prefill_port=int(os.getenv("PREFILL_PORT", "50052")),
        decode_host=os.getenv("DECODE_HOST", "localhost"),
        decode_port=int(os.getenv("DECODE_PORT", "50053")),
        min_throughput=float(os.getenv("MIN_THROUGHPUT", "500.0")),
        max_throughput=float(os.getenv("MAX_THROUGHPUT", "3000.0")),
        request_interval=float(os.getenv("REQUEST_INTERVAL", "1.0")),
        timeout_seconds=float(os.getenv("TIMEOUT_SECONDS", "10.0")),
        max_retries=int(os.getenv("MAX_RETRIES", "3")),
        retry_delay=float(os.getenv("RETRY_DELAY", "1.0")),
        enable_metrics=os.getenv("ENABLE_METRICS", "true").lower() == "true",
    )


def main():
    """Main entry point."""
    logger.info("Starting Bitrate Client Chain...")

    try:
        config = load_config_from_env()
        client = BitrateClient(config)
        client.start()
    except Exception as e:
        logger.error(f"Client failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
