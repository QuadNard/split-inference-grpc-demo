import logging
import os
import signal
import sys
import time
from concurrent import futures
from typing import Optional

import grpc
import torch
from grpc_reflection.v1alpha import reflection
from model import BitrateLSTM

from service.proto import bitrate_pb2, bitrate_pb2_grpc

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("prefill_server.log"),
    ],
)
logger = logging.getLogger(__name__)


class PrefillServicer(bitrate_pb2_grpc.BitrateServiceServicer):
    def __init__(self):
        """Initialize the servicer with the LSTM model."""
        try:
            self.model = BitrateLSTM()
            self.model.eval()
            logger.info("LSTM model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load LSTM model: {e}")
            raise

    def Prefill(self, request, context):
        start_time = time.perf_counter()

        try:
            # Validate input
            if not hasattr(request, "kbps"):
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details("Missing 'kbps' field in request")
                return bitrate_pb2.Embedding()

            if request.kbps < 0:
                logger.warning(f"Negative kbps value received: {request.kbps}")
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details("kbps value cannot be negative")
                return bitrate_pb2.Embedding()

            # Prepare input tensor - normalize kbps to reasonable range
            normalized_kbps = request.kbps / 1000.0
            input_tensor = torch.tensor([[[normalized_kbps]]], dtype=torch.float32)

            # Forward pass through LSTM only (no final FC layer)
            with torch.no_grad():
                lstm_output, hidden_state = self.model.lstm(input_tensor)

            # Extract embedding from last timestep
            embedding = lstm_output[:, -1, :].numpy().flatten()

            # Convert to list for protobuf
            embedding_values = embedding.tolist()

            # Calculate latency
            latency_ms = (time.perf_counter() - start_time) * 1000

            logger.info(
                f"[PREFILL] kbps={request.kbps:.1f}, "
                f"embedding_dim={len(embedding_values)}, "
                f"latency={latency_ms:.2f}ms"
            )

            return bitrate_pb2.Embedding(values=embedding_values)

        except Exception as e:
            logger.error(f"Error in Prefill: {e}", exc_info=True)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Internal server error: {str(e)}")
            return bitrate_pb2.Embedding()


class GracefulServer:
    """Wrapper for gRPC server with graceful shutdown handling."""

    def __init__(self, max_workers: int = 16, port: int = 50052):
        self.max_workers = max_workers
        self.port = port
        self.server: Optional[grpc.Server] = None

    def _setup_signal_handlers(self):
        """Set up signal handlers for graceful shutdown."""

        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating graceful shutdown...")
            self.stop()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    def start(self):
        """Start the gRPC server."""
        try:
            # Create server with thread pool
            self.server = grpc.server(
                futures.ThreadPoolExecutor(max_workers=self.max_workers),
                options=[
                    ("grpc.keepalive_time_ms", 30000),
                    ("grpc.keepalive_timeout_ms", 5000),
                    ("grpc.keepalive_permit_without_calls", True),
                    ("grpc.http2.max_pings_without_data", 0),
                    ("grpc.http2.min_time_between_pings_ms", 10000),
                    ("grpc.http2.min_ping_interval_without_data_ms", 300000),
                ],
            )

            # Add servicer
            prefill_servicer = PrefillServicer()
            bitrate_pb2_grpc.add_BitrateServiceServicer_to_server(
                prefill_servicer, self.server
            )

            # Add reflection for debugging (optional)
            service_names = (
                bitrate_pb2.DESCRIPTOR.services_by_name["BitrateService"].full_name,
                reflection.SERVICE_NAME,
            )
            reflection.enable_server_reflection(service_names, self.server)

            # Bind to port
            listen_addr = f"[::]:{self.port}"
            self.server.add_insecure_port(listen_addr)

            # Start server
            self.server.start()
            logger.info(f"Prefill server started on {listen_addr}")
            logger.info(f"Max workers: {self.max_workers}")

            # Setup signal handlers
            self._setup_signal_handlers()

            # Wait for termination
            self.server.wait_for_termination()

        except Exception as e:
            logger.error(f"Failed to start server: {e}", exc_info=True)
            raise

    def stop(self, grace_period: int = 5):
        """Stop the server gracefully."""
        if self.server:
            logger.info(f"Stopping server with {grace_period}s grace period...")
            self.server.stop(grace_period)
            logger.info("Server stopped")


def main():
    """Main entry point."""
    # Read configuration from environment
    max_workers = int(os.getenv("GRPC_MAX_WORKERS", "16"))
    port = int(os.getenv("GRPC_PORT", "50052"))

    logger.info("Starting Bitrate Prefill Server...")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")

    try:
        server = GracefulServer(max_workers=max_workers, port=port)
        server.start()
    except KeyboardInterrupt:
        logger.info("Server interrupted by user")
    except Exception as e:
        logger.error(f"Server failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
