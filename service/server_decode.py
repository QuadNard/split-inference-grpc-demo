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

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("decode_server.log"),
    ],
)
logger = logging.getLogger(__name__)


class DecodeServicer(bitrate_pb2_grpc.BitrateServiceServicer):
    """Service for handling bitrate decode requests using LSTM model."""

    def __init__(self):
        """Initialize the servicer with the LSTM model."""
        try:
            self.model = BitrateLSTM()
            self.model.eval()
            logger.info("LSTM model loaded successfully for decode service")
        except Exception as e:
            logger.error(f"Failed to load LSTM model: {e}")
            raise

    def Decode(self, request, context):
        """
        Decode embeddings back to bitrate values using the FC layer.

        Args:
            request: Embedding message containing values list
            context: gRPC context

        Returns:
            Bitrate message containing the decoded kbps value
        """
        start_time = time.perf_counter()

        try:
            # Validate input
            if not hasattr(request, "values"):
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details("Missing 'values' field in request")
                return bitrate_pb2.Bitrate()

            if not request.values:
                logger.warning("Empty embedding values received")
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details("Embedding values cannot be empty")
                return bitrate_pb2.Bitrate()

            # Validate embedding dimensions
            embedding_values = list(request.values)
            if len(embedding_values) == 0:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details("Embedding values list is empty")
                return bitrate_pb2.Bitrate()

            # Check for invalid values (NaN, inf)
            if any(not self._is_valid_float(val) for val in embedding_values):
                logger.warning(
                    f"Invalid values in embedding: {embedding_values[:5]}..."
                )
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details("Embedding contains invalid values (NaN or inf)")
                return bitrate_pb2.Bitrate()

            # Convert embedding back to torch tensor
            # Shape: [batch_size, embedding_dim] = [1, len(values)]
            hidden_tensor = torch.tensor([embedding_values], dtype=torch.float32)

            # Forward pass through FC layer only
            with torch.no_grad():
                output = self.model.fc(hidden_tensor)

            # Convert to bitrate (assuming model outputs normalized values)
            # Scale factor of 4000 based on original code - may need adjustment
            raw_output = float(output.item())
            bitrate_kbps = raw_output * 4000.0

            # Clamp bitrate to reasonable bounds
            bitrate_kbps = max(0.0, min(bitrate_kbps, 1000000.0))  # 0 to 1Gbps

            # Calculate latency
            latency_ms = (time.perf_counter() - start_time) * 1000

            logger.info(
                f"[DECODE] embedding_dim={len(embedding_values)}, "
                f"raw_output={raw_output:.4f}, "
                f"bitrate={bitrate_kbps:.1f}kbps, "
                f"latency={latency_ms:.2f}ms"
            )

            return bitrate_pb2.Bitrate(kbps=bitrate_kbps)

        except torch.OutOfMemoryError as e:
            logger.error(f"Out of memory in Decode: {e}")
            context.set_code(grpc.StatusCode.RESOURCE_EXHAUSTED)
            context.set_details("Insufficient memory to process request")
            return bitrate_pb2.Bitrate()

        except Exception as e:
            logger.error(f"Error in Decode: {e}", exc_info=True)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Internal server error: {str(e)}")
            return bitrate_pb2.Bitrate()

    def _is_valid_float(self, value) -> bool:
        """Check if a value is a valid float (not NaN or inf)."""
        try:
            return not (
                torch.isnan(torch.tensor(value)) or torch.isinf(torch.tensor(value))
            )
        except (TypeError, ValueError):
            return False


class GracefulDecodeServer:
    """Wrapper for gRPC decode server with graceful shutdown handling."""

    def __init__(self, max_workers: int = 4, port: int = 50053):
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
        """Start the gRPC decode server."""
        try:
            # Create server with thread pool (smaller pool for decode operations)
            self.server = grpc.server(
                futures.ThreadPoolExecutor(max_workers=self.max_workers),
                options=[
                    ("grpc.keepalive_time_ms", 30000),
                    ("grpc.keepalive_timeout_ms", 5000),
                    ("grpc.keepalive_permit_without_calls", True),
                    ("grpc.http2.max_pings_without_data", 0),
                    ("grpc.http2.min_time_between_pings_ms", 10000),
                    ("grpc.http2.min_ping_interval_without_data_ms", 300000),
                    ("grpc.max_receive_message_length", 4 * 1024 * 1024),  # 4MB
                    ("grpc.max_send_message_length", 4 * 1024 * 1024),  # 4MB
                ],
            )

            # Add servicer
            decode_servicer = DecodeServicer()
            bitrate_pb2_grpc.add_BitrateServiceServicer_to_server(
                decode_servicer, self.server
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
            logger.info(f"Decode server started on {listen_addr}")
            logger.info(f"Max workers: {self.max_workers}")

            # Setup signal handlers
            self._setup_signal_handlers()

            # Wait for termination
            self.server.wait_for_termination()

        except Exception as e:
            logger.error(f"Failed to start decode server: {e}", exc_info=True)
            raise

    def stop(self, grace_period: int = 5):
        """Stop the server gracefully."""
        if self.server:
            logger.info(f"Stopping decode server with {grace_period}s grace period...")
            self.server.stop(grace_period)
            logger.info("Decode server stopped")


def main():
    """Main entry point."""
    # Read configuration from environment
    max_workers = int(os.getenv("GRPC_MAX_WORKERS", "4"))
    port = int(os.getenv("GRPC_PORT", "50053"))

    logger.info("Starting Bitrate Decode Server...")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")

    try:
        server = GracefulDecodeServer(max_workers=max_workers, port=port)
        server.start()
    except KeyboardInterrupt:
        logger.info("Decode server interrupted by user")
    except Exception as e:
        logger.error(f"Decode server failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
