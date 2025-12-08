"""immich-ml-tag: ML-based automatic tagging for Immich."""

import argparse
import logging
import os
import signal
import sys
import time
from datetime import datetime

from .train import run_inference, run_training


def setup_logging(verbose: bool = False):
    """Configure logging for the application."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


logger = logging.getLogger(__name__)

# Environment variable defaults
DEFAULT_TRAIN_TIME = os.environ.get("TRAIN_TIME", "02:00")
DEFAULT_INFERENCE_INTERVAL = int(os.environ.get("INFERENCE_INTERVAL", "5"))
DEFAULT_THRESHOLD = float(os.environ.get("THRESHOLD", "0.5"))
DEFAULT_MIN_SAMPLES = int(os.environ.get("MIN_SAMPLES", "10"))


def run_scheduler(
    train_time: str = "02:00",
    inference_interval: int = 5,
    threshold: float = 0.5,
    min_samples: int = 10,
):
    """
    Run the scheduler service.

    Args:
        train_time: Time to run training daily (HH:MM format).
        inference_interval: Minutes between inference runs.
        threshold: Prediction threshold for tagging.
        min_samples: Minimum samples required for training.
    """
    # Parse training time
    try:
        train_hour, train_minute = map(int, train_time.split(":"))
    except ValueError:
        logger.error(f"Invalid train time format: {train_time}. Use HH:MM format.")
        sys.exit(1)

    logger.info("Starting scheduler service")
    logger.info(f"  - Training daily at {train_time}")
    logger.info(f"  - Inference every {inference_interval} minutes")
    logger.info(f"  - Threshold: {threshold}")

    last_train_date = None
    last_inference_time = None
    running = True

    def handle_signal(signum, frame):
        nonlocal running
        logger.info("Received shutdown signal, stopping...")
        running = False

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    while running:
        now = datetime.now()

        # Check if we should run training
        if now.hour == train_hour and now.minute == train_minute:
            if last_train_date != now.date():
                logger.info("Starting scheduled training...")
                try:
                    run_training(
                        min_samples=min_samples,
                        threshold=threshold,
                        force=False,
                    )
                    last_train_date = now.date()
                except Exception as e:
                    logger.error(f"Training failed: {e}")

        # Check if we should run inference
        if (
            last_inference_time is None
            or (now - last_inference_time).total_seconds() >= inference_interval * 60
        ):
            logger.info("Starting scheduled inference...")
            try:
                run_inference(threshold=threshold, incremental=True)
                last_inference_time = now
            except Exception as e:
                logger.error(f"Inference failed: {e}")

        # Sleep for a bit before checking again
        time.sleep(30)

    logger.info("Scheduler stopped.")


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        prog="immich-ml-tag",
        description="ML-based automatic tagging for Immich photo library",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Train command
    train_parser = subparsers.add_parser(
        "train",
        help="Train classifiers for all tags and run inference",
    )
    train_parser.add_argument(
        "--min-samples",
        type=int,
        default=10,
        help="Minimum positive samples required for training (default: 10)",
    )
    train_parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Prediction threshold for tagging (default: 0.5)",
    )
    train_parser.add_argument(
        "--force",
        action="store_true",
        help="Force retraining even if training data hasn't changed",
    )

    # Inference command
    inference_parser = subparsers.add_parser(
        "inference",
        help="Run inference using existing models",
    )
    inference_parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Prediction threshold for tagging (default: 0.5)",
    )
    inference_parser.add_argument(
        "--full",
        action="store_true",
        help="Run full inference on all assets, not just new ones",
    )

    # Serve command (daemon mode)
    serve_parser = subparsers.add_parser(
        "serve",
        help="Run as a service with scheduled training and inference",
    )
    serve_parser.add_argument(
        "--train-time",
        type=str,
        default=DEFAULT_TRAIN_TIME,
        help=f"Time to run daily training in HH:MM format (default: {DEFAULT_TRAIN_TIME}, env: TRAIN_TIME)",
    )
    serve_parser.add_argument(
        "--inference-interval",
        type=int,
        default=DEFAULT_INFERENCE_INTERVAL,
        help=f"Minutes between inference runs (default: {DEFAULT_INFERENCE_INTERVAL}, env: INFERENCE_INTERVAL)",
    )
    serve_parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"Prediction threshold for tagging (default: {DEFAULT_THRESHOLD}, env: THRESHOLD)",
    )
    serve_parser.add_argument(
        "--min-samples",
        type=int,
        default=DEFAULT_MIN_SAMPLES,
        help=f"Minimum positive samples required for training (default: {DEFAULT_MIN_SAMPLES}, env: MIN_SAMPLES)",
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    setup_logging(args.verbose)

    if args.command == "train":
        run_training(
            min_samples=args.min_samples,
            threshold=args.threshold,
            force=args.force,
        )
    elif args.command == "inference":
        run_inference(
            threshold=args.threshold,
            incremental=not args.full,
        )
    elif args.command == "serve":
        run_scheduler(
            train_time=args.train_time,
            inference_interval=args.inference_interval,
            threshold=args.threshold,
            min_samples=args.min_samples,
        )


if __name__ == "__main__":
    main()
