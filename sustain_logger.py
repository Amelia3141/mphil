"""
Logging configuration for OrdinalSustain GPU analysis.

Provides structured logging with both file and console output,
color-coded messages, and progress tracking.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path


class ColoredFormatter(logging.Formatter):
    """Custom formatter with color-coded log levels."""

    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'        # Reset
    }

    def format(self, record):
        # Add color to levelname
        if record.levelname in self.COLORS:
            record.levelname = (
                f"{self.COLORS[record.levelname]}{record.levelname}"
                f"{self.COLORS['RESET']}"
            )
        return super().format(record)


def setup_logger(name='sustain_analysis', config=None):
    """
    Set up logger with file and console handlers.

    Args:
        name: Logger name
        config: Logging configuration dict (optional)

    Returns:
        Configured logger instance
    """
    # Default config
    if config is None:
        config = {
            'level': 'INFO',
            'log_file': './sustain_analysis.log',
            'console_output': True
        }

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, config.get('level', 'INFO')))

    # Clear existing handlers
    logger.handlers.clear()

    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    console_formatter = ColoredFormatter(
        '%(levelname)-8s | %(message)s'
    )

    # File handler
    if config.get('log_file'):
        log_path = Path(config['log_file'])
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.DEBUG)  # File gets all messages
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    # Console handler
    if config.get('console_output', True):
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, config.get('level', 'INFO')))
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    return logger


def log_section(logger, title, char='=', width=70):
    """Log a formatted section header."""
    logger.info(char * width)
    logger.info(title.center(width))
    logger.info(char * width)


def log_config(logger, config):
    """Log configuration details."""
    logger.info("Configuration:")
    for key, value in config.items():
        if isinstance(value, dict):
            logger.info(f"  {key}:")
            for sub_key, sub_value in value.items():
                logger.info(f"    {sub_key}: {sub_value}")
        else:
            logger.info(f"  {key}: {value}")


def log_data_info(logger, prob_nl, prob_score, biomarker_labels):
    """Log dataset information."""
    n_subjects = prob_nl.shape[0]
    n_biomarkers = prob_nl.shape[1]
    n_scores = prob_score.shape[2]

    logger.info("Dataset Information:")
    logger.info(f"  Subjects: {n_subjects:,}")
    logger.info(f"  Biomarkers: {n_biomarkers}")
    logger.info(f"  Severity levels: {n_scores}")
    logger.info(f"  Biomarker labels: {', '.join(biomarker_labels)}")


def log_runtime_stats(logger, start_time, end_time):
    """Log runtime statistics."""
    runtime = end_time - start_time
    hours = runtime / 3600
    days = hours / 24

    logger.info("Runtime Statistics:")
    logger.info(f"  Start: {datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"  End: {datetime.fromtimestamp(end_time).strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"  Duration: {runtime:.1f}s = {hours:.1f}h = {days:.2f}d")

    if days < 30:
        speedup = 30 / days
        logger.info(f"  GPU Speedup: {speedup:.1f}x vs CPU estimate (30 days)")


def log_gpu_info(logger, sustain_instance):
    """Log GPU device information."""
    if hasattr(sustain_instance, 'use_gpu') and sustain_instance.use_gpu:
        device = sustain_instance.torch_backend.device_manager.device
        logger.info(f"GPU Device: {device}")

        try:
            import torch
            if torch.cuda.is_available():
                props = torch.cuda.get_device_properties(0)
                logger.info(f"  Name: {props.name}")
                logger.info(f"  Memory: {props.total_memory / 1024**3:.1f} GB")
                logger.info(f"  Compute Capability: {props.major}.{props.minor}")
        except Exception as e:
            logger.debug(f"Could not get detailed GPU info: {e}")
    else:
        logger.warning("GPU not available - running on CPU")
