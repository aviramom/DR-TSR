from typing import List

from .base_logger import BaseLogger
from .print_logger import PrintLogger, TqdmLogger, LoggerL
from .wandb_logger import WandbLogger
from .composite_logger import CompositeLogger


def setup_logger(args, rank=0):
    """Set up the experiment logger based on args."""
    loggers: List[BaseLogger] = [TqdmLogger(rank=rank)]

    if args.use_wandb:
        try:
            wandb_logger = WandbLogger(
                project=args.project,
                rank=rank,
                stdout=True,
                configs=args,
            )
            loggers.append(wandb_logger)
        except FileNotFoundError:
            print("Wandb credentials not found, falling back to PrintLogger only")

    return CompositeLogger(loggers) if len(loggers) > 1 else loggers[0]


__all__ = [
    'BaseLogger',
    'PrintLogger',
    'TqdmLogger',
    'LoggerL',
    'WandbLogger',
    'CompositeLogger',
]
