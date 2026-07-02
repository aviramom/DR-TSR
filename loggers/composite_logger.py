from typing import Any, Dict, List

from .base_logger import BaseLogger


class CompositeLogger(BaseLogger):

    def __init__(self, loggers, *args, **kwargs):
        super(CompositeLogger, self).__init__(*args, **kwargs)
        self.loggers = loggers

    def __enter__(self):
        return self

    def stop(self):
        if self.rank == 0:
            for logger in self.loggers:
                logger.stop()

    def __exit__(self, type, value, traceback):
        self.stop()

    def log(self, name: str, data: Any, step=None):
        if self.rank == 0:
            for logger in self.loggers:
                logger.log(name, data, step)

    def _log_fig(self, name: str, fig: Any):
        if self.rank == 0:
            for logger in self.loggers:
                logger.log_fig(name, fig)

    def log_hparams(self, params: Dict[str, Any]):
        if self.rank == 0:
            for logger in self.loggers:
                logger.log_hparams(params)

    def log_params(self, params: Dict[str, Any]):
        if self.rank == 0:
            for logger in self.loggers:
                logger.log_params(params)

    def add_tags(self, tags: List[str]):
        if self.rank == 0:
            for logger in self.loggers:
                logger.add_tags(tags)

    def add(self, name, params):
        if self.rank == 0:
            for logger in self.loggers:
                logger.add(name, params)

    def log_name_params(self, name : str, params: Any):
        if self.rank == 0:
            for logger in self.loggers:
                logger.log_name_params(name, params)

    def log_metrics(self, metrics: Dict[str, Any]):
        if self.rank == 0:
            for logger in self.loggers:
                logger.log_metrics(metrics)

    def close(self):
        self.stop()

    def log_audio(self, name : str, path : str):
        if self.rank == 0:
            for logger in self.loggers:
                logger.log_audio(name, path)
    
    def upload(self, name : str, path : str):
        if self.rank == 0:
            for logger in self.loggers:
                logger.upload(name, path)

    # Optional: proxy W&B-specific helpers when available
    def exists_run_with_args(self, args, keys_to_match=None, state=None):
        if self.rank != 0:
            return False
        for logger in self.loggers:
            if hasattr(logger, "exists_run_with_args"):
                try:
                    return logger.exists_run_with_args(args, keys_to_match, state)
                except Exception:
                    return False
        return False

    def get_matching_run(self, args, keys_to_match=None, state=None):
        if self.rank != 0:
            return None
        for logger in self.loggers:
            if hasattr(logger, "get_matching_run"):
                try:
                    return logger.get_matching_run(args, keys_to_match, state)
                except Exception:
                    return None
        return None


    def is_completed(self):
        if self.rank != 0:
            return False
        for logger in self.loggers:
            if hasattr(logger, "is_completed"):
                try:
                    if logger.is_completed():
                        return True
                except Exception:
                    continue
        return False
