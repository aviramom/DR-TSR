import warnings
from pathlib import Path

from .base_logger import BaseLogger
from typing import Dict, Any, List, Optional
import numpy as np
from PIL import Image


def is_basic(x):
    return isinstance(x, str) or isinstance(x, int) or isinstance(x, float) or isinstance(x, bool)


def convert_no_basic_to_str(sub_dict: Dict[str, Any]):
    return {k: v if is_basic(v)
    else str(v) if not isinstance(v, dict) else convert_no_basic_to_str(v)
            for k, v in sub_dict.items()}


def convert_no_basic_to_str_from_any(p: Any):
    if is_basic(p):
        return p
    elif isinstance(p, dict):
        return convert_no_basic_to_str(p)
    else:
        return str(p)


class WandbLogger(BaseLogger):

    def __init__(self, project=None, stdout=True, configs=None, *args, **kwargs):
        super(WandbLogger, self).__init__(*args, **kwargs)
        if self.rank != 0:
            return
        import wandb
        self.wandb = wandb

        # Check for login token if not logged in
        if self.wandb.api.api_key is None:
            local_dir_api_token = ['wandb_logger', 'logger', '']
            for folder in local_dir_api_token:
                local_path_api_token = Path(folder) / 'token.txt'
                if local_path_api_token.exists():
                    token = local_path_api_token.read_text().strip()
                    self.wandb.login(key=token)
                    break
                else:
                    warnings.warn('''Please create a file at wandb_logger/token.txt with your WandB API token''')
                    raise FileNotFoundError('WandB API token file not found')

        local_dir_api_project = ['wandb_logger', 'logger', '']
        if project is None:
            for folder in local_dir_api_project:
                local_path_api_project = Path(folder) / 'project.txt'
                if local_path_api_project.exists():
                    project = local_path_api_project.read_text().strip()
                    entity = project.split('/')[0]
                    project = project.split('/')[-1]
                    break
        else:
            entity = project.split('/')[0]
            project = project.split('/')[-1]
        if project is None:
            warnings.warn('''Please create a file at neptune/project.txt with your Neptune project name''')
            raise FileNotFoundError('project file name not found')
        # Persist for later API queries
        self.entity = entity
        self.project = project
        run = self.get_matching_run(args_dict=vars(configs), keys_to_match = configs.keys_to_match) if configs is not None else None
        if run is not None and not configs.override_run:
            print(f"Found existing wandb run: {run.name} ({run.id}), reusing it.")
            self.run = run
            self.wandb.run = run
            self.completed = True
        else:
            self.completed = False
            self.run = wandb.init(
                # set the wandb project where this run will be logged
                project=project,
                entity=entity,
                settings=wandb.Settings(console="off") if not stdout else None,
            )

    def is_completed(self):
        return self.completed

    def stop(self):
        if self.rank == 0 :
            # add field to mark the run as finished
            self.wandb.run.summary["finished"] = True
            self.run.finish()

    def log(self, name: str, data: Any, step=None):
        if self.rank == 0:
            self.wandb.log({name: data})

    def log_metrics(self, metrics: Dict[str, Any]):
        if self.rank == 0:
            self.wandb.log(metrics)

    def close(self):
        self.stop()

    def _log_fig(self, name: str, fig: Any):
        if self.rank == 0:
            if isinstance(fig, np.ndarray):
                if fig.dtype != np.uint8:
                    fig = fig * 255
                    fig = fig.astype(np.uint8)
                fig = Image.fromarray(fig)
            self.wandb.log({name: self.wandb.Image(fig)})

    def log_hparams(self, params: Dict[str, Any]):
        if self.rank == 0:
            params = convert_no_basic_to_str(params)
            if isinstance(params, dict):
                self.wandb.config.update(params)
            else:
                self.wandb.config.update({'hparams': params})


    def log_params(self, params: Dict[str, Any]):
        if self.rank == 0:
            params = convert_no_basic_to_str(params)
            if isinstance(params, dict):
                self.wandb.config.update(params)
            else:
                self.wandb.config.update({'params': params})

    def add_tags(self, tags: List[str]):
        if self.rank == 0:
            self.run.tags = self.run.tags + tuple(tags)

    def log_name_params(self, name: str, params: Any):
        if self.rank == 0:
            params = convert_no_basic_to_str_from_any(params)
            self.wandb.config.update({name: params}, allow_val_change=True)

    def add(self, name: str, params: Any):
        if self.rank == 0:
            self.wandb.log({name: params})

    def log_audio(self, name: str, path: str):
        if self.rank == 0:
            self.wandb.log({name: self.wandb.Audio(path)})

    def upload(self, name: str, path: str):
        if self.rank == 0:
            self.wandb.save(path)

    def get_matching_run(
        self,
        args_dict: Dict[str, Any],
        keys_to_match: Optional[List[str]] = None,
        state: Optional[str] = None,
    ) -> Optional[Any]:
        """
        Return the first matching W&B run (or None) with the same config keys.
        Useful if you want to resume or fetch artifacts.
        """
        if self.rank != 0:
            return None

        api = self.wandb.Api()
        path = f"{self.entity}/{self.project}"

        flat_args: Dict[str, Any] = convert_no_basic_to_str(args_dict) if isinstance(args_dict, dict) else {}
        if keys_to_match is None or len(keys_to_match) == 0:
            keys_to_match = [k for k, v in flat_args.items() if is_basic(v)]

        filters: Dict[str, Any] = {}
        if state:
            filters["state"] = state
        # Ensure we only look at finished runs (marked in summary)
        # Server-side filter uses summary metrics namespace
        filters["summary_metrics.finished"] = True
        for k in keys_to_match:
            if k in flat_args:
                filters[f"config.{k}"] = flat_args[k]

        try:
            runs = api.runs(path=path, filters=filters)
        except Exception:
            runs = api.runs(path=path)

        def _normalize(v: Any) -> Any:
            return convert_no_basic_to_str_from_any(v)
        
        for run in runs:
            cfg = dict(run.config or {})
            # Double-check the run is marked finished in its summary
            if not bool(getattr(run, "summary", {})).__bool__():
                continue
            if not bool(run.summary.get("finished", False)):
                continue
            if all(_normalize(cfg.get(k)) == _normalize(flat_args.get(k)) for k in keys_to_match):
                return run
        return None
