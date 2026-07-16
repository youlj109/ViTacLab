import copy
import inspect
import os
import pathlib
import threading
from typing import Optional

import dill
import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf


class BaseWorkspace:
    include_keys = tuple()
    exclude_keys = tuple()

    def __init__(self, cfg: OmegaConf, output_dir: Optional[str] = None):
        self.cfg = cfg
        self._output_dir = output_dir
        self._saving_thread = None

    @property
    def output_dir(self):
        output_dir = self._output_dir
        if output_dir is None:
            output_dir = HydraConfig.get().runtime.output_dir
        return output_dir

    def run(self):
        """
        Create any resource shouldn't be serialized as local variables
        """
        pass

    def save_checkpoint(
        self,
        path=None,
        tag="latest",
        exclude_keys=None,
        include_keys=None,
        use_thread=True,
    ):
        if path is None:
            path = pathlib.Path(self.output_dir).joinpath("checkpoints", f"{tag}.ckpt")
        else:
            path = pathlib.Path(path)
        if exclude_keys is None:
            exclude_keys = tuple(self.exclude_keys)
        if include_keys is None:
            include_keys = tuple(self.include_keys) + ("_output_dir",)

        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"cfg": self.cfg, "state_dicts": dict(), "pickles": dict()}

        for key, value in self.__dict__.items():
            if hasattr(value, "state_dict") and hasattr(value, "load_state_dict"):
                # modules, optimizers and samplers etc
                if key not in exclude_keys:
                    if use_thread:
                        payload["state_dicts"][key] = _copy_to_cpu(value.state_dict())
                    else:
                        payload["state_dicts"][key] = value.state_dict()
            elif key in include_keys:
                payload["pickles"][key] = dill.dumps(value)
        if use_thread:
            self._saving_thread = threading.Thread(
                target=lambda: torch.save(payload, path.open("wb"), pickle_module=dill)
            )
            self._saving_thread.start()
        else:
            torch.save(payload, path.open("wb"), pickle_module=dill)
        return str(path.absolute())

    def get_checkpoint_path(self, tag="latest"):
        return pathlib.Path(self.output_dir).joinpath("checkpoints", f"{tag}.ckpt")

    def load_payload(self, payload, exclude_keys=None, include_keys=None, **kwargs):
        if exclude_keys is None:
            exclude_keys = tuple()
        if include_keys is None:
            include_keys = payload["pickles"].keys()

        for key, value in payload["state_dicts"].items():
            if key not in exclude_keys:
                target = self.__dict__[key]
                if hasattr(target, "state_dict"):
                    value = _adapt_state_dict_keys_for_module_prefix(
                        source_state_dict=value, target_state_dict=target.state_dict()
                    )
                load_kwargs = _filter_load_state_dict_kwargs(target, kwargs)
                target.load_state_dict(value, **load_kwargs)
        for key in include_keys:
            if key in payload["pickles"]:
                self.__dict__[key] = dill.loads(payload["pickles"][key])

    def load_checkpoint(
        self, path=None, tag="latest", exclude_keys=None, include_keys=None, **kwargs
    ):
        if path is None:
            path = self.get_checkpoint_path(tag=tag)
        else:
            path = pathlib.Path(path)
        # split kwargs for torch.load and load_state_dict
        load_state_dict_kwargs = {}
        for key in ("strict", "assign"):
            if key in kwargs:
                load_state_dict_kwargs[key] = kwargs.pop(key)

        payload = torch.load(path.open("rb"), pickle_module=dill, **kwargs)
        self.load_payload(
            payload,
            exclude_keys=exclude_keys,
            include_keys=include_keys,
            **load_state_dict_kwargs,
        )
        return payload

    @classmethod
    def create_from_checkpoint(
        cls, path, exclude_keys=None, include_keys=None, **kwargs
    ):
        payload = torch.load(open(path, "rb"), pickle_module=dill)
        instance = cls(payload["cfg"])
        instance.load_payload(
            payload=payload,
            exclude_keys=exclude_keys,
            include_keys=include_keys,
            **kwargs,
        )
        return instance

    def save_snapshot(self, tag="latest"):
        """
        Quick loading and saving for reserach, saves full state of the workspace.

        However, loading a snapshot assumes the code stays exactly the same.
        Use save_checkpoint for long-term storage.
        """
        path = pathlib.Path(self.output_dir).joinpath("snapshots", f"{tag}.pkl")
        path.parent.mkdir(parents=False, exist_ok=True)
        torch.save(self, path.open("wb"), pickle_module=dill)
        return str(path.absolute())

    @classmethod
    def create_from_snapshot(cls, path):
        return torch.load(open(path, "rb"), pickle_module=dill)


def _copy_to_cpu(x):
    if isinstance(x, torch.Tensor):
        return x.detach().to("cpu")
    elif isinstance(x, dict):
        result = dict()
        for k, v in x.items():
            result[k] = _copy_to_cpu(v)
        return result
    elif isinstance(x, list):
        return [_copy_to_cpu(k) for k in x]
    else:
        return copy.deepcopy(x)


def _adapt_state_dict_keys_for_module_prefix(source_state_dict, target_state_dict):
    source_keys = list(source_state_dict.keys())
    target_keys = list(target_state_dict.keys())
    if not source_keys or not target_keys:
        return source_state_dict

    source_has_module_prefix = all(k.startswith("module.") for k in source_keys)
    target_has_module_prefix = all(k.startswith("module.") for k in target_keys)
    if source_has_module_prefix == target_has_module_prefix:
        return source_state_dict

    # DDP -> non-DDP
    if source_has_module_prefix and not target_has_module_prefix:
        return {
            (k[len("module.") :] if k.startswith("module.") else k): v
            for k, v in source_state_dict.items()
        }

    # non-DDP -> DDP
    return {
        (k if k.startswith("module.") else f"module.{k}"): v
        for k, v in source_state_dict.items()
    }


def _filter_load_state_dict_kwargs(target, kwargs):
    if not kwargs:
        return {}
    try:
        signature = inspect.signature(target.load_state_dict)
        parameters = signature.parameters
        has_var_kwargs = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values()
        )
        if has_var_kwargs:
            return dict(kwargs)
        return {k: v for k in kwargs.items() if k in parameters}
    except (TypeError, ValueError):
        # fallback for callables without introspectable signature
        return {}
