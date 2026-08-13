from __future__ import annotations

import gc
import os
from typing import Any

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def llm_engine_kwargs(model: str, max_model_len: int = 4096, trust_remote_code: bool = False) -> dict:
    return {
        "model": model,
        "dtype": "float16",
        "trust_remote_code": trust_remote_code,
        "enforce_eager": False,
        "gpu_memory_utilization": 0.85,
        "tensor_parallel_size": 1,
        "disable_log_stats": True,
        "disable_custom_all_reduce": True,
        "max_model_len": max_model_len,
    }


def cleanup_llm(llm: Any) -> None:
    engine = getattr(llm, "llm_engine", None)
    if engine is not None:
        shutdown = getattr(engine, "shutdown", None)
        if callable(shutdown):
            shutdown()

    del llm
    gc.collect()

    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
