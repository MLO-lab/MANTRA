"""GPU utility functions."""

import logging

import torch

logger = logging.getLogger(__name__)


def get_free_gpu_idx() -> int:
    """Get the index of the GPU with the most free memory.

    Returns
    -------
    int
        Index of the GPU with lowest memory usage.
        Returns 0 if no GPU is available or memory info cannot be retrieved.

    Example
    -------
    >>> from mantra.utils import get_free_gpu_idx
    >>> gpu_idx = get_free_gpu_idx()
    >>> device = f"cuda:{gpu_idx}"
    """
    if not torch.cuda.is_available():
        logger.debug("No CUDA devices available, returning 0")
        return 0

    try:
        num_gpus = torch.cuda.device_count()
        if num_gpus == 0:
            return 0

        # Get memory usage for each GPU
        memory_usage = []
        for i in range(num_gpus):
            mem_allocated = torch.cuda.memory_allocated(i)
            memory_usage.append(mem_allocated)

        # Return the GPU with lowest memory usage
        best_gpu = memory_usage.index(min(memory_usage))
        logger.debug(
            "Selected GPU %d (memory usage: %d bytes)",
            best_gpu,
            memory_usage[best_gpu],
        )
        return best_gpu

    except Exception as e:
        logger.warning("Could not determine free GPU, defaulting to 0: %s", e)
        return 0
