def resolve_device(device: str = "auto") -> str:
    """Use CUDA when it is healthy, otherwise keep training available on CPU."""
    if device != "auto":
        return device
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        # A broken/incomplete CUDA runtime must not stop CPU-only deployments.
        return "cpu"
