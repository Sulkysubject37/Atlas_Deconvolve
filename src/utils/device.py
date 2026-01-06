import torch

def get_device(cfg_device_str: str = "auto") -> torch.device:
    """
    Robustly determines the best available device (CUDA, MPS, or CPU).
    
    Args:
        cfg_device_str (str): The device requested in configuration. 
                              If "auto", selects the best available.
                              If specific (e.g., "cuda", "cpu"), attempts to use it but falls back if unavailable.
    
    Returns:
        torch.device: The selected device.
    """
    if cfg_device_str != "auto":
        # Check if the requested device is available
        if cfg_device_str == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        if cfg_device_str == "mps" and torch.backends.mps.is_available():
            return torch.device("mps")
        if cfg_device_str == "cpu":
            return torch.device("cpu")
        # Fallback if requested device is not valid/available, proceed to auto detection
        print(f"Warning: Requested device '{cfg_device_str}' unavailable. Falling back to auto-detection.")

    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")
