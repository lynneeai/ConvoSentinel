import os


def configure_gpu_device(devices=None, devices_str=None):
    """
    Set cuda visible deviceds and re-index config.devices
    This is required for device_map="auto" model loading,
    because cuda tries to get all available GPUs available,
    so CUDA_VISIBLE_DEVICES should be set to specified devices.
    After setting CUDA_VISIBLE_DEVICES, torch device index will be reset to start from 0.
    """
    assert (devices != None) ^ (devices_str != None)
    if devices_str:
        devices = devices_str.split(",")
    
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(d) for d in devices)
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"