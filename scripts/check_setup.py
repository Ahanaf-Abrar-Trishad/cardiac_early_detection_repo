import os, torch
print("PYTHONPATH =", os.environ.get("PYTHONPATH", "<unset>"))
print("torch      =", torch.__version__)
print("cuda.is_available =", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device name:", torch.cuda.get_device_name(0))
