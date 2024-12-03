import platform
import psutil
import torch

def get_hardware_info():
    info = {}

    # CPU Details
    info["CPU"] = platform.processor()
    info["Cores"] = psutil.cpu_count(logical=False)
    info["Threads"] = psutil.cpu_count(logical=True)

    # RAM Details
    info["RAM"] = f"{round(psutil.virtual_memory().total / (1024 ** 3), 2)} GB"

    # GPU Details (if available)
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        info["GPU"] = gpu_name
        info["CUDA Version"] = torch.version.cuda
    else:
        info["GPU"] = "No GPU detected"

    # OS Details
    info["OS"] = platform.system()
    info["OS Version"] = platform.version()
    info["Python Version"] = platform.python_version()

    return info

def save_hardware_info(filename="hardware_info.txt"):
    info = get_hardware_info()
    with open(filename, "w") as file:
        for key, value in info.items():
            file.write(f"{key}: {value}\n")
    print(f"Hardware information saved to {filename}")

if __name__ == "__main__":
    save_hardware_info()
