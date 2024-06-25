import torch

if torch.cuda.is_available():
    available_gpus = torch.cuda.device_count()
    print(f"Number of available GPUs: {available_gpus}")
    for i in range(available_gpus):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("No GPUs available.")
