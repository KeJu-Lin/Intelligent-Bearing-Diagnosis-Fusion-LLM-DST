import torch

def print_cuda_info():
    try:
        print("-" * 40)
        print("PyTorch CUDA Environment Information:")
        print("-" * 40)
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            print(f"Number of CUDA devices: {device_count}")
            if device_count > 0:
                device_name = torch.cuda.get_device_name(0)
                print(f"0th CUDA Device Name: {device_name}")
                total_memory = torch.cuda.get_device_properties(0).total_memory
                allocated_memory = torch.cuda.memory_allocated(0)
                free_memory = total_memory - allocated_memory
                print(f"Total Memory: {total_memory / (1024 ** 3):.2f} GB")
                print(f"Allocated Memory: {allocated_memory / (1024 ** 3):.2f} GB")
                print(f"Free Memory: {free_memory / (1024 ** 3):.2f} GB")
            else:
                print("No CUDA devices found.")
        else:
            print("CUDA is not available.")
        print("-" * 40)
    except Exception as e:
        print("-" * 40)
        print(f"An error occurred: {e}")
        print("-" * 40)

if __name__ == "__main__":
    print_cuda_info()