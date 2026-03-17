import torch
import time

def list_gpus():
    if not torch.cuda.is_available():
        print("❌ CUDA is not available on this system.")
        return []

    gpu_count = torch.cuda.device_count()
    gpus = []

    print("\nAvailable GPUs:")
    for i in range(gpu_count):
        name = torch.cuda.get_device_name(i)
        total_mem = torch.cuda.get_device_properties(i).total_memory / (1024**3)
        print(f"[{i}] {name} ({total_mem:.2f} GB)")
        gpus.append(i)

    return gpus


def select_gpu(gpus):
    while True:
        try:
            choice = int(input("\nSelect GPU ID to stress test: "))
            if choice in gpus:
                return choice
            else:
                print("Invalid GPU ID.")
        except ValueError:
            print("Enter a valid number.")


def allocate_memory(device, target_gb=5):
    print(f"\nAllocating ~{target_gb} GB GPU memory...")

    bytes_per_float = 4  # float32
    target_bytes = target_gb * 1024**3
    num_elements = target_bytes // bytes_per_float

    tensor = torch.empty(int(num_elements), dtype=torch.float32, device=device)
    print("✅ Memory allocated successfully.")
    return tensor


def stress_test(device, duration_sec=60):
    print(f"\n🔥 Running stress test on {device} for {duration_sec} seconds...")

    size = 4096  # large matrix for heavy load
    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)

    torch.cuda.synchronize()
    start_time = time.time()

    iterations = 0

    while time.time() - start_time < duration_sec:
        c = torch.matmul(a, b)
        torch.cuda.synchronize()
        iterations += 1

    print(f"\n✅ Stress test completed.")
    print(f"Total iterations: {iterations}")
    print(f"Average iterations/sec: {iterations / duration_sec:.2f}")


def main():
    gpus = list_gpus()
    if not gpus:
        return

    gpu_id = select_gpu(gpus)
    device = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(device)

    print(f"\nUsing GPU: {torch.cuda.get_device_name(device)}")

    # Allocate memory
    mem_tensor = allocate_memory(device, target_gb=5)

    # Run stress test
    stress_test(device, duration_sec=60)

    # Keep memory allocated for a bit (optional)
    input("\nPress Enter to release GPU memory and exit...")

    del mem_tensor
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()