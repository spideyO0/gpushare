import torch
from torchvision.utils import save_image

# ── Detect All GPUs ───────────────────────────────────────
print("=" * 50)
print("       GPU DETECTION & SELECTOR")
print("=" * 50)
print(f"PyTorch Version  : {torch.__version__}")
print(f"CUDA Available   : {torch.cuda.is_available()}")
print()

if not torch.cuda.is_available():
    print("No CUDA GPUs detected on this machine.")
    print("   Possible reasons:")
    print("   - PyTorch CPU-only build installed")
    print("   - No NVIDIA GPU physically present")
    print("   - Missing/outdated NVIDIA drivers")
    print()
    print("   Fix: pip uninstall torch torchvision torchaudio -y")
    print("        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124")
    exit()

# ── List All GPUs ─────────────────────────────────────────
# With gpushare installed, this includes both local and remote GPUs
# automatically via the startup hook. No extra imports needed.
gpu_count = torch.cuda.device_count()
print(f"Found {gpu_count} GPU(s):\n")

for i in range(gpu_count):
    props = torch.cuda.get_device_properties(i)
    marker = " <-- current default" if i == torch.cuda.current_device() else ""
    print(f"  [{i}] {props.name}{marker}")
    print(f"       VRAM        : {props.total_memory / 1024**3:.2f} GB")
    print(f"       Compute Cap : {props.major}.{props.minor}")
    print(f"       SMs         : {props.multi_processor_count}")
    print()

# ── Interactive Selection ─────────────────────────────────
while True:
    try:
        choice = input(f"Select GPU index [0 - {gpu_count - 1}]: ").strip()
        selected = int(choice)
        if 0 <= selected < gpu_count:
            break
        else:
            print(f"Invalid choice. Enter a number between 0 and {gpu_count - 1}.")
    except ValueError:
        print("Please enter a valid integer.")

# ── Set Device ────────────────────────────────────────────
torch.cuda.set_device(selected)
props = torch.cuda.get_device_properties(selected)
print()
print(f"Selected        : cuda:{selected} -- {props.name}")
print(f"   VRAM Total     : {props.total_memory / 1024**3:.2f} GB")
print()

# ── Generate Image on Selected GPU ───────────────────────
# Check if this is a remote device (device index >= local count)
# For remote devices, gpushare handles the GPU operations via its Python API
try:
    local_count = len([i for i in range(torch.cuda.device_count())
                       if i < selected or not hasattr(torch.cuda.get_device_properties(i), '_gpushare_remote')])
except Exception:
    local_count = torch.cuda.device_count()

is_remote = False
try:
    import gpushare_hook
    is_remote = gpushare_hook._is_remote_device(selected)
except Exception:
    pass

if is_remote:
    print("Generating 512x512 image on REMOTE GPU (via gpushare)...")
    import numpy as np
    import gpushare
    gpu = gpushare.connect()

    # Create random data, send to remote GPU, get it back
    host_data = np.random.rand(1, 3, 512, 512).astype(np.float32)
    size = host_data.nbytes
    dev_ptr = gpu.malloc(size)
    gpu.memcpy_h2d(dev_ptr, host_data.tobytes())
    result = np.empty_like(host_data)
    gpu.memcpy_d2h(result, dev_ptr, size)
    gpu.free(dev_ptr)

    noise_image = torch.from_numpy(result)
    output_file = f"generated_gpu{selected}.png"
    save_image(noise_image, output_file)

    print(f"Transfer Size     : {size / 1024**2:.2f} MB (round-trip via network)")
    print(f"Image saved       : {output_file}")
else:
    print("Generating 512x512 image on LOCAL GPU...")
    device = torch.device(f"cuda:{selected}")
    noise_image = torch.rand(1, 3, 512, 512, device=device)

    output_file = f"generated_gpu{selected}.png"
    save_image(noise_image, output_file)

    print(f"Tensor Device     : {noise_image.device}")
    print(f"Memory Used       : {torch.cuda.memory_allocated(selected) / 1024**2:.2f} MB")
    print(f"Image saved       : {output_file}")
