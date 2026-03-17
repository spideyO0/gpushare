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
    print("❌ No CUDA GPUs detected on this machine.")
    print("   Possible reasons:")
    print("   - PyTorch CPU-only build installed")
    print("   - No NVIDIA GPU physically present")
    print("   - Missing/outdated NVIDIA drivers")
    print()
    print("   Fix: pip uninstall torch torchvision torchaudio -y")
    print("        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124")
    exit()

# ── List All GPUs ─────────────────────────────────────────
gpu_count = torch.cuda.device_count()
print(f"Found {gpu_count} GPU(s):\n")

for i in range(gpu_count):
    props = torch.cuda.get_device_properties(i)
    marker = " ← current default" if i == torch.cuda.current_device() else ""
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
            print(f"❌ Invalid choice. Enter a number between 0 and {gpu_count - 1}.")
    except ValueError:
        print("❌ Please enter a valid integer.")

# ── Set Device ────────────────────────────────────────────
device = torch.device(f"cuda:{selected}")
torch.cuda.set_device(device)

props = torch.cuda.get_device_properties(selected)
print()
print(f"✅ Selected        : cuda:{selected} — {props.name}")
print(f"   VRAM Total     : {props.total_memory / 1024**3:.2f} GB")
print(f"   VRAM Free      : {(props.total_memory - torch.cuda.memory_allocated(selected)) / 1024**3:.2f} GB")
print()

# ── Generate Image on Selected GPU ───────────────────────
print("Generating 512x512 image on selected GPU...")
noise_image = torch.rand(1, 3, 512, 512, device=device)

output_file = f"generated_gpu{selected}.png"
save_image(noise_image, output_file)

print(f"Tensor Device     : {noise_image.device}")
print(f"Memory Used       : {torch.cuda.memory_allocated(selected) / 1024**2:.2f} MB")
print(f"✅ Image saved    : {output_file}")
