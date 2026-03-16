import subprocess
import platform
import sys
import ctypes
import ctypes.util
import os


def run_command(cmd):
    try:
        result = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True, text=True)
        return result.strip()
    except Exception:
        return None


def detect_nvidia_smi():
    """Try nvidia-smi binary (real or gpushare shim)."""
    print("=== NVIDIA GPU Detection (nvidia-smi) ===")
    output = run_command(
        "nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used,"
        "temperature.gpu,utilization.gpu --format=csv"
    )
    if output:
        print(output)
        return True
    return False


def detect_nvidia_nvml():
    """Directly load NVML library — works even without nvidia-smi binary."""
    print("=== NVIDIA GPU Detection (NVML) ===")

    # Find the NVML library (real or gpushare)
    lib = None
    lib_names = []
    system = platform.system()

    if system == "Darwin":
        lib_names = [
            "/usr/local/lib/libnvidia-ml.dylib",
            "/usr/local/lib/gpushare/libgpushare_client.dylib",
        ]
    elif system == "Windows":
        lib_names = [
            "nvml.dll",
            os.path.join(os.environ.get("ProgramFiles", ""), "gpushare", "nvml.dll"),
        ]
    else:
        lib_names = [
            "/usr/local/lib/gpushare/libnvidia-ml.so.1",
            "/usr/local/lib/gpushare/libgpushare_client.so",
        ]
        # Also try system library
        found = ctypes.util.find_library("nvidia-ml")
        if found:
            lib_names.insert(0, found)

    for name in lib_names:
        try:
            lib = ctypes.CDLL(name)
            break
        except OSError:
            continue

    if lib is None:
        print("  NVML library not found")
        return False

    # Init
    err = lib.nvmlInit_v2()
    if err != 0:
        print("  nvmlInit failed (server not reachable?)")
        return False

    # Device count
    count = ctypes.c_uint(0)
    lib.nvmlDeviceGetCount_v2(ctypes.byref(count))
    print(f"  GPU count: {count.value}")

    for i in range(count.value):
        dev = ctypes.c_void_p()
        lib.nvmlDeviceGetHandleByIndex_v2(i, ctypes.byref(dev))

        name_buf = ctypes.create_string_buffer(256)
        lib.nvmlDeviceGetName(dev, name_buf, 256)

        driver = ctypes.create_string_buffer(80)
        lib.nvmlSystemGetDriverVersion(driver, 80)

        class Mem(ctypes.Structure):
            _fields_ = [("total", ctypes.c_ulonglong),
                        ("free", ctypes.c_ulonglong),
                        ("used", ctypes.c_ulonglong)]
        mem = Mem()
        lib.nvmlDeviceGetMemoryInfo(dev, ctypes.byref(mem))

        temp = ctypes.c_uint(0)
        lib.nvmlDeviceGetTemperature(dev, 0, ctypes.byref(temp))

        class Util(ctypes.Structure):
            _fields_ = [("gpu", ctypes.c_uint), ("memory", ctypes.c_uint)]
        util = Util()
        lib.nvmlDeviceGetUtilizationRates(dev, ctypes.byref(util))

        power = ctypes.c_uint(0)
        lib.nvmlDeviceGetPowerUsage(dev, ctypes.byref(power))

        power_limit = ctypes.c_uint(0)
        lib.nvmlDeviceGetEnforcedPowerLimit(dev, ctypes.byref(power_limit))

        print(f"\n  GPU {i}: {name_buf.value.decode()}")
        print(f"    Driver:      {driver.value.decode()}")
        print(f"    VRAM:        {mem.used // (1024**2)} / {mem.total // (1024**2)} MiB")
        print(f"    Temperature: {temp.value} C")
        print(f"    GPU Util:    {util.gpu}%")
        print(f"    Power:       {power.value / 1000:.1f}W / {power_limit.value / 1000:.1f}W")

    lib.nvmlShutdown()
    return True


def detect_cuda_driver():
    """Load CUDA driver API — tests if CUDA device is accessible."""
    print("\n=== CUDA Driver API ===")

    lib = None
    system = platform.system()
    if system == "Darwin":
        candidates = ["/usr/local/lib/libcuda.dylib",
                      "/usr/local/lib/gpushare/libgpushare_client.dylib"]
    elif system == "Windows":
        candidates = [
            "nvcuda.dll",
            os.path.join(os.environ.get("ProgramFiles", ""), "gpushare", "nvcuda.dll"),
            os.path.join(os.environ.get("ProgramFiles", ""), "gpushare", "gpushare_client.dll"),
        ]
        # Also check local build directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        for sub in ["build\\Release", "build\\Debug", "build"]:
            candidates.append(os.path.join(script_dir, "..", sub, "gpushare_client.dll"))
    else:
        candidates = []
        found = ctypes.util.find_library("cuda")
        if found:
            candidates.append(found)
        candidates += ["/usr/local/lib/gpushare/libcuda.so.1",
                       "/usr/local/lib/gpushare/libgpushare_client.so"]

    for c in candidates:
        try:
            lib = ctypes.CDLL(c)
            break
        except OSError:
            continue

    if lib is None:
        print("  libcuda not found")
        return False

    err = lib.cuInit(0)
    if err != 0:
        print(f"  cuInit failed: {err}")
        return False

    count = ctypes.c_int(0)
    lib.cuDeviceGetCount(ctypes.byref(count))
    print(f"  CUDA devices: {count.value}")

    for i in range(count.value):
        dev = ctypes.c_int(0)
        lib.cuDeviceGet(ctypes.byref(dev), i)

        name = ctypes.create_string_buffer(256)
        lib.cuDeviceGetName(name, 256, dev)

        total = ctypes.c_size_t(0)
        lib.cuDeviceTotalMem_v2(ctypes.byref(total), dev)

        major = ctypes.c_int(0)
        minor = ctypes.c_int(0)
        lib.cuDeviceComputeCapability(ctypes.byref(major), ctypes.byref(minor), dev)

        print(f"  Device {i}: {name.value.decode()}")
        print(f"    VRAM: {total.value // (1024**2)} MiB")
        print(f"    Compute: SM {major.value}.{minor.value}")

    return True


def detect_pytorch():
    try:
        import torch
        print("\n=== PyTorch GPU Info ===")
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
                props = torch.cuda.get_device_properties(i)
                print(f"    Total Memory: {props.total_memory / (1024**3):.2f} GB")
                print(f"    CUDA Capability: {props.major}.{props.minor}")
                print(f"    Multiprocessors: {props.multi_processor_count}")
        else:
            print("  PyTorch CUDA: not available")
            print("  (macOS PyTorch is CPU-only; use gpushare Python API directly)")
        return True
    except ImportError:
        return False


def detect_opencl():
    try:
        import pyopencl as cl
        print("\n=== OpenCL GPU Info ===")
        platforms = cl.get_platforms()
        for p in platforms:
            print(f"  Platform: {p.name}")
            devices = p.get_devices()
            for d in devices:
                print(f"    Device: {d.name}")
                print(f"      Type: {cl.device_type.to_string(d.type)}")
                print(f"      Compute Units: {d.max_compute_units}")
                print(f"      Memory: {d.global_mem_size / (1024**3):.2f} GB")
        return True
    except Exception:
        return False


def detect_gpushare_python():
    """Test the native gpushare Python client."""
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
        import gpushare
        print("\n=== gpushare Python Client ===")
        gpu = gpushare.connect()
        props = gpu.device_properties()
        print(f"  GPU: {props['name']}")
        print(f"  VRAM: {props['total_mem_mb']:.0f} MB")
        print(f"  SM: {props['major']}.{props['minor']}")
        print(f"  SMs: {props['multi_processor_count']}")
        gpu.close()
        return True
    except Exception as e:
        print(f"\n=== gpushare Python Client ===")
        print(f"  Not connected: {e}")
        return False


def main():
    print("================================")
    print("GPU Detection Script")
    print("================================")
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"Python: {sys.version.split()[0]}")
    print("--------------------------------")

    found = False

    # 1. nvidia-smi (binary or shim)
    if detect_nvidia_smi():
        found = True

    # 2. NVML direct (works even without nvidia-smi)
    if detect_nvidia_nvml():
        found = True

    # 3. CUDA driver API
    if detect_cuda_driver():
        found = True

    # 4. PyTorch
    if detect_pytorch():
        found = True

    # 5. OpenCL
    if detect_opencl():
        found = True

    # 6. gpushare Python client
    if detect_gpushare_python():
        found = True

    print("\n================================")
    if found:
        print("GPU detected successfully!")
    else:
        print("No GPU detected.")
        print("If using gpushare, check that the server is running")
        print("and the client config has the correct server address.")
    print("================================")


if __name__ == "__main__":
    main()
