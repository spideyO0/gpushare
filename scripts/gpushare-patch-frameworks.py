#!/usr/bin/env python3
"""
gpushare-patch-frameworks — Make PyTorch/TensorFlow/JAX use the remote GPU.

ML frameworks bundle their own libcudart inside the pip package, bypassing
the system library entirely. This script replaces the bundled CUDA runtime
with our gpushare client library so frameworks see the remote GPU natively.

Usage:
    gpushare-patch-frameworks              # auto-detect and patch all
    gpushare-patch-frameworks --pytorch    # patch PyTorch only
    gpushare-patch-frameworks --undo       # restore original libraries
    gpushare-patch-frameworks --status     # check current state

After patching, 'python3 -c "import torch; print(torch.cuda.is_available())"'
will return True and use the remote GPU — zero code changes needed.
"""

import argparse
import glob
import importlib
import importlib.util
import os
import platform
import shutil
import sys

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
NC = "\033[0m"

def ok(msg):   print(f"  {GREEN}+{NC} {msg}")
def fail(msg): print(f"  {RED}x{NC} {msg}")
def info(msg): print(f"  {CYAN}>{NC} {msg}")
def warn(msg): print(f"  {YELLOW}!{NC} {msg}")


def find_gpushare_lib():
    """Find the gpushare client library."""
    system = platform.system()
    candidates = []
    if system == "Darwin":
        candidates = [
            "/usr/local/lib/gpushare/libgpushare_client.dylib",
            "/usr/local/lib/libgpushare_client.dylib",
        ]
    elif system == "Windows":
        candidates = [
            r"C:\Program Files\gpushare\gpushare_client.dll",
        ]
    else:
        candidates = [
            "/usr/local/lib/gpushare/libgpushare_client.so",
        ]

    for path in candidates:
        if os.path.isfile(path):
            return path

    # Try relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    build_dir = os.path.join(script_dir, "..", "build")
    for ext in [".so", ".dylib", ".dll"]:
        p = os.path.join(build_dir, f"libgpushare_client{ext}")
        if os.path.isfile(p):
            return p

    return None


def find_framework_cudart(framework_name):
    """Find bundled libcudart inside a Python framework package."""
    results = []

    try:
        if framework_name == "pytorch":
            import torch
            torch_dir = os.path.dirname(torch.__file__)
            # PyTorch bundles CUDA in torch/lib/
            lib_dir = os.path.join(torch_dir, "lib")
            if os.path.isdir(lib_dir):
                for f in os.listdir(lib_dir):
                    if f.startswith("libcudart") and (".so" in f or ".dylib" in f):
                        results.append(os.path.join(lib_dir, f))

        elif framework_name == "tensorflow":
            try:
                import tensorflow as tf
                tf_dir = os.path.dirname(tf.__file__)
            except ImportError:
                import importlib
                spec = importlib.util.find_spec("tensorflow")
                if spec and spec.origin:
                    tf_dir = os.path.dirname(spec.origin)
                else:
                    return results

            # TF bundles in various locations
            for pattern in [
                os.path.join(tf_dir, "**", "libcudart*"),
                os.path.join(tf_dir, "python", "_pywrap_tensorflow_internal.so"),
            ]:
                results.extend(glob.glob(pattern, recursive=True))

        elif framework_name == "jax":
            try:
                import jaxlib
                jax_dir = os.path.dirname(jaxlib.__file__)
                for f in os.listdir(jax_dir):
                    if f.startswith("libcudart") and (".so" in f or ".dylib" in f):
                        results.append(os.path.join(jax_dir, f))
                # Also check nvidia subpackage
                nvidia_dir = os.path.join(os.path.dirname(jax_dir), "nvidia")
                if os.path.isdir(nvidia_dir):
                    for f in glob.glob(os.path.join(nvidia_dir, "**", "libcudart*"), recursive=True):
                        results.append(f)
            except ImportError:
                pass

        # Also check nvidia-cuda-runtime-cu* package (pip installed)
        try:
            import importlib.metadata
            for dist in importlib.metadata.distributions():
                if "cuda-runtime" in (dist.metadata.get("Name", "") or "").lower():
                    if dist._path:
                        pkg_dir = str(dist._path.parent)
                        for f in glob.glob(os.path.join(pkg_dir, "**", "libcudart*"), recursive=True):
                            results.append(f)
        except Exception:
            pass

    except ImportError:
        pass

    # Deduplicate, skip backups
    seen = set()
    unique = []
    for r in results:
        real = os.path.realpath(r)
        if real not in seen and ".gpushare_backup" not in r:
            seen.add(real)
            unique.append(r)
    return unique


def patch_library(cudart_path, gpushare_lib, undo=False):
    """Replace a bundled libcudart with our gpushare client library."""
    backup_path = cudart_path + ".gpushare_backup"

    if undo:
        if os.path.isfile(backup_path):
            shutil.copy2(backup_path, cudart_path)
            os.remove(backup_path)
            ok(f"Restored: {cudart_path}")
            return True
        else:
            warn(f"No backup found for: {cudart_path}")
            return False

    # Create backup if not already backed up
    if not os.path.isfile(backup_path):
        shutil.copy2(cudart_path, backup_path)
        ok(f"Backed up: {cudart_path}")

    # Replace with our library
    shutil.copy2(gpushare_lib, cudart_path)
    ok(f"Patched: {cudart_path}")
    return True


def check_status(framework_name):
    """Check if a framework's CUDA libs are patched."""
    libs = find_framework_cudart(framework_name)
    if not libs:
        return "not_installed"

    for lib in libs:
        backup = lib + ".gpushare_backup"
        if os.path.isfile(backup):
            return "patched"
    return "original"


def main():
    parser = argparse.ArgumentParser(
        description="Patch ML frameworks to use gpushare remote GPU",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  gpushare-patch-frameworks                 # patch everything found
  gpushare-patch-frameworks --pytorch       # PyTorch only
  gpushare-patch-frameworks --undo          # restore all originals
  gpushare-patch-frameworks --status        # show what's patched
"""
    )
    parser.add_argument("--pytorch", action="store_true", help="Patch PyTorch only")
    parser.add_argument("--tensorflow", action="store_true", help="Patch TensorFlow only")
    parser.add_argument("--jax", action="store_true", help="Patch JAX only")
    parser.add_argument("--undo", action="store_true", help="Restore original libraries")
    parser.add_argument("--status", action="store_true", help="Show patch status")
    args = parser.parse_args()

    # Determine which frameworks to process
    frameworks = []
    if args.pytorch:
        frameworks = ["pytorch"]
    elif args.tensorflow:
        frameworks = ["tensorflow"]
    elif args.jax:
        frameworks = ["jax"]
    else:
        frameworks = ["pytorch", "tensorflow", "jax"]

    print(f"\n{BOLD}gpushare framework patcher{NC}")
    print(f"{'=' * 40}\n")

    # Status mode
    if args.status:
        for fw in frameworks:
            status = check_status(fw)
            libs = find_framework_cudart(fw)
            if status == "not_installed":
                print(f"  {fw:12s}  not installed")
            elif status == "patched":
                print(f"  {fw:12s}  {GREEN}patched{NC} ({len(libs)} libs)")
            else:
                print(f"  {fw:12s}  {YELLOW}original{NC} ({len(libs)} libs)")
        print()
        return

    # Find our library
    gpushare_lib = find_gpushare_lib()
    if not gpushare_lib and not args.undo:
        fail("Cannot find gpushare client library!")
        fail("Run the install script first, or build with: cmake --build build")
        sys.exit(1)

    if not args.undo:
        info(f"Using: {gpushare_lib}")
        print()

    total_patched = 0
    total_failed = 0

    for fw in frameworks:
        print(f"{BOLD}{fw}{NC}")
        libs = find_framework_cudart(fw)

        if not libs:
            print(f"  (not installed or no CUDA libs found)\n")
            continue

        for lib in libs:
            try:
                if patch_library(lib, gpushare_lib, undo=args.undo):
                    total_patched += 1
                else:
                    total_failed += 1
            except PermissionError:
                fail(f"Permission denied: {lib}")
                fail("Try running with sudo, or use a virtualenv")
                total_failed += 1
            except Exception as e:
                fail(f"Error patching {lib}: {e}")
                total_failed += 1
        print()

    # Summary
    action = "restored" if args.undo else "patched"
    if total_patched > 0:
        print(f"{GREEN}{BOLD}Done!{NC} {total_patched} libraries {action}.")
        if not args.undo:
            print(f"\nTest it:")
            if "pytorch" in frameworks:
                print(f'  python3 -c "import torch; print(torch.cuda.is_available())"')
            if "tensorflow" in frameworks:
                print(f'  python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices(\'GPU\'))"')
    elif total_failed > 0:
        fail(f"Failed to {action.rstrip('ed')} {total_failed} libraries.")
    else:
        info("No frameworks with bundled CUDA found. Nothing to patch.")
        info("If using pip-installed PyTorch: pip install torch")

    print()


if __name__ == "__main__":
    main()
