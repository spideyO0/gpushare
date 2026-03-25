#!/usr/bin/env python3
"""
Simple PyTorch test for gpushare
Tests basic tensor operations on the remote GPU
"""
import torch

def main():
    print("=== PyTorch + gpushare Test ===\n")
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("❌ CUDA is not available")
        return 1
    
    # Get device count
    device_count = torch.cuda.device_count()
    print(f"✓ CUDA available, found {device_count} device(s)")
    
    for i in range(device_count):
        name = torch.cuda.get_device_name(i)
        props = torch.cuda.get_device_properties(i)
        print(f"\nDevice {i}: {name}")
        print(f"  Total Memory: {props.total_memory / 1024**3:.2f} GB")
        print(f"  Compute Capability: {props.major}.{props.minor}")
        print(f"  Multi-processor count: {props.multi_processor_count}")
    
    # Test simple tensor operation
    print("\nTesting tensor operations...")
    
    # Create a tensor on the default GPU (which should be the remote GPU)
    device = torch.device('cuda:0')
    print(f"Default device: {device}")
    
    # Create a simple tensor
    x = torch.randn(1000, 1000, device=device)
    print(f"✓ Created tensor: {x.shape}, device: {x.device}")
    
    # Simple computation
    y = x * 2
    print(f"✓ Element-wise multiply: {y.shape}, device: {y.device}")
    
    # Matrix multiplication
    z=torch.matmul(x, y.T) if y.shape == x.shape else torch.matmul(x, y)
    print(f"✓ Matrix multiply: {z.shape}, device: {z.device}")
    
    # Test synchronization
    torch.cuda.synchronize()
    print("✓ Synchronized")
    
    # Test memory info
    print(f"\nGPU Memory: {torch.cuda.memory_allocated() / 1024**2:.2f} MB allocated")
    print(f"GPU Memory: {torch.cuda.memory_reserved() / 1024**2:.2f} MB reserved")
    
    print("\n=== All tests passed ===")
    return 0

if __name__ == '__main__':
    import sys
    sys.exit(main())
