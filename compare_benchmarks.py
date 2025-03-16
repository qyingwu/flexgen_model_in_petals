#!/usr/bin/env python3
"""
Script to compare our custom CPU/GPU offloading with FlexGen's approach
"""

import os
import time
import torch
import argparse
import numpy as np
from datetime import datetime

# Initialize CUDA and print debug info
if torch.cuda.is_available():
    # There's no torch.cuda.init() - we'll initialize by calling device properties
    _ = torch.cuda.get_device_properties(0)
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
    print(f"CUDA current device: {torch.cuda.current_device()}")
    print(f"CUDA memory allocated: {torch.cuda.memory_allocated() / (1024**3):.2f} GB")
    print(f"CUDA memory reserved: {torch.cuda.memory_reserved() / (1024**3):.2f} GB")
    
    # Create and delete a small test tensor to warm up CUDA
    print("Warming up CUDA with a test tensor...")
    test_tensor = torch.zeros(1, device='cuda')
    del test_tensor
    torch.cuda.empty_cache()
else:
    print("CUDA is not available. This benchmark requires a GPU.")
    exit(1)

def print_memory_usage(message=""):
    """Print current memory usage"""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        # System memory
        system_memory = psutil.virtual_memory()
        system_memory_used_gb = system_memory.used / (1024**3)
        system_memory_total_gb = system_memory.total / (1024**3)
        system_memory_percent = system_memory.percent
        
        # Process memory
        process_memory_gb = process.memory_info().rss / (1024**3)
        
        # GPU memory
        gpu_memory_used_gb = 0
        gpu_memory_total_gb = 0
        gpu_name = "N/A"
        
        try:
            if torch.cuda.is_available():
                device = torch.cuda.current_device()
                gpu_name = torch.cuda.get_device_name(device)
                gpu_memory_total_gb = torch.cuda.get_device_properties(device).total_memory / (1024**3)
                gpu_memory_used_gb = (torch.cuda.memory_allocated() + torch.cuda.memory_reserved()) / (1024**3)
        except Exception as e:
            print(f"Error getting CUDA memory: {e}")
            
        # PyTorch GPU memory
        torch_gpu_allocated_gb = 0
        torch_gpu_max_allocated_gb = 0
        
        if torch.cuda.is_available():
            torch_gpu_allocated_gb = torch.cuda.memory_allocated() / (1024**3)
            torch_gpu_max_allocated_gb = torch.cuda.max_memory_allocated() / (1024**3)
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n----- Memory Usage at {timestamp} {message} -----")
        print(f"System Memory: {system_memory_used_gb:.2f} GB / {system_memory_total_gb:.2f} GB ({system_memory_percent:.1f}%)")
        print(f"Process Memory: {process_memory_gb:.2f} GB")
        print(f"GPU Memory: {gpu_memory_used_gb:.2f} GB used / {gpu_memory_total_gb:.2f} GB total")
        if torch.cuda.is_available():
            print(f"PyTorch GPU Allocated: {torch_gpu_allocated_gb:.2f} GB")
            print(f"PyTorch GPU Max Allocated: {torch_gpu_max_allocated_gb:.2f} GB")
            
    except Exception as e:
        print(f"Error monitoring memory: {e}")

class CustomOffloadingBenchmark:
    """Benchmark for our custom CPU/GPU offloading approach"""
    
    def __init__(self, size=4096, num_layers=8, num_iterations=10, warmup=2):
        self.size = size
        self.num_layers = num_layers
        self.num_iterations = num_iterations
        self.warmup = warmup
        
    def benchmark(self, gpu_percent):
        """Run benchmark with specified GPU percentage"""
        print(f"\n===== Custom Offloading Benchmark with {gpu_percent}% on GPU =====")
        print_memory_usage("Before benchmark")
        
        gpu_layers = int(self.num_layers * gpu_percent / 100)
        cpu_layers = self.num_layers - gpu_layers
        
        # Create input tensor on GPU
        x = torch.randn(1, self.size, device='cuda')
        
        # Create weight tensors with appropriate distribution
        weights_gpu = [torch.randn(self.size, self.size, device='cuda') for _ in range(gpu_layers)]
        weights_cpu = [torch.randn(self.size, self.size, device='cpu') for _ in range(cpu_layers)]
        
        print_memory_usage("After tensor creation")
        
        # Warmup
        print(f"Running {self.warmup} warmup iterations...")
        for _ in range(self.warmup):
            # Forward pass
            for w in weights_gpu:
                x = torch.matmul(x, w)
                x = torch.relu(x)
                
            for w in weights_cpu:
                x_cpu = x.to('cpu')
                x_cpu = torch.matmul(x_cpu, w)
                x_cpu = torch.relu(x_cpu)
                x = x_cpu.to('cuda')
                
        torch.cuda.synchronize()
        
        # Benchmark
        print(f"Running {self.num_iterations} benchmark iterations...")
        times = []
        for i in range(self.num_iterations):
            start_time = time.time()
            
            # Forward pass
            for w in weights_gpu:
                x = torch.matmul(x, w)
                x = torch.relu(x)
                
            for w in weights_cpu:
                x_cpu = x.to('cpu')
                x_cpu = torch.matmul(x_cpu, w)
                x_cpu = torch.relu(x_cpu)
                x = x_cpu.to('cuda')
                
            torch.cuda.synchronize()
            end_time = time.time()
            times.append(end_time - start_time)
            print(f"Iteration {i+1}/{self.num_iterations}: {times[-1]:.4f} seconds")
        
        mean_time = np.mean(times)
        std_time = np.std(times)
        print(f"\nCustom Offloading Results ({gpu_percent}% GPU):")
        print(f"  Mean execution time: {mean_time:.4f} seconds")
        print(f"  Std deviation: {std_time:.4f} seconds")
        
        # Cleanup
        del x, weights_gpu, weights_cpu
        torch.cuda.empty_cache()
        print_memory_usage("After cleanup")
        
        return mean_time, std_time

class FlexGenBenchmark:
    """Simulate FlexGen's offloading approach"""
    
    def __init__(self, size=4096, num_layers=8, num_iterations=10, warmup=2):
        self.size = size
        self.num_layers = num_layers
        self.num_iterations = num_iterations
        self.warmup = warmup
        
    def benchmark(self, gpu_percent):
        """Run benchmark with FlexGen-style offloading at specified GPU percentage"""
        print(f"\n===== FlexGen-style Benchmark with {gpu_percent}% on GPU =====")
        print_memory_usage("Before benchmark")
        
        # FlexGen keeps all weights on CPU but computes on GPU
        # Create input tensor on GPU
        x = torch.randn(1, self.size, device='cuda')
        
        # Create all weight tensors on CPU
        weights = [torch.randn(self.size, self.size, device='cpu') for _ in range(self.num_layers)]
        
        print_memory_usage("After tensor creation")
        
        # Precompute which layers should use GPU based on percentage
        gpu_layers = int(self.num_layers * gpu_percent / 100)
        gpu_layer_indices = set(range(self.num_layers)[:gpu_layers])
        
        # Warmup
        print(f"Running {self.warmup} warmup iterations...")
        for _ in range(self.warmup):
            # Forward pass
            for i, w in enumerate(weights):
                # Move weights to GPU for computation then back to CPU
                w_gpu = w.to('cuda')
                x = torch.matmul(x, w_gpu)
                x = torch.relu(x)
                
                # FlexGen offloads activations for some layers based on memory budget
                if i not in gpu_layer_indices:
                    x = x.to('cpu').to('cuda')
                
        torch.cuda.synchronize()
        
        # Benchmark
        print(f"Running {self.num_iterations} benchmark iterations...")
        times = []
        for i in range(self.num_iterations):
            start_time = time.time()
            
            # Forward pass
            for j, w in enumerate(weights):
                # Move weights to GPU for computation then back to CPU
                w_gpu = w.to('cuda')
                x = torch.matmul(x, w_gpu)
                x = torch.relu(x)
                
                # FlexGen offloads activations for some layers based on memory budget
                if j not in gpu_layer_indices:
                    x = x.to('cpu').to('cuda')
                    
            torch.cuda.synchronize()
            end_time = time.time()
            times.append(end_time - start_time)
            print(f"Iteration {i+1}/{self.num_iterations}: {times[-1]:.4f} seconds")
        
        mean_time = np.mean(times)
        std_time = np.std(times)
        print(f"\nFlexGen-style Results ({gpu_percent}% GPU):")
        print(f"  Mean execution time: {mean_time:.4f} seconds")
        print(f"  Std deviation: {std_time:.4f} seconds")
        
        # Cleanup
        del x, weights
        torch.cuda.empty_cache()
        print_memory_usage("After cleanup")
        
        return mean_time, std_time

def run_comparison(size, num_layers, num_iterations, warmup, gpu_percentages, output_file=None):
    """Run both benchmarks and compare results"""
    print("\n===== Running Benchmark Comparison =====")
    print(f"Matrix size: {size}x{size}")
    print(f"Number of layers: {num_layers}")
    print(f"Iterations: {num_iterations} (with {warmup} warmup)")
    
    custom_results = {}
    flexgen_results = {}
    
    for gpu_percent in gpu_percentages:
        # Run custom offloading benchmark
        custom_mean, custom_std = CustomOffloadingBenchmark(
            size=size, 
            num_layers=num_layers, 
            num_iterations=num_iterations, 
            warmup=warmup
        ).benchmark(gpu_percent)
        custom_results[gpu_percent] = (custom_mean, custom_std)
        
        # Small delay between tests
        time.sleep(2)
        
        # Run FlexGen-style benchmark
        flexgen_mean, flexgen_std = FlexGenBenchmark(
            size=size, 
            num_layers=num_layers, 
            num_iterations=num_iterations, 
            warmup=warmup
        ).benchmark(gpu_percent)
        flexgen_results[gpu_percent] = (flexgen_mean, flexgen_std)
        
        # Small delay between tests
        time.sleep(2)
    
    # Print comparative results
    print("\n===== Benchmark Comparison Summary =====")
    print(f"{'GPU %':<8} {'Custom Mean (s)':<15} {'FlexGen Mean (s)':<15} {'Speedup':<10}")
    
    for gpu_percent in sorted(gpu_percentages):
        custom_mean, _ = custom_results[gpu_percent]
        flexgen_mean, _ = flexgen_results[gpu_percent]
        speedup = flexgen_mean / custom_mean  # Higher means custom is faster
        
        print(f"{gpu_percent:<8} {custom_mean:<15.4f} {flexgen_mean:<15.4f} {speedup:<10.2f}x")
    
    # Save results if requested
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"comparison_results_{timestamp}.txt"
    
    with open(output_file, 'w') as f:
        f.write("===== Benchmark Comparison Results =====\n")
        f.write(f"Matrix size: {size}x{size}\n")
        f.write(f"Number of layers: {num_layers}\n")
        f.write(f"Iterations: {num_iterations} (with {warmup} warmup)\n\n")
        
        f.write(f"{'GPU %':<8} {'Custom Mean (s)':<15} {'Custom Std':<10} {'FlexGen Mean (s)':<15} {'FlexGen Std':<10} {'Speedup':<10}\n")
        
        for gpu_percent in sorted(gpu_percentages):
            custom_mean, custom_std = custom_results[gpu_percent]
            flexgen_mean, flexgen_std = flexgen_results[gpu_percent]
            speedup = flexgen_mean / custom_mean
            
            f.write(f"{gpu_percent:<8} {custom_mean:<15.4f} {custom_std:<10.4f} {flexgen_mean:<15.4f} {flexgen_std:<10.4f} {speedup:<10.2f}x\n")
    
    print(f"\nComparison results saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare offloading strategies")
    parser.add_argument("--gpu-percentages", type=str, default="25,50,75,100",
                      help="Comma-separated list of GPU percentages to test")
    parser.add_argument("--size", type=int, default=4096,
                      help="Size of the square matrix (default is 4096)")
    parser.add_argument("--layers", type=int, default=8,
                      help="Number of layers to simulate (default is 8)")
    parser.add_argument("--iterations", type=int, default=5,
                      help="Number of benchmark iterations (default is 5)")
    parser.add_argument("--warmup", type=int, default=2,
                      help="Number of warmup iterations (default is 2)")
    parser.add_argument("--output", type=str, default=None,
                      help="Output file for results (default is timestamped file)")
    
    args = parser.parse_args()
    
    # Parse GPU percentages
    gpu_percentages = [int(p) for p in args.gpu_percentages.split(',')]
    
    # Run comparison
    run_comparison(
        size=args.size,
        num_layers=args.layers,
        num_iterations=args.iterations,
        warmup=args.warmup,
        gpu_percentages=gpu_percentages,
        output_file=args.output
    ) 