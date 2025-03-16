#!/usr/bin/env python3
"""
Comprehensive benchmark script to evaluate memory usage during weight loading,
with various strategies and options for testing both standard and extreme scenarios.

This script combines functionality from:
1. weight_loading_benchmark.py - Basic benchmark for measuring memory usage and loading time
2. weight_loading_extreme.py - Extreme benchmark for demonstrating memory savings with high redundancy

Features:
- Benchmark different weight loading strategies (naive, shared, lazy)
- Test with various GPU/CPU offloading percentages
- Configure redundancy factor to simulate models with weight patterns
- Detailed memory tracking (system, process, GPU)
- Save detailed benchmark results to file
"""

import os
import time
import torch
import argparse
import numpy as np
from datetime import datetime
import gc
import psutil

# Initialize CUDA
if torch.cuda.is_available():
    device = torch.device("cuda")
    # Initialize by calling device properties
    _ = torch.cuda.get_device_properties(0)
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    # Warm up CUDA
    test_tensor = torch.zeros(1, device='cuda')
    del test_tensor
    torch.cuda.empty_cache()
else:
    print("CUDA is not available. This benchmark requires a GPU.")
    exit(1)

def print_memory_usage(message=""):
    """Print detailed memory usage statistics"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # System memory
    system_memory = psutil.virtual_memory()
    system_memory_used_gb = system_memory.used / (1024**3)
    system_memory_total_gb = system_memory.total / (1024**3)
    system_memory_percent = system_memory.percent
    
    # Process memory
    process = psutil.Process(os.getpid())
    process_memory_gb = process.memory_info().rss / (1024**3)
    
    # GPU memory
    if torch.cuda.is_available():
        gpu_memory_allocated_gb = torch.cuda.memory_allocated() / (1024**3)
        gpu_memory_reserved_gb = torch.cuda.memory_reserved() / (1024**3)
        gpu_max_memory_allocated_gb = torch.cuda.max_memory_allocated() / (1024**3)
    else:
        gpu_memory_allocated_gb = 0
        gpu_memory_reserved_gb = 0
        gpu_max_memory_allocated_gb = 0
    
    print(f"\n----- Memory Usage at {timestamp} {message} -----")
    print(f"System Memory: {system_memory_used_gb:.2f} GB / {system_memory_total_gb:.2f} GB ({system_memory_percent:.1f}%)")
    print(f"Process Memory: {process_memory_gb:.2f} GB")
    print(f"GPU Memory Allocated: {gpu_memory_allocated_gb:.2f} GB")
    print(f"GPU Memory Reserved: {gpu_memory_reserved_gb:.2f} GB")
    print(f"GPU Max Memory Allocated: {gpu_max_memory_allocated_gb:.2f} GB")
    
    return {
        "timestamp": timestamp,
        "system_memory": system_memory_used_gb,
        "system_memory_total": system_memory_total_gb,
        "system_memory_percent": system_memory_percent,
        "process_memory": process_memory_gb,
        "gpu_memory_allocated": gpu_memory_allocated_gb,
        "gpu_memory_reserved": gpu_memory_reserved_gb,
        "gpu_max_memory_allocated": gpu_max_memory_allocated_gb
    }

class WeightLoadingBenchmark:
    """Benchmark class to test different weight loading strategies"""
    
    def __init__(self, layer_size=4096, num_layers=8, redundancy_factor=1, benchmark_mode="standard"):
        """
        Initialize the benchmark
        
        Args:
            layer_size: Size of square weight matrices (default: 4096)
            num_layers: Number of layers to simulate (default: 8)
            redundancy_factor: How many identical copies of each unique layer (default: 1)
            benchmark_mode: "standard" or "extreme" (default: "standard")
        """
        self.layer_size = layer_size
        self.num_layers = num_layers
        self.redundancy_factor = redundancy_factor
        self.benchmark_mode = benchmark_mode
        self.unique_layers = num_layers // max(1, redundancy_factor)
        self.results = []
        
        print(f"Benchmark settings: {layer_size}x{layer_size} matrices, {num_layers} layers")
        
        if redundancy_factor > 1:
            print(f"Redundancy factor: {redundancy_factor} (only {self.unique_layers} unique layers)")
            
            # Calculate theoretical memory requirements
            bytes_per_weight = 4  # float32
            matrix_bytes = layer_size * layer_size * bytes_per_weight
            
            # Calculate memory requirements with and without sharing
            naive_gpu_mem = matrix_bytes * num_layers / (1024**3)
            sharing_gpu_mem = matrix_bytes * self.unique_layers / (1024**3)
            
            print(f"Theoretical memory for each layer: {matrix_bytes / (1024**3):.2f} GB")
            print(f"Theoretical GPU memory (naive): {naive_gpu_mem:.2f} GB")
            print(f"Theoretical GPU memory (with sharing): {sharing_gpu_mem:.2f} GB")
            print(f"Theoretical memory savings: {naive_gpu_mem - sharing_gpu_mem:.2f} GB ({(naive_gpu_mem - sharing_gpu_mem)/naive_gpu_mem*100:.1f}%)")
        
        # Clean up before starting
        gc.collect()
        torch.cuda.empty_cache()
        print_memory_usage("Benchmark initialization")
    
    def create_weight_dict(self):
        """Create a dictionary of weights to simulate model weights with possible redundancy"""
        weights = {}
        
        if self.redundancy_factor > 1:
            # Create unique layers
            unique_weights = []
            unique_biases = []
            
            print(f"Creating {self.unique_layers} unique weight matrices...")
            for i in range(self.unique_layers):
                unique_weights.append(torch.randn(self.layer_size, self.layer_size))
                unique_biases.append(torch.randn(self.layer_size))
            
            # Assign layers (with redundancy)
            for i in range(self.num_layers):
                # Use modulo to repeat weights for redundancy
                source_idx = i % self.unique_layers
                weights[f"layer.{i}.weight"] = unique_weights[source_idx]
                weights[f"layer.{i}.bias"] = unique_biases[source_idx]
        else:
            # Standard mode - create unique weights for each layer
            for i in range(self.num_layers):
                weights[f"layer.{i}.weight"] = torch.randn(self.layer_size, self.layer_size)
                weights[f"layer.{i}.bias"] = torch.randn(self.layer_size)
                
        return weights
    
    def benchmark_naive_loading(self, gpu_percent=100):
        """
        Benchmark naive loading where weights are directly loaded to the specified device
        without any memory optimization
        """
        print(f"\n===== Benchmarking Naive Loading (GPU: {gpu_percent}%) =====")
        results = {"method": "naive", "gpu_percent": gpu_percent}
        
        gc.collect()
        torch.cuda.empty_cache()
        results["initial_memory"] = print_memory_usage("Before weight creation")
        
        # Create weight dictionary (simulating model weights)
        weights = self.create_weight_dict()
        results["after_weight_creation"] = print_memory_usage("After weight creation (CPU)")
        
        # Calculate how many layers should be on GPU vs CPU
        gpu_layers = int(self.num_layers * gpu_percent / 100)
        
        # Load weights to appropriate devices
        loaded_weights = {}
        start_time = time.time()
        
        for i in range(self.num_layers):
            device = 'cuda' if i < gpu_layers else 'cpu'
            loaded_weights[f"layer.{i}.weight"] = weights[f"layer.{i}.weight"].to(device)
            loaded_weights[f"layer.{i}.bias"] = weights[f"layer.{i}.bias"].to(device)
        
        # Force completion of CUDA operations
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            
        load_time = time.time() - start_time
        results["after_loading"] = print_memory_usage(f"After loading (naive method, {load_time:.4f}s)")
        
        # Delete weights and measure memory again
        del weights
        del loaded_weights
        gc.collect()
        torch.cuda.empty_cache()
        
        results["after_cleanup"] = print_memory_usage("After cleanup")
        results["load_time"] = load_time
        
        self.results.append(results)
        return results
    
    def benchmark_shared_weights(self, gpu_percent=100):
        """
        Benchmark loading with weight sharing where identical weights
        reference the same memory
        """
        print(f"\n===== Benchmarking Shared Weight Loading (GPU: {gpu_percent}%) =====")
        results = {"method": "shared", "gpu_percent": gpu_percent}
        
        gc.collect()
        torch.cuda.empty_cache()
        results["initial_memory"] = print_memory_usage("Before weight creation")
        
        # Create weight dictionary (simulating model weights)
        weights = self.create_weight_dict()
        results["after_weight_creation"] = print_memory_usage("After weight creation (CPU)")
        
        # Calculate how many layers should be on GPU vs CPU
        gpu_layers = int(self.num_layers * gpu_percent / 100)
        
        # Create a dictionary to track weight data pointers
        weight_cache = {}
        loaded_weights = {}
        start_time = time.time()
        
        # Track reuse statistics
        reused_weights = 0
        total_weights = 0
        
        for i in range(self.num_layers):
            device = 'cuda' if i < gpu_layers else 'cpu'
            
            # The key idea: Check if we've already loaded this weight to this device
            weight_key = f"layer.{i}.weight"
            bias_key = f"layer.{i}.bias"
            
            # Get the CPU tensor
            cpu_weight = weights[weight_key]
            cpu_bias = weights[bias_key]
            
            # Generate cache key based on tensor data pointer and target device
            w_cache_key = (cpu_weight.data_ptr(), device)
            b_cache_key = (cpu_bias.data_ptr(), device)
            
            total_weights += 2  # Count both weight and bias
            
            # Check if we've already loaded this exact tensor to this device
            if w_cache_key in weight_cache:
                loaded_weights[weight_key] = weight_cache[w_cache_key]
                reused_weights += 1
                print(f"Reusing weight for layer {i}")
            else:
                loaded_weights[weight_key] = cpu_weight.to(device)
                weight_cache[w_cache_key] = loaded_weights[weight_key]
            
            if b_cache_key in weight_cache:
                loaded_weights[bias_key] = weight_cache[b_cache_key]
                reused_weights += 1
                print(f"Reusing bias for layer {i}")
            else:
                loaded_weights[bias_key] = cpu_bias.to(device)
                weight_cache[b_cache_key] = loaded_weights[bias_key]
        
        # Force completion of CUDA operations
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            
        load_time = time.time() - start_time
        results["after_loading"] = print_memory_usage(f"After loading (shared method, {load_time:.4f}s)")
        
        # Report reuse statistics
        reuse_percentage = (reused_weights / total_weights) * 100
        print(f"Weight sharing statistics: {reused_weights}/{total_weights} tensors reused ({reuse_percentage:.1f}%)")
        
        # Delete weights and measure memory again
        del weights
        del loaded_weights
        del weight_cache
        gc.collect()
        torch.cuda.empty_cache()
        
        results["after_cleanup"] = print_memory_usage("After cleanup")
        results["load_time"] = load_time
        results["reuse_percentage"] = reuse_percentage
        
        self.results.append(results)
        return results
    
    def benchmark_lazy_loading(self, gpu_percent=100):
        """
        Benchmark lazy loading where weights are loaded on-demand and
        offloaded when not needed
        """
        print(f"\n===== Benchmarking Lazy Loading (GPU: {gpu_percent}%) =====")
        results = {"method": "lazy", "gpu_percent": gpu_percent}
        
        gc.collect()
        torch.cuda.empty_cache()
        results["initial_memory"] = print_memory_usage("Before weight creation")
        
        # Create weight dictionary (simulating model weights)
        weights = self.create_weight_dict()
        results["after_weight_creation"] = print_memory_usage("After weight creation (CPU)")
        
        # Calculate how many layers should be on GPU vs CPU
        gpu_layers = int(self.num_layers * gpu_percent / 100)
        
        # Cache for loaded weights
        weight_cache = {}
        
        # Simulate a forward pass where weights are loaded on-demand
        start_time = time.time()
        
        # Create small input tensor to simulate forward pass
        input_tensor = torch.randn(1, self.layer_size, device='cuda')
        
        # Simulate forward pass through layers
        for i in range(self.num_layers):
            device = 'cuda' if i < gpu_layers else 'cpu'
            
            # Get parameters for this layer
            weight_key = f"layer.{i}.weight"
            bias_key = f"layer.{i}.bias"
            
            # Load weights on demand
            if i > 0:
                # Offload previous layer weights if they're not needed anymore
                prev_weight_key = f"layer.{i-1}.weight"
                prev_bias_key = f"layer.{i-1}.bias"
                
                if prev_weight_key in weight_cache:
                    del weight_cache[prev_weight_key]
                if prev_bias_key in weight_cache:
                    del weight_cache[prev_bias_key]
                
                # Force garbage collection to clean up offloaded weights
                gc.collect()
                torch.cuda.empty_cache()
            
            # Load current layer weights
            weight = weights[weight_key].to(device)
            bias = weights[bias_key].to(device)
            
            # Store in cache
            weight_cache[weight_key] = weight
            weight_cache[bias_key] = bias
            
            # If on GPU, do a mock computation to simulate usage
            if device == 'cuda':
                # Simulate a linear computation
                output = torch.matmul(input_tensor, weight.t()) + bias
                input_tensor = output
        
        load_time = time.time() - start_time
        results["after_loading"] = print_memory_usage(f"After loading (lazy method, {load_time:.4f}s)")
        
        # Clean up
        del weights
        del weight_cache
        gc.collect()
        torch.cuda.empty_cache()
        
        results["after_cleanup"] = print_memory_usage("After cleanup")
        results["load_time"] = load_time
        
        self.results.append(results)
        return results
    
    def run_benchmarks(self, gpu_percentages=[0, 50, 100], methods=None):
        """
        Run selected benchmark methods with different GPU percentages
        
        Args:
            gpu_percentages: List of GPU percentages to test (default: [0, 50, 100])
            methods: List of methods to benchmark (default: all available methods)
        """
        if methods is None:
            # Default to all methods
            methods = ["naive", "shared"]
            
            # Add lazy loading only for standard benchmark
            if self.benchmark_mode == "standard":
                methods.append("lazy")
        
        for gpu_percent in gpu_percentages:
            for method in methods:
                if method == "naive":
                    self.benchmark_naive_loading(gpu_percent)
                elif method == "shared":
                    self.benchmark_shared_weights(gpu_percent)
                elif method == "lazy":
                    self.benchmark_lazy_loading(gpu_percent)
    
    def print_summary(self):
        """Print a summary of benchmark results"""
        print("\n===== Benchmark Summary =====")
        print(f"{'Method':<10} {'GPU %':<6} {'Load Time (s)':<15} {'Peak Proc Mem (GB)':<20} {'Peak GPU Mem (GB)':<20}")
        
        for result in self.results:
            method = result["method"]
            gpu_percent = result["gpu_percent"]
            load_time = result["load_time"]
            
            # Find peak memory
            memory_readings = [
                result["initial_memory"],
                result["after_weight_creation"],
                result["after_loading"],
                result["after_cleanup"]
            ]
            
            peak_process_memory = max(m["process_memory"] for m in memory_readings)
            peak_gpu_memory = max(m["gpu_memory_allocated"] for m in memory_readings)
            
            print(f"{method:<10} {gpu_percent:<6} {load_time:<15.4f} {peak_process_memory:<20.2f} {peak_gpu_memory:<20.2f}")
        
        # Calculate memory savings (for 100% GPU)
        naive_results = [r for r in self.results if r["method"] == "naive" and r["gpu_percent"] == 100]
        shared_results = [r for r in self.results if r["method"] == "shared" and r["gpu_percent"] == 100]
        
        if naive_results and shared_results:
            naive_peak = max(m["gpu_memory_allocated"] for m in [naive_results[0]["after_loading"]])
            shared_peak = max(m["gpu_memory_allocated"] for m in [shared_results[0]["after_loading"]])
            
            memory_saved = naive_peak - shared_peak
            memory_saved_percent = (memory_saved / naive_peak) * 100 if naive_peak > 0 else 0
            
            print(f"\nGPU Memory saved with shared weights: {memory_saved:.2f} GB ({memory_saved_percent:.1f}%)")
            
            if "reuse_percentage" in shared_results[0]:
                print(f"Weight tensors reused: {shared_results[0]['reuse_percentage']:.1f}%")
    
    def save_results(self, filename=None):
        """Save benchmark results to a file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            benchmark_type = "extreme" if self.redundancy_factor > 1 else "standard"
            filename = f"weight_loading_{benchmark_type}_benchmark_{timestamp}.txt"
        
        with open(filename, 'w') as f:
            f.write(f"===== Weight Loading Benchmark Results ({self.benchmark_mode.upper()}) =====\n")
            f.write(f"Layer size: {self.layer_size}x{self.layer_size}\n")
            f.write(f"Number of layers: {self.num_layers}\n")
            
            if self.redundancy_factor > 1:
                f.write(f"Redundancy factor: {self.redundancy_factor} ({self.unique_layers} unique layers)\n")
            
            f.write(f"\n{'Method':<10} {'GPU %':<6} {'Load Time (s)':<15} {'Peak Proc Mem (GB)':<20} {'Peak GPU Mem (GB)':<20}\n")
            
            for result in self.results:
                method = result["method"]
                gpu_percent = result["gpu_percent"]
                load_time = result["load_time"]
                
                # Find peak memory
                memory_readings = [
                    result["initial_memory"],
                    result["after_weight_creation"],
                    result["after_loading"],
                    result["after_cleanup"]
                ]
                
                peak_process_memory = max(m["process_memory"] for m in memory_readings)
                peak_gpu_memory = max(m["gpu_memory_allocated"] for m in memory_readings)
                
                f.write(f"{method:<10} {gpu_percent:<6} {load_time:<15.4f} {peak_process_memory:<20.2f} {peak_gpu_memory:<20.2f}\n")
            
            # Calculate memory savings (for 100% GPU)
            naive_results = [r for r in self.results if r["method"] == "naive" and r["gpu_percent"] == 100]
            shared_results = [r for r in self.results if r["method"] == "shared" and r["gpu_percent"] == 100]
            
            if naive_results and shared_results:
                naive_peak = max(m["gpu_memory_allocated"] for m in [naive_results[0]["after_loading"]])
                shared_peak = max(m["gpu_memory_allocated"] for m in [shared_results[0]["after_loading"]])
                
                memory_saved = naive_peak - shared_peak
                memory_saved_percent = (memory_saved / naive_peak) * 100 if naive_peak > 0 else 0
                
                f.write(f"\nGPU Memory saved with shared weights: {memory_saved:.2f} GB ({memory_saved_percent:.1f}%)\n")
                
                if "reuse_percentage" in shared_results[0]:
                    f.write(f"Weight tensors reused: {shared_results[0]['reuse_percentage']:.1f}%\n")
            
            # Detailed results
            f.write("\n\n===== Detailed Results =====\n")
            for result in self.results:
                f.write(f"\n--- {result['method']} method, {result['gpu_percent']}% GPU ---\n")
                f.write(f"Load time: {result['load_time']:.4f} seconds\n")
                
                for stage in ["initial_memory", "after_weight_creation", "after_loading", "after_cleanup"]:
                    memory = result[stage]
                    f.write(f"\n{stage}:\n")
                    f.write(f"  System Memory: {memory['system_memory']:.2f} GB / {memory['system_memory_total']:.2f} GB ({memory['system_memory_percent']:.1f}%)\n")
                    f.write(f"  Process Memory: {memory['process_memory']:.2f} GB\n")
                    f.write(f"  GPU Memory Allocated: {memory['gpu_memory_allocated']:.2f} GB\n")
                    f.write(f"  GPU Memory Reserved: {memory['gpu_memory_reserved']:.2f} GB\n")
                    f.write(f"  GPU Max Memory Allocated: {memory['gpu_max_memory_allocated']:.2f} GB\n")
        
        print(f"\nResults saved to {filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Comprehensive benchmark for weight loading strategies")
    parser.add_argument("--mode", type=str, choices=["standard", "extreme"], default="standard",
                      help="Benchmark mode: standard or extreme (default: standard)")
    parser.add_argument("--gpu-percentages", type=str, default="0,50,100",
                      help="Comma-separated list of GPU percentages to test")
    parser.add_argument("--layer-size", type=int, default=4096,
                      help="Size of square weight matrices (default is 4096)")
    parser.add_argument("--num-layers", type=int, default=8,
                      help="Number of layers to simulate (default is 8)")
    parser.add_argument("--redundancy-factor", type=int, default=1,
                      help="How many identical copies of each unique layer (default is 1 for standard, 2 for extreme)")
    parser.add_argument("--methods", type=str, default=None,
                      help="Comma-separated list of methods to benchmark (naive,shared,lazy)")
    parser.add_argument("--output", type=str, default=None,
                      help="Output file for results (default is timestamped file)")
    
    args = parser.parse_args()
    
    # Parse GPU percentages
    gpu_percentages = [int(p) for p in args.gpu_percentages.split(',')]
    
    # Parse methods if provided
    methods = None
    if args.methods:
        methods = args.methods.split(',')
    
    # Set default redundancy factor for extreme mode if not specified
    redundancy_factor = args.redundancy_factor
    if args.mode == "extreme" and redundancy_factor == 1:
        redundancy_factor = 2
    
    # Run benchmarks
    benchmark = WeightLoadingBenchmark(
        layer_size=args.layer_size,
        num_layers=args.num_layers,
        redundancy_factor=redundancy_factor,
        benchmark_mode=args.mode
    )
    
    benchmark.run_benchmarks(gpu_percentages, methods)
    benchmark.print_summary()
    benchmark.save_results(args.output) 