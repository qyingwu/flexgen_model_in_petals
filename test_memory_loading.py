import torch
import psutil
import os
from pynvml import *
import time

def print_memory_usage(message):
    """Print both CPU and GPU memory usage"""
    # CPU Memory
    process = psutil.Process(os.getpid())
    cpu_mem = process.memory_info().rss / (1024 * 1024 * 1024)  # Convert to GB
    
    # GPU Memory
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    gpu_mem = info.used / (1024 * 1024 * 1024)  # Convert to GB
    
    print(f"\n{message}")
    print(f"CPU Memory Used: {cpu_mem:.2f} GB")
    print(f"GPU Memory Used: {gpu_mem:.2f} GB")
    if torch.cuda.is_available():
        print(f"Torch GPU Memory Allocated: {torch.cuda.memory_allocated() / (1024 * 1024 * 1024):.2f} GB")
        print(f"Torch GPU Memory Reserved: {torch.cuda.memory_reserved() / (1024 * 1024 * 1024):.2f} GB")

def test_block_loading():
    """Test memory usage during block loading"""
    from petals.server.server import Server
    
    print_memory_usage("Initial Memory State")
    
    # Initialize server with minimal configuration
    server = Server(
        initial_peers=["http://localhost:8080"],  # Example peer
        converted_model_name_or_path="meta-llama/Llama-2-7b-hf",  # Use your model
        dht_prefix="test",
        num_blocks=1,  # Test with single block
        throughput="auto",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    print_memory_usage("After Server Initialization")
    time.sleep(2)  # Wait to stabilize
    
    # Test loading first block
    print("\nStarting block loading test...")
    
    try:
        # Create ModuleContainer to trigger block loading
        server.module_container = server.ModuleContainer.create(
            dht=server.dht,
            dht_prefix=server.dht_prefix,
            converted_model_name_or_path=server.converted_model_name_or_path,
            block_config=server.block_config,
            env=server.env,
            policy=server.policy,
            weight_home=server.weight_home,
            path=server.path,
            attn_cache_bytes=server.attn_cache_bytes,
            server_info=server.server_info,
            model_info=server.model_info,
            block_indices=[0],  # Test with first block
            num_handlers=1,
            min_batch_size=1,
            max_batch_size=1,
            max_chunk_size_bytes=256 * 1024 * 1024,
            max_alloc_timeout=600,
            torch_dtype=server.torch_dtype,
            cache_dir=server.cache_dir,
            max_disk_space=server.max_disk_space,
            device=server.device,
            compression=server.compression,
            start=False  # Don't actually start serving
        )
        
        print_memory_usage("After Block Loading")
        
    except Exception as e:
        print(f"Error during testing: {str(e)}")
    finally:
        # Cleanup
        if hasattr(server, 'module_container') and server.module_container:
            server.module_container.shutdown()
        server.shutdown()
        
    print_memory_usage("After Cleanup")

if __name__ == "__main__":
    test_block_loading() 