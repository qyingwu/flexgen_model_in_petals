#!/usr/bin/env python3
"""
Script to check memory usage of running Petals processes
"""

import os
import sys
import time
import psutil
import argparse
from datetime import datetime

# Try to import GPU monitoring libraries
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available")

try:
    from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlDeviceGetName, nvmlShutdown
    NVML_AVAILABLE = True
    nvmlInit()
except ImportError:
    NVML_AVAILABLE = False
    print("NVML not available for GPU monitoring")
except Exception as e:
    NVML_AVAILABLE = False
    print(f"Warning: Failed to initialize NVML: {e}")

def print_header(title):
    """Print a section header"""
    print("\n" + "="*80)
    print(f" {title} ")
    print("="*80 + "\n")

def get_gpu_memory_info():
    """Get GPU memory using NVML if available"""
    if not NVML_AVAILABLE:
        return "Unknown", 0, 0, 0
    
    try:
        handle = nvmlDeviceGetHandleByIndex(0)
        info = nvmlDeviceGetMemoryInfo(handle)
        device_name = nvmlDeviceGetName(handle)
        total = info.total / (1024**3)  # Convert to GB
        used = info.used / (1024**3)
        free = info.free / (1024**3)
        return device_name, total, used, free
    except Exception as e:
        print(f"Error getting GPU info: {e}")
        return "Error", 0, 0, 0

def find_petals_processes():
    """Find all running Petals server processes"""
    petals_processes = []
    for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'memory_info']):
        try:
            if proc.info['cmdline'] and any('petals' in cmd.lower() for cmd in proc.info['cmdline'] if isinstance(cmd, str)):
                petals_processes.append(proc)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    return petals_processes

def categorize_petals_processes(processes):
    """Categorize Petals processes into servers and DHT nodes"""
    servers = []
    dht_nodes = []
    
    for proc in processes:
        try:
            cmdline = proc.cmdline()
            if any('run_server' in cmd.lower() for cmd in cmdline if isinstance(cmd, str)):
                servers.append(proc)
            elif any('run_dht' in cmd.lower() for cmd in cmdline if isinstance(cmd, str)):
                dht_nodes.append(proc)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    
    return servers, dht_nodes

def monitor_petals_processes(interval=5, duration=60):
    """Monitor memory usage of Petals processes over time"""
    start_time = time.time()
    end_time = start_time + duration
    
    print_header("Petals Memory Monitoring")
    print(f"Monitoring Petals processes every {interval} seconds for {duration} seconds")
    
    # Get initial GPU memory info as baseline
    if NVML_AVAILABLE:
        _, _, initial_gpu_used, _ = get_gpu_memory_info()
    else:
        initial_gpu_used = 0
    
    iteration = 1
    try:
        while time.time() < end_time:
            print(f"\n--- Iteration {iteration} at {datetime.now().strftime('%H:%M:%S')} ---")
            iteration += 1
            
            # Get overall GPU usage
            if NVML_AVAILABLE:
                device_name, total_gpu, used_gpu, free_gpu = get_gpu_memory_info()
                print(f"GPU ({device_name}): {used_gpu:.2f}GB used / {total_gpu:.2f}GB total ({used_gpu/total_gpu*100:.1f}%)")
                print(f"GPU memory increase from baseline: {used_gpu - initial_gpu_used:.2f}GB")
            
            # Find all Petals processes
            all_petals_processes = find_petals_processes()
            if not all_petals_processes:
                print("No Petals processes found running")
                if time.time() < end_time:
                    time.sleep(interval)
                continue
                
            # Categorize processes
            server_processes, dht_processes = categorize_petals_processes(all_petals_processes)
            
            # Report on DHT nodes
            if dht_processes:
                print(f"\nFound {len(dht_processes)} Petals DHT nodes:")
                total_dht_mem = sum(p.info['memory_info'].rss for p in dht_processes) / (1024**3)
                print(f"Total memory used by all DHT nodes: {total_dht_mem:.2f}GB")
                
                for i, proc in enumerate(dht_processes, 1):
                    try:
                        pid = proc.info['pid']
                        cmd = ' '.join(str(c) for c in proc.info['cmdline'][:3]) + '...'
                        cpu_mem = proc.info['memory_info'].rss / (1024**3)
                        print(f"{i}. PID {pid}: {cmd}")
                        print(f"   Memory: {cpu_mem:.2f}GB")
                        print(f"   Running time: {(time.time() - proc.create_time()) / 60:.1f} minutes")
                    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess) as e:
                        print(f"{i}. Process info unavailable: {e}")
            
            # Report on server processes
            if not server_processes:
                print("\nNo Petals server processes found running")
            else:
                print(f"\nFound {len(server_processes)} Petals server processes:")
                
                # Calculate total memory across all Petals server processes
                total_cpu_mem = sum(p.info['memory_info'].rss for p in server_processes) / (1024**3)
                print(f"Total memory used by all server processes: {total_cpu_mem:.2f}GB")
                
                # Show individual process info
                for i, proc in enumerate(server_processes, 1):
                    try:
                        # Get process info
                        pid = proc.info['pid']
                        cmd = ' '.join(str(c) for c in proc.info['cmdline'][:3]) + '...'
                        cpu_mem = proc.info['memory_info'].rss / (1024**3)
                        
                        # Try to get children processes
                        try:
                            children = proc.children(recursive=True)
                            children_mem = sum(child.memory_info().rss for child in children) / (1024**3)
                        except:
                            children = []
                            children_mem = 0
                        
                        # Print process info
                        print(f"{i}. PID {pid}: {cmd}")
                        print(f"   Memory: {cpu_mem:.2f}GB (process) + {children_mem:.2f}GB (children) = {cpu_mem + children_mem:.2f}GB total")
                        
                        # Try to get specific num_blocks info if available
                        block_info = ""
                        try:
                            cmdline = proc.cmdline()
                            if '--num_blocks' in cmdline:
                                idx = cmdline.index('--num_blocks')
                                if idx + 1 < len(cmdline):
                                    block_info = f", Blocks: {cmdline[idx+1]}"
                        except:
                            pass
                        
                        print(f"   Running time: {(time.time() - proc.create_time()) / 60:.1f} minutes{block_info}")
                    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess) as e:
                        print(f"{i}. Process info unavailable: {e}")
            
            # Check if we should continue
            if time.time() < end_time:
                time.sleep(interval)
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user")
    
    print_header("Monitoring Complete")

def main():
    parser = argparse.ArgumentParser(description="Monitor memory usage of Petals processes")
    parser.add_argument("--interval", type=int, default=5, help="Monitoring interval in seconds")
    parser.add_argument("--duration", type=int, default=60, help="Monitoring duration in seconds")
    args = parser.parse_args()
    
    # Initial system information
    print_header("System Information")
    print(f"CPU: {psutil.cpu_count(logical=False)} cores ({psutil.cpu_count()} logical)")
    print(f"System Memory: {psutil.virtual_memory().total / (1024**3):.2f}GB")
    if NVML_AVAILABLE:
        device_name, total, used, free = get_gpu_memory_info()
        print(f"GPU: {device_name} with {total:.2f}GB memory")
    
    # Start monitoring
    monitor_petals_processes(args.interval, args.duration)

if __name__ == "__main__":
    main() 