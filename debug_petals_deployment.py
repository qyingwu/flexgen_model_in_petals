#!/usr/bin/env python3
"""
Debug script for Petals distributed deployment
This script automates the process of starting a backbone DHT node and multiple peer servers,
while providing monitoring and debugging capabilities.
"""

import os
import signal
import subprocess
import sys
import time
import tempfile
import re
import threading
import json
import argparse
from pathlib import Path
import psutil
import torch
import numpy as np
try:
    from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False
    print("Warning: pynvml not available, GPU memory monitoring will be limited")

# Configuration
DEFAULT_MODEL = "huggyllama/llama-7b"
DEFAULT_NUM_BLOCKS = 2  # Reduced from 4 to 2 blocks for lower memory usage
DEFAULT_PORT = 31340
DEFAULT_VENV_PATH = os.path.expanduser("~/LLM")
DEFAULT_MAX_RAM_GB = 4  # Default RAM limit per server

class GPUMemoryMonitor(threading.Thread):
    """Monitors GPU memory usage in a separate thread"""
    
    def __init__(self, interval=1.0):
        super().__init__()
        self.daemon = True
        self.interval = interval
        self.running = True
        self.memory_history = []
        
        if NVML_AVAILABLE:
            nvmlInit()
            self.handle = nvmlDeviceGetHandleByIndex(0)
        
    def run(self):
        while self.running:
            self.memory_history.append(self.get_gpu_memory())
            time.sleep(self.interval)
    
    def get_gpu_memory(self):
        """Get current GPU memory usage in GB"""
        if NVML_AVAILABLE:
            info = nvmlDeviceGetMemoryInfo(self.handle)
            return info.used / (1024**3)
        
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024**3)
        
        return 0
    
    def stop(self):
        self.running = False
    
    def get_summary(self):
        """Return summary statistics of memory usage"""
        if not self.memory_history:
            return "No memory data collected"
        
        mem_array = np.array(self.memory_history)
        return {
            "min": float(np.min(mem_array)),
            "max": float(np.max(mem_array)),
            "mean": float(np.mean(mem_array)),
            "current": float(mem_array[-1]) if len(mem_array) > 0 else 0
        }


class PetalsDeployment:
    """Manages the deployment of a Petals backbone and server nodes"""
    
    def __init__(self, model=DEFAULT_MODEL, num_blocks=DEFAULT_NUM_BLOCKS, 
                 port=DEFAULT_PORT, venv_path=DEFAULT_VENV_PATH, max_ram_gb=DEFAULT_MAX_RAM_GB):
        self.model = model
        self.num_blocks = num_blocks
        self.port = port
        self.venv_path = venv_path
        self.max_ram_gb = max_ram_gb
        self.workspace_dir = os.getcwd()
        
        # Use temporary file paths with unique names to avoid conflicts
        unique_id = str(int(time.time()))
        self.backbone_id_path = os.path.join(self.workspace_dir, f"backbone_debug_{unique_id}.id")
        self.server1_id_path = os.path.join(self.workspace_dir, f"server1_debug_{unique_id}.id")
        self.server2_id_path = os.path.join(self.workspace_dir, f"server2_debug_{unique_id}.id")
        
        # Process handles
        self.backbone_process = None
        self.server1_process = None
        self.server2_process = None
        self.peer_id = None
        
        # Log files
        self.backbone_output = tempfile.NamedTemporaryFile(prefix="backbone_", suffix=".log", delete=False, mode="w")
        self.server1_output = tempfile.NamedTemporaryFile(prefix="server1_", suffix=".log", delete=False, mode="w")
        self.server2_output = tempfile.NamedTemporaryFile(prefix="server2_", suffix=".log", delete=False, mode="w")
        
        # Memory monitor
        self.memory_monitor = None
    
    def activate_venv_cmd(self):
        """Get command to activate virtual environment"""
        if os.name == 'nt':  # Windows
            return f"call {os.path.join(self.venv_path, 'Scripts', 'activate.bat')}"
        else:  # Linux/Mac
            activate_path = os.path.join(self.venv_path, 'bin', 'activate')
            # Verify that the activation script exists
            if os.path.exists(activate_path):
                return f"source {activate_path}"
            else:
                print(f"Warning: Virtual environment activation script not found at {activate_path}")
                print(f"Using full path to Python in virtual environment instead")
                # Use direct path to Python in venv
                return f"export PYTHONPATH={os.getcwd()}:$PYTHONPATH && {os.path.join(self.venv_path, 'bin', 'python')}"
    
    def cleanup_old_identity_files(self):
        """Clean up any old identity files from previous runs"""
        for filename in [self.backbone_id_path, self.server1_id_path, self.server2_id_path]:
            if os.path.exists(filename):
                try:
                    os.remove(filename)
                    print(f"Removed old identity file: {filename}")
                except OSError as e:
                    print(f"Error removing {filename}: {e}")
    
    def kill_existing_processes(self):
        """Kill any existing Petals processes"""
        print("Terminating any existing Petals processes...")
        
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                cmdline = ' '.join(proc.cmdline())
                if 'petals.cli.run' in cmdline:
                    print(f"Killing process {proc.pid}: {cmdline}")
                    proc.terminate()
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass
            
        # Give processes time to terminate
        time.sleep(2)
    
    def start_backbone(self):
        """Start the backbone DHT node"""
        print("Starting backbone DHT node...")
        
        # Use the Python from the virtual environment directly
        python_path = os.path.join(self.venv_path, 'bin', 'python')
        if not os.path.exists(python_path):
            raise RuntimeError(f"Python interpreter not found at {python_path}")
        
        cmd = f"{python_path} -m petals.cli.run_dht --host_maddrs /ip4/0.0.0.0/tcp/{self.port} --identity_path {self.backbone_id_path}"
        
        print(f"Running command: {cmd}")
        
        if os.name == 'nt':  # Windows
            self.backbone_process = subprocess.Popen(
                cmd, shell=True, stdout=self.backbone_output, stderr=subprocess.STDOUT
            )
        else:  # Linux/Mac
            self.backbone_process = subprocess.Popen(
                cmd, shell=True, executable='/bin/bash', 
                stdout=self.backbone_output, stderr=subprocess.STDOUT
            )
        
        print(f"Backbone process started with PID: {self.backbone_process.pid}")
        print(f"Backbone logs being written to: {self.backbone_output.name}")
        
        # Wait for the backbone to initialize and extract peer ID
        self._wait_for_peer_id()
    
    def _wait_for_peer_id(self):
        """Wait for backbone to initialize and extract peer ID from logs"""
        print("Waiting for backbone to initialize and extract peer ID...")
        
        max_wait = 60  # Extended timeout from 30 to 60 seconds
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            # Check if process is still running
            if self.backbone_process.poll() is not None:
                # Process terminated - this is an error but we'll check the logs for details
                with open(self.backbone_output.name, 'r') as f:
                    log_content = f.read()
                
                print("\nBackbone process terminated prematurely. Last 20 lines of log:")
                lines = log_content.split('\n')
                for line in lines[-20:]:
                    print(f"  {line}")
                
                raise RuntimeError(f"Backbone process terminated prematurely with code {self.backbone_process.returncode}")
            
            # Read the current log file
            with open(self.backbone_output.name, 'r') as f:
                content = f.read()
            
            # Look for peer ID in the log
            match = re.search(r'--initial_peers (/ip4/[\d\.]+/tcp/\d+/p2p/[a-zA-Z0-9]+)', content)
            if match:
                self.peer_id = match.group(1)
                print(f"Found peer ID: {self.peer_id}")
                return
            
            # Check for errors
            if "Error" in content or "ERROR" in content:
                print("\nDetected error in backbone logs:")
                error_lines = [line for line in content.split('\n') if "Error" in line or "ERROR" in line]
                for line in error_lines:
                    print(f"  {line}")
                print("\nFull log content:")
                print(content)
                raise RuntimeError("Errors found in backbone initialization")
            
            time.sleep(2)
        
        print("\nTimeout waiting for backbone to initialize. Last 20 lines of log:")
        with open(self.backbone_output.name, 'r') as f:
            content = f.read()
        lines = content.split('\n')
        for line in lines[-20:]:
            print(f"  {line}")
        
        raise TimeoutError("Timeout waiting for backbone to initialize and provide peer ID")
    
    def start_server(self, server_num, id_path, output_file):
        """Start a server node"""
        print(f"Starting server {server_num}...")
        
        if self.peer_id is None:
            raise ValueError("Backbone peer ID not available. Cannot start server.")
        
        # Use the Python from the virtual environment directly
        python_path = os.path.join(self.venv_path, 'bin', 'python')
        if not os.path.exists(python_path):
            raise RuntimeError(f"Python interpreter not found at {python_path}")
        
        # Command with explicit device and cache directory, but without max_ram_gb
        cmd = (
            f"export PEER={self.peer_id} && "
            f"{python_path} -m petals.cli.run_server {self.model} "
            f"--initial_peers $PEER "
            f"--num_blocks {self.num_blocks} "
            f"--identity_path {id_path} "
            f"--device cuda:0 "  # Explicitly set device 
            f"--cache_dir ./cache"  # Explicitly set cache directory
        )
        
        print(f"Running command for server {server_num}: {cmd}")
        
        if os.name == 'nt':  # Windows
            process = subprocess.Popen(
                cmd, shell=True, stdout=output_file, stderr=subprocess.STDOUT
            )
        else:  # Linux/Mac
            process = subprocess.Popen(
                cmd, shell=True, executable='/bin/bash', 
                stdout=output_file, stderr=subprocess.STDOUT
            )
        
        print(f"Server {server_num} process started with PID: {process.pid}")
        print(f"Server {server_num} logs being written to: {output_file.name}")
        return process
    
    def start_all_servers(self):
        """Start all server nodes"""
        self.server1_process = self.start_server(1, self.server1_id_path, self.server1_output)
        
        # Wait a bit before starting the second server to avoid race conditions
        time.sleep(10)  # Increased from 5 to 10 seconds
        
        self.server2_process = self.start_server(2, self.server2_id_path, self.server2_output)
    
    def check_server_health(self, server_num, process, log_file):
        """Check if a server is running and healthy"""
        if process.poll() is not None:
            print(f"Server {server_num} terminated with code {process.returncode}")
            print("Last few lines of log:")
            
            with open(log_file, 'r') as f:
                lines = f.readlines()
                for line in lines[-20:]:  # Show last 20 lines
                    print(f"  {line.strip()}")
            
            # Check for specific errors
            with open(log_file, 'r') as f:
                content = f.read()
                if "AttributeError: 'ValueHolder' object has no attribute 'data'" in content:
                    print("❌ ERROR: 'ValueHolder' object has no attribute 'data' - This needs to be fixed in the codebase")
                    print("   The ValueHolder.data should be accessed as ValueHolder.val")
                
                if "TypeError: get_block_size() missing" in content:
                    print("❌ ERROR: 'get_block_size()' is missing required parameters - Check the function call")
            
            return False
        
        return True
    
    def monitor_deployment(self, duration=60):
        """Monitor the deployment for the specified duration"""
        print(f"\nMonitoring deployment for {duration} seconds...")
        
        # Start memory monitoring
        self.memory_monitor = GPUMemoryMonitor()
        self.memory_monitor.start()
        
        start_time = time.time()
        check_interval = 5  # seconds
        
        while time.time() - start_time < duration:
            # Check backbone health
            if self.backbone_process.poll() is not None:
                print(f"❌ Backbone process terminated with code {self.backbone_process.returncode}")
                with open(self.backbone_output.name, 'r') as f:
                    lines = f.readlines()
                    for line in lines[-20:]:  # Show last 20 lines
                        print(f"  {line.strip()}")
                break
            
            # Check server health
            server1_healthy = self.check_server_health(1, self.server1_process, self.server1_output.name)
            server2_healthy = self.check_server_health(2, self.server2_process, self.server2_output.name)
            
            if not server1_healthy and not server2_healthy:
                print("❌ Both servers have terminated. Stopping monitoring.")
                break
            
            # Get current memory usage
            current_mem = self.memory_monitor.get_gpu_memory()
            print(f"Current GPU memory usage: {current_mem:.2f} GB")
            
            time.sleep(check_interval)
        
        # Stop memory monitoring
        self.memory_monitor.stop()
        
        # Print memory summary
        mem_summary = self.memory_monitor.get_summary()
        if isinstance(mem_summary, dict):
            print("\nGPU Memory Usage Summary:")
            print(f"  Min: {mem_summary['min']:.2f} GB")
            print(f"  Max: {mem_summary['max']:.2f} GB")
            print(f"  Mean: {mem_summary['mean']:.2f} GB")
            print(f"  Current: {mem_summary['current']:.2f} GB")
        else:
            print(f"\nGPU Memory Usage Summary: {mem_summary}")
    
    def analyze_logs(self):
        """Analyze logs for common issues"""
        print("\nAnalyzing logs for common issues...")
        
        self._check_for_valueholder_issues()
        # You can add more analysis methods here
    
    def _check_for_valueholder_issues(self):
        """Check logs for ValueHolder related issues"""
        print("\nChecking for ValueHolder issues...")
        
        # Pattern for ValueHolder issues
        data_attr_error = "AttributeError: 'ValueHolder' object has no attribute 'data'"
        value_attr_error = "AttributeError: 'ValueHolder' object has no attribute 'value'"
        none_val_warning = "ValueHolder for block [0-9]+ (found but val is None|has None val)"
        
        issues_found = False
        
        # Check server1 logs
        if os.path.exists(self.server1_output.name):
            with open(self.server1_output.name, 'r') as f:
                content = f.read()
                
                if re.search(data_attr_error, content):
                    print("❌ ERROR: 'ValueHolder' object has no attribute 'data' found in server1 logs")
                    print("   Fix: Update from_pretrained.py to use .val instead of .data")
                    issues_found = True
                
                if re.search(value_attr_error, content):
                    print("❌ ERROR: 'ValueHolder' object has no attribute 'value' found in server1 logs")
                    issues_found = True
                
                none_vals = re.findall(none_val_warning, content)
                if none_vals:
                    print(f"⚠️ WARNING: Found {len(none_vals)} instances of ValueHolder with None val in server1 logs")
                    print("   Fix: Ensure weights are properly stored using ValueHolder.store() method")
                    issues_found = True
        
        # Check server2 logs
        if os.path.exists(self.server2_output.name):
            with open(self.server2_output.name, 'r') as f:
                content = f.read()
                
                if re.search(data_attr_error, content):
                    print("❌ ERROR: 'ValueHolder' object has no attribute 'data' found in server2 logs")
                    print("   Fix: Update from_pretrained.py to use .val instead of .data")
                    issues_found = True
                
                if re.search(value_attr_error, content):
                    print("❌ ERROR: 'ValueHolder' object has no attribute 'value' found in server2 logs")
                    issues_found = True
                
                none_vals = re.findall(none_val_warning, content)
                if none_vals:
                    print(f"⚠️ WARNING: Found {len(none_vals)} instances of ValueHolder with None val in server2 logs")
                    print("   Fix: Ensure weights are properly stored using ValueHolder.store() method")
                    issues_found = True
        
        if not issues_found:
            print("✅ No ValueHolder issues found in logs")
    
    def cleanup(self):
        """Clean up resources"""
        print("\nCleaning up...")
        
        # Terminate processes
        for process_name, process in [
            ("Backbone", self.backbone_process),
            ("Server1", self.server1_process),
            ("Server2", self.server2_process)
        ]:
            if process is not None and process.poll() is None:
                print(f"Terminating {process_name} process (PID: {process.pid})...")
                try:
                    process.terminate()
                    # Wait for process to terminate
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        print(f"Killing {process_name} process forcefully...")
                        process.kill()
                except Exception as e:
                    print(f"Error terminating {process_name} process: {e}")
        
        # Display log file paths
        for name, log_file in [
            ("Backbone", self.backbone_output),
            ("Server1", self.server1_output),
            ("Server2", self.server2_output)
        ]:
            if hasattr(log_file, 'name'):
                print(f"{name} logs available at: {log_file.name}")
        
        # Remove identity files
        for filename in [self.backbone_id_path, self.server1_id_path, self.server2_id_path]:
            if os.path.exists(filename):
                try:
                    os.remove(filename)
                except OSError:
                    pass
        
        print("Cleanup complete")
    
    def run_full_deployment(self, monitor_duration=120):
        """Run the full deployment process"""
        try:
            print("=" * 60)
            print("PETALS DISTRIBUTED DEPLOYMENT DEBUGGING")
            print("=" * 60)
            print(f"Model: {self.model}")
            print(f"Number of blocks per server: {self.num_blocks}")
            print(f"Max RAM per server: {self.max_ram_gb} GB")
            print(f"Using GPU memory optimization with balanced CPU/GPU distribution")
            print("=" * 60)
            
            # Create cache directory if it doesn't exist
            os.makedirs("./cache", exist_ok=True)
            
            self.kill_existing_processes()
            self.cleanup_old_identity_files()
            
            self.start_backbone()
            time.sleep(10)  # Give backbone time to stabilize
            
            self.start_all_servers()
            
            self.monitor_deployment(duration=monitor_duration)
            
            self.analyze_logs()
            
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.cleanup()


def main():
    parser = argparse.ArgumentParser(description="Debug and analyze Petals distributed deployment")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Model to use (default: {DEFAULT_MODEL})")
    parser.add_argument("--blocks", type=int, default=DEFAULT_NUM_BLOCKS, help=f"Number of blocks per server (default: {DEFAULT_NUM_BLOCKS})")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"Port for backbone DHT node (default: {DEFAULT_PORT})")
    parser.add_argument("--venv", default=DEFAULT_VENV_PATH, help=f"Path to virtual environment (default: {DEFAULT_VENV_PATH})")
    parser.add_argument("--duration", type=int, default=120, help="Duration to monitor the deployment in seconds (default: 120)")
    parser.add_argument("--max-ram", type=int, default=DEFAULT_MAX_RAM_GB, help=f"Maximum RAM usage in GB per server (default: {DEFAULT_MAX_RAM_GB})")
    
    args = parser.parse_args()
    
    deployment = PetalsDeployment(
        model=args.model,
        num_blocks=args.blocks,
        port=args.port,
        venv_path=args.venv,
        max_ram_gb=args.max_ram
    )
    
    deployment.run_full_deployment(monitor_duration=args.duration)


if __name__ == "__main__":
    main() 