#!/usr/bin/env python3
"""
GPU and System Memory utility functions for Jetson llama.cpp benchmarking.

Unlike PyTorch profiler, llama.cpp doesn't expose PyTorch memory APIs.
We track memory using:
- System RAM: /proc/meminfo
- GPU memory: tegrastats or nvidia-smi (Jetson unified memory)
"""
import subprocess
import sys
import time
import re


def get_system_memory():
    """Get system RAM usage from /proc/meminfo.

    Returns:
        dict: {'total': MB, 'used': MB, 'free': MB, 'available': MB,
               'buffers': MB, 'cached': MB} or None if unavailable
    """
    try:
        with open('/proc/meminfo', 'r') as f:
            meminfo = {}
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    key = parts[0].rstrip(':')
                    value = int(parts[1]) / 1024  # Convert KB to MB
                    meminfo[key] = value
            total = meminfo.get('MemTotal', 0)
            free = meminfo.get('MemFree', 0)
            available = meminfo.get('MemAvailable', 0)
            buffers = meminfo.get('Buffers', 0)
            cached = meminfo.get('Cached', 0)
            slab = meminfo.get('Slab', 0)
            used = total - free - buffers - cached
            return {
                'total': total,
                'used': used,
                'free': free,
                'available': available,
                'buffers': buffers,
                'cached': cached,
                'slab': slab,
            }
    except Exception as e:
        print(f"Warning: Could not read memory info: {e}", file=sys.stderr)
        return None


def get_gpu_memory_tegrastats():
    """Get GPU memory from tegrastats (Jetson-specific).

    On Jetson, GPU memory is unified with system memory.
    tegrastats shows GR3D usage (GPU utilization) and RAM usage.

    Returns:
        dict: {'gr3d_pct': int, 'ram_used_mb': int, 'ram_total_mb': int} or None
    """
    try:
        # Run tegrastats once and capture output
        result = subprocess.run(
            ['tegrastats', '--interval', '100'],
            capture_output=True,
            text=True,
            timeout=0.5
        )
        line = result.stdout.strip()
        if not line:
            return None

        # Parse RAM usage: RAM 5678/15815MB
        ram_match = re.search(r'RAM (\d+)/(\d+)MB', line)
        # Parse GR3D (GPU): GR3D_FREQ 0%@306
        gr3d_match = re.search(r'GR3D_FREQ (\d+)%', line)

        return {
            'ram_used_mb': int(ram_match.group(1)) if ram_match else 0,
            'ram_total_mb': int(ram_match.group(2)) if ram_match else 0,
            'gr3d_pct': int(gr3d_match.group(1)) if gr3d_match else 0,
        }
    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
        return None


def get_gpu_memory_nvidia_smi():
    """Get GPU memory using nvidia-smi (discrete GPU or Jetson with nvidia-smi).

    Note: On Jetson with unified memory, nvidia-smi may return [N/A] for memory.
    In that case, we return None and fall back to system memory tracking.

    Returns:
        dict: {'used_mb': float, 'total_mb': float, 'free_mb': float} or None
    """
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used,memory.total,memory.free',
             '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode != 0:
            return None

        parts = result.stdout.strip().split(',')
        if len(parts) >= 3:
            # Check for [N/A] values (common on Jetson unified memory)
            used_str = parts[0].strip()
            total_str = parts[1].strip()
            free_str = parts[2].strip()

            if '[N/A]' in used_str or '[N/A]' in total_str or '[N/A]' in free_str:
                # Jetson unified memory - GPU memory is shared with system RAM
                return None

            return {
                'used_mb': float(used_str),
                'total_mb': float(total_str),
                'free_mb': float(free_str),
            }
        return None
    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError, ValueError):
        return None


def get_mem():
    """Get comprehensive memory snapshot (System RAM and GPU if available).

    For llama.cpp on Jetson (unified memory architecture):
    - GPU memory is shared with system RAM
    - We track both system RAM and tegrastats/nvidia-smi readings

    Returns:
        dict with keys:
            - sys_total: Total system RAM in MB
            - sys_used: Used system RAM in MB
            - sys_free: Free system RAM in MB
            - sys_available: Available system RAM in MB
            - sys_buffers: Buffers in MB
            - sys_cached: Cached in MB
            - gpu_used: GPU memory used in MB (from nvidia-smi, 0 if unavailable)
            - gpu_total: GPU memory total in MB (from nvidia-smi, 0 if unavailable)
            - gpu_free: GPU memory free in MB (from nvidia-smi, 0 if unavailable)
    """
    sys_mem = get_system_memory()
    gpu_mem = get_gpu_memory_nvidia_smi()

    result = {
        'sys_total': sys_mem['total'] if sys_mem else 0,
        'sys_used': sys_mem['used'] if sys_mem else 0,
        'sys_free': sys_mem['free'] if sys_mem else 0,
        'sys_available': sys_mem['available'] if sys_mem else 0,
        'sys_buffers': sys_mem['buffers'] if sys_mem else 0,
        'sys_cached': sys_mem['cached'] if sys_mem else 0,
        'sys_slab': sys_mem.get('slab', 0) if sys_mem else 0,
        'gpu_used': gpu_mem['used_mb'] if gpu_mem else 0,
        'gpu_total': gpu_mem['total_mb'] if gpu_mem else 0,
        'gpu_free': gpu_mem['free_mb'] if gpu_mem else 0,
    }

    return result


def drop_caches():
    """Drop page caches to ensure clean benchmark state (requires root/privileged container)."""
    try:
        subprocess.run(['sync'], check=True)
        # Try native path first, then container path
        for path in ['/proc/sys/vm/drop_caches', '/host/proc/sys/vm/drop_caches']:
            try:
                with open(path, 'w') as f:
                    f.write('3\n')
                print("  Page cache cleared", file=sys.stderr)
                return True
            except (PermissionError, FileNotFoundError, OSError):
                continue
        print("  Warning: Could not drop caches (requires root)", file=sys.stderr)
        return False
    except Exception as e:
        print(f"  Warning: Could not drop caches: {e}", file=sys.stderr)
        return False


def get_free_memory():
    """
    Get system memory info using 'free -m' command (shows MB with more precision).

    Returns:
        str: Output of 'free -m' command, or None if command fails
    """
    try:
        result = subprocess.run(
            ['free', '-m'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            return result.stdout
        return None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


def print_memory_bar(label: str, used: float, total: float, extra_info: str = ""):
    """Print a visual memory bar."""
    if total <= 0:
        return
    pct = (used / total) * 100
    bar_len = 30
    filled = int(bar_len * used / total)
    bar = "█" * filled + "░" * (bar_len - filled)
    info = f" | {extra_info}" if extra_info else ""
    print(f"  [{label}] [{bar}] {used:.0f}/{total:.0f} MB ({pct:.1f}%){info}", file=sys.stderr)


def print_memory_status(label: str) -> dict:
    """Print and return current memory status with visual bar."""
    mem = get_mem()

    # System RAM bar
    print_memory_bar(
        f"{label} SysRAM",
        mem['sys_used'],
        mem['sys_total'],
        f"Avail: {mem['sys_available']:.0f} MB"
    )

    # GPU memory bar (if available)
    if mem['gpu_total'] > 0:
        print_memory_bar(
            f"{label} GPU",
            mem['gpu_used'],
            mem['gpu_total'],
            f"Free: {mem['gpu_free']:.0f} MB"
        )

    return mem


def print_memory_summary_table(stages: list):
    """Print a memory summary table showing all stages.

    Args:
        stages: List of tuples (name, mem_dict, prev_mem_dict, theory_str)
                where mem_dict is from get_mem()
    """
    print(f"\n{'='*85}", file=sys.stderr)
    print(f"  MEMORY SUMMARY", file=sys.stderr)
    print(f"{'='*85}", file=sys.stderr)

    # Header
    print(f"\n  {'Stage':<20} {'SysUsed':>9} {'SysAvail':>9} {'Cached':>8} {'GPU':>8} {'Delta':>8}", file=sys.stderr)
    print(f"  {'-'*20} {'-'*9} {'-'*9} {'-'*8} {'-'*8} {'-'*8}", file=sys.stderr)

    for name, mem, prev, theory in stages:
        if mem is None:
            continue

        # Calculate delta from previous stage
        if prev:
            delta_sys = mem['sys_used'] - prev['sys_used']
            delta_str = f"+{delta_sys:.0f}" if delta_sys >= 0 else f"{delta_sys:.0f}"
        else:
            delta_str = "-"

        gpu_str = f"{mem['gpu_used']:.0f}" if mem['gpu_total'] > 0 else "N/A"

        print(f"  {name:<20} {mem['sys_used']:>9.0f} {mem['sys_available']:>9.0f} {mem['sys_cached']:>8.0f} {gpu_str:>8} {delta_str:>8}", file=sys.stderr)

    print(f"{'='*85}\n", file=sys.stderr)


def estimate_gguf_memory(model_path: str) -> dict:
    """Estimate memory requirements for a GGUF model.

    Args:
        model_path: Path to the GGUF model file

    Returns:
        dict with memory estimates in MB
    """
    import os

    try:
        file_size_bytes = os.path.getsize(model_path)
        file_size_mb = file_size_bytes / 1024 / 1024

        # GGUF models are typically loaded ~1:1 into memory
        # Plus some overhead for metadata and context
        estimated_model_mb = file_size_mb * 1.05  # 5% overhead

        return {
            'file_size_mb': file_size_mb,
            'estimated_model_mb': estimated_model_mb,
        }
    except (OSError, IOError):
        return {
            'file_size_mb': 0,
            'estimated_model_mb': 0,
        }


def estimate_kv_cache_memory(ctx_size: int, num_layers: int = 32,
                             hidden_size: int = 4096, num_kv_heads: int = 8,
                             head_dim: int = 128, dtype_bytes: int = 2) -> float:
    """Estimate KV cache memory for a given context size.

    KV cache per token = 2 (K+V) * num_kv_heads * head_dim * dtype_bytes * num_layers

    Args:
        ctx_size: Context size (max sequence length)
        num_layers: Number of transformer layers
        hidden_size: Hidden dimension
        num_kv_heads: Number of key-value heads (for GQA)
        head_dim: Dimension per head
        dtype_bytes: Bytes per element (2 for FP16, 4 for FP32)

    Returns:
        Estimated KV cache memory in MB
    """
    # KV cache: 2 tensors (K and V) per layer
    # Each tensor: ctx_size * num_kv_heads * head_dim * dtype_bytes
    kv_per_token = 2 * num_kv_heads * head_dim * dtype_bytes * num_layers
    total_bytes = kv_per_token * ctx_size
    return total_bytes / 1024 / 1024


def wait_for_memory_settle(seconds: float = 0.5):
    """Wait for memory operations to settle."""
    time.sleep(seconds)


def minimize_memory():
    """Attempt to minimize memory usage by dropping caches and waiting."""
    drop_caches()
    wait_for_memory_settle(0.5)


if __name__ == "__main__":
    # Test memory functions
    print("Testing memory functions...\n", file=sys.stderr)

    mem = get_mem()
    print(f"System Memory:", file=sys.stderr)
    print(f"  Total: {mem['sys_total']:.0f} MB", file=sys.stderr)
    print(f"  Used: {mem['sys_used']:.0f} MB", file=sys.stderr)
    print(f"  Available: {mem['sys_available']:.0f} MB", file=sys.stderr)
    print(f"  Cached: {mem['sys_cached']:.0f} MB", file=sys.stderr)
    print(f"  Buffers: {mem['sys_buffers']:.0f} MB", file=sys.stderr)

    if mem['gpu_total'] > 0:
        print(f"\nGPU Memory:", file=sys.stderr)
        print(f"  Total: {mem['gpu_total']:.0f} MB", file=sys.stderr)
        print(f"  Used: {mem['gpu_used']:.0f} MB", file=sys.stderr)
        print(f"  Free: {mem['gpu_free']:.0f} MB", file=sys.stderr)

    print(f"\nVisual memory status:", file=sys.stderr)
    print_memory_status("Test")
