#!/usr/bin/env python3
"""
GPU utility functions for Jetson benchmarking.
"""
import ctypes
import subprocess
import sys
import time
import torch


def get_system_memory():
    """Get system RAM usage from /proc/meminfo.

    Returns:
        dict: {'total': MB, 'used': MB, 'free': MB, 'available': MB} or None if unavailable
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
            used = total - free - buffers - cached
            return {'total': total, 'used': used, 'free': free, 'available': available}
    except:
        return None


def get_mem():
    """Get comprehensive memory snapshot (GPU and system).

    Returns:
        dict with keys:
            - gpu_total: Total GPU memory in MB
            - gpu_used: Used GPU memory in MB
            - gpu_free: Free GPU memory in MB
            - pytorch_alloc: PyTorch allocated memory in MB
            - pytorch_reserved: PyTorch reserved (pool) memory in MB
            - pool_overhead: Reserved - Allocated (fragmentation) in MB
            - non_pytorch: GPU used - PyTorch reserved (CUDA/libs overhead) in MB
            - sys_used: System RAM used in MB
            - sys_available: System RAM available in MB
    """
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    gpu_used = (total - free) / 1024**2
    pytorch_alloc = torch.cuda.memory_allocated() / 1024**2
    pytorch_reserved = torch.cuda.memory_reserved() / 1024**2
    pool_overhead = pytorch_reserved - pytorch_alloc
    sys_mem = get_system_memory()

    return {
        'gpu_total': total / 1024**2,
        'gpu_used': gpu_used,
        'gpu_free': free / 1024**2,
        'pytorch_alloc': pytorch_alloc,
        'pytorch_reserved': pytorch_reserved,
        'pool_overhead': pool_overhead,
        'non_pytorch': gpu_used - pytorch_reserved,
        'sys_used': sys_mem['used'] if sys_mem else 0,
        'sys_available': sys_mem['available'] if sys_mem else 0,
    }


def drop_caches():
    """Drop page caches to ensure clean benchmark state (requires root/privileged container)."""
    try:
        subprocess.run(['sync'], check=True)
        # Try native path first, then container path
        for path in ['/proc/sys/vm/drop_caches', '/host/proc/sys/vm/drop_caches']:
            try:
                with open(path, 'w') as f:
                    f.write('3\n')
                print("Page cache cleared", file=sys.stderr)
                return
            except (PermissionError, FileNotFoundError, OSError):
                continue
        print("Warning: Could not drop caches (requires root)", file=sys.stderr)
    except Exception as e:
        print(f"Warning: Could not drop caches: {e}", file=sys.stderr)


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


def test_gpu():
    """Test basic GPU/CUDA access."""
    print("=" * 50, file=sys.stderr)
    print("GPU/CUDA Test", file=sys.stderr)
    print("=" * 50, file=sys.stderr)

    print(f"PyTorch version: {torch.__version__}", file=sys.stderr)
    print(f"CUDA available: {torch.cuda.is_available()}", file=sys.stderr)

    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available!", file=sys.stderr)
        return False

    print(f"CUDA version: {torch.version.cuda}", file=sys.stderr)
    print(f"Device count: {torch.cuda.device_count()}", file=sys.stderr)
    print(f"Current device: {torch.cuda.current_device()}", file=sys.stderr)
    print(f"Device name: {torch.cuda.get_device_name(0)}", file=sys.stderr)

    print("\nTesting tensor operation on GPU...", file=sys.stderr)
    try:
        x = torch.randn(100, 100, device='cuda')
        y = torch.matmul(x, x)
        print(f"Tensor operation successful! Result shape: {y.shape}", file=sys.stderr)
        del x, y
        torch.cuda.empty_cache()
        return True
    except Exception as e:
        print(f"ERROR: Tensor operation failed: {e}", file=sys.stderr)
        return False


def clean_docker_cache():
    """
    Clean Docker build cache to free up disk space.
    
    This removes Docker's build cache, which is different from GPU memory cache.
    Docker cache is stored on disk and contains intermediate build layers.
    
    Note: This only works if Docker is installed and accessible. If running inside
    a container, Docker may not be available in the PATH.
    
    Returns:
        bool: True if successful, False otherwise
    """
    # Try to find docker in common locations
    docker_paths = [
        'docker',  # Try PATH first
        '/usr/bin/docker',
        '/usr/local/bin/docker',
        '/bin/docker',
    ]
    
    docker_cmd = None
    for path in docker_paths:
        try:
            # Check if command exists
            result = subprocess.run(
                ['which', path] if path == 'docker' else ['test', '-f', path],
                capture_output=True,
                timeout=1
            )
            if result.returncode == 0:
                docker_cmd = path
                break
        except (subprocess.TimeoutExpired, FileNotFoundError):
            continue
    
    if docker_cmd is None:
        # Docker not found - silently skip (common when running inside container)
        print("Docker not found - silently skip (common when running inside container)", file=sys.stderr)
        return False
    
    try:
        # Remove all unused build cache
        result = subprocess.run(
            [docker_cmd, 'builder', 'prune', '-a', '-f'],
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        if result.returncode == 0:
            print("Docker build cache cleaned successfully", file=sys.stderr)
            return True
        else:
            # Only print error if it's not a permission/access issue
            if 'permission denied' not in result.stderr.lower():
                print(f"Docker cache cleanup failed: {result.stderr}", file=sys.stderr)
            else:
                print(f"Docker cache cleanup failed: {result.stderr}", file=sys.stderr)
            return False
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        # Silently fail - Docker may not be available
        print(f"Docker cache cleanup failed: {e}", file=sys.stderr)
        return False
    except Exception as e:
        # Other errors - log but don't fail loudly
        print(f"Docker cache cleanup failed: {e}", file=sys.stderr)
        return False


def minimize_memory_pool():
    """Attempt to minimize PyTorch's memory pool overhead and clean Docker cache.
    
    This function tries to release cached memory blocks back to CUDA to minimize
    the gap between memory_allocated() and memory_reserved(). Note that PyTorch's
    allocator may still keep some memory for efficiency, so allocated may not
    exactly equal reserved.
    
    Also cleans Docker build cache to free up disk space.
    
    Returns:
        tuple: (allocated_before, reserved_before, allocated_after, reserved_after) in MB
    """
    if not torch.cuda.is_available():
        return None
    
    allocated_before = torch.cuda.memory_allocated() / 1024**2
    reserved_before = torch.cuda.memory_reserved() / 1024**2
    
    # Synchronize to ensure all operations complete
    torch.cuda.synchronize()
    
    # Clear PyTorch's GPU memory cache (not Docker cache)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    
    # Collect IPC handles (for multi-process scenarios)
    try:
        torch.cuda.ipc_collect()
    except:
        pass
    
    # Clean Docker build cache (frees disk space, separate from GPU memory)
    # clean_docker_cache()
    drop_caches()
    
    allocated_after = torch.cuda.memory_allocated() / 1024**2
    reserved_after = torch.cuda.memory_reserved() / 1024**2
    
    return (allocated_before, reserved_before, allocated_after, reserved_after)


def print_gpu_memory(label: str, detailed: bool = False, minimize_pool: bool = False, show_nvidia_smi: bool = True):
    """Print current GPU memory usage with visual bar.
    
    Args:
        label: Label for this memory checkpoint
        detailed: If True, show detailed breakdown of memory sources
        minimize_pool: If True, attempt to minimize memory pool before printing
        show_nvidia_smi: If True, also show nvidia-smi output for comparison
    """
    if torch.cuda.is_available():
        # Get initial state before minimizing pool
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        
        # Try CUDA runtime first (no PyTorch dependency), fallback to PyTorch
        # free_mb, total_mb = get_gpu_memory_cuda_runtime()
        # if free_mb is None or total_mb is None:
        #     # Fallback to PyTorch
        #     free, total = torch.cuda.mem_get_info()
        #     total_mb = total / 1024**2
        #     free_mb = free / 1024**2
        
        # Also get PyTorch values for comparison
        free, total = torch.cuda.mem_get_info()
        total_mb = total / 1024**2
        free_mb = free / 1024**2
        
        used = total_mb - free_mb
        used_pct = (used / total_mb) * 100 if total_mb > 0 else 0
        bar_len = 30
        filled = int(bar_len * used / total_mb) if total_mb > 0 else 0
        bar = "█" * filled + "░" * (bar_len - filled)
        
        # Calculate unaccounted memory (total GPU usage - PyTorch tracked)
        unaccounted = used - allocated
        
        # Print BEFORE state if minimizing pool
        if minimize_pool:
            # Print system memory BEFORE pool minimization
            free_output_before = get_free_memory()
            if free_output_before:
                print(f"  [{label}] System memory BEFORE pool minimization:", file=sys.stderr)
                for line in free_output_before.strip().split('\n'):
                    print(f"    {line}", file=sys.stderr)
            
            print(f"  [{label}] BEFORE pool minimization:", file=sys.stderr)
            print(f"  [{label}] GPU: [{bar}] {used:.0f}/{total_mb:.0f} MB ({used_pct:.1f}%) | Free: {free_mb:.0f} MB | PyTorch: {allocated:.0f} MB", file=sys.stderr)
            # print(f"  [{label}] Comparison - CUDA runtime: {free_mb:.0f}/{total_mb:.0f} MB, PyTorch: {free_mb_torch:.0f}/{total_mb_torch:.0f} MB", file=sys.stderr)
            
            if detailed:
                print(f"    └─ PyTorch Breakdown:", file=sys.stderr)
                print(f"       • Allocated (active tensors): {allocated:.0f} MB", file=sys.stderr)
                print(f"       • Pool overhead:              {max(0, reserved - allocated):.0f} MB", file=sys.stderr)
                print(f"       • Reserved (memory pool):     {reserved:.0f} MB", file=sys.stderr)

                if reserved > allocated:
                    stats = torch.cuda.memory_stats()
                    # Components of pool overhead:
                    inactive_split = stats.get('inactive_split_bytes.all.current', 0) / 1024**2  # MB
                    segment_pool = stats.get('segment_pool_bytes.all.current', 0) / 1024**2  # MB
                    active = stats.get('active_bytes.all.current', 0) / 1024**2  # MB
                    print(f"  [{label}] Pool stats: {inactive_split:.0f} MB (inactive split), {segment_pool:.0f} MB (segment pool), {active:.0f} MB (active)", file=sys.stderr)
        
        # Optionally minimize pool overhead before measuring
        if minimize_pool:
            pool_stats = minimize_memory_pool()
            if pool_stats:
                # print(f"  [{label}] Pool stats: {pool_stats}", file=sys.stderr)
                allocated_before, reserved_before, allocated_after, reserved_after = pool_stats
                pool_reduction = reserved_before - reserved_after
                if pool_reduction > 1:  # Only print if significant reduction
                    print(f"  [{label}] Pool minimized: reserved {reserved_before:.0f} → {reserved_after:.0f} MB (freed {pool_reduction:.0f} MB)", file=sys.stderr)
            
            # Get state AFTER minimizing pool
            allocated = torch.cuda.memory_allocated() / 1024**2
            reserved = torch.cuda.memory_reserved() / 1024**2
            
            # Try CUDA runtime first (no PyTorch dependency), fallback to PyTorch
            # free_mb, total_mb = get_gpu_memory_cuda_runtime()
            # if free_mb is None or total_mb is None:
            #     # Fallback to PyTorch
            #     free, total = torch.cuda.mem_get_info()
            #     total_mb = total / 1024**2
            #     free_mb = free / 1024**2
            
            # Also get PyTorch values for comparison
            free, total = torch.cuda.mem_get_info()
            total_mb = total / 1024**2
            free_mb = free / 1024**2
            
            used = total_mb - free_mb
            used_pct = (used / total_mb) * 100 if total_mb > 0 else 0
            bar_len = 30
            filled = int(bar_len * used / total_mb) if total_mb > 0 else 0
            bar = "█" * filled + "░" * (bar_len - filled)
            
            # Calculate unaccounted memory (total GPU usage - PyTorch tracked)
            unaccounted = used - allocated
            
            # Pause to allow memory to stabilize after pool minimization
            time.sleep(1)
            
            # Print system memory AFTER pool minimization
            free_output_after = get_free_memory()
            if free_output_after:
                print(f"  [{label}] System memory AFTER pool minimization:", file=sys.stderr)
                for line in free_output_after.strip().split('\n'):
                    print(f"    {line}", file=sys.stderr)
        
        print(f"  [{label}] After pool minimization:", file=sys.stderr)
        print(f"  [{label}] GPU: [{bar}] {used:.0f}/{total_mb:.0f} MB ({used_pct:.1f}%) | Free: {free_mb:.0f} MB | PyTorch: {allocated:.0f} MB", file=sys.stderr)
        
        # Show comparison if values differ
        # if abs(total_mb - total_mb_torch) > 1 or abs(free_mb - free_mb_torch) > 1:
        # print(f"  [{label}] Comparison - CUDA runtime: {free_mb:.0f}/{total_mb:.0f} MB, PyTorch: {free_mb_torch:.0f}/{total_mb_torch:.0f} MB", file=sys.stderr)
        
        if detailed:
            print(f"    └─ PyTorch Breakdown:", file=sys.stderr)
            print(f"       • Allocated (active tensors): {allocated:.0f} MB", file=sys.stderr)
            print(f"       • Pool overhead:              {max(0, reserved - allocated):.0f} MB", file=sys.stderr)
            print(f"       • Reserved (memory pool):     {reserved:.0f} MB", file=sys.stderr)

            if reserved > allocated:
                stats = torch.cuda.memory_stats()
                # Components of pool overhead:
                inactive_split = stats.get('inactive_split_bytes.all.current', 0) / 1024**2  # MB
                segment_pool = stats.get('segment_pool_bytes.all.current', 0) / 1024**2  # MB
                active = stats.get('active_bytes.all.current', 0) / 1024**2  # MB
                print(f"  [{label}] Pool stats: {inactive_split:.0f} MB (inactive split), {segment_pool:.0f} MB (segment pool), {active:.0f} MB (active)", file=sys.stderr)
                


def get_gpu_memory_used_mb():
    """Get current GPU memory used in MB (same as nvidia-smi/tegrastats)."""
    if not torch.cuda.is_available():
        return 0
    free, total = torch.cuda.mem_get_info()
    return (total - free) / 1024**2


def get_gpu_memory_cuda_runtime(device_id: int = 0):
    """
    Get GPU memory info using CUDA Runtime API directly (no PyTorch dependency).
    
    Uses ctypes to call libcuda.so functions directly. This is useful when you
    want to check GPU memory without initializing PyTorch or when PyTorch isn't available.
    
    Args:
        device_id: CUDA device ID (default: 0)
    
    Returns:
        tuple: (free_mb, total_mb) in MB, or (None, None) if CUDA not available
    """
    try:
        # Load CUDA runtime library
        libcuda = ctypes.CDLL('libcuda.so')
        
        # Define CUDA function signatures
        # cuMemGetInfo_v2(unsigned long long *free, unsigned long long *total)
        libcuda.cuMemGetInfo_v2.argtypes = [ctypes.POINTER(ctypes.c_size_t), 
                                             ctypes.POINTER(ctypes.c_size_t)]
        libcuda.cuMemGetInfo_v2.restype = ctypes.c_int
        
        # cuInit(unsigned int Flags)
        libcuda.cuInit.argtypes = [ctypes.c_uint]
        libcuda.cuInit.restype = ctypes.c_int
        
        # cuDeviceGet(ctypes.POINTER(ctypes.c_int), int)
        libcuda.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
        libcuda.cuDeviceGet.restype = ctypes.c_int
        
        # cuCtxCreate_v2(ctypes.POINTER(ctypes.c_void_p), unsigned int, int)
        libcuda.cuCtxCreate_v2.argtypes = [ctypes.POINTER(ctypes.c_void_p), 
                                            ctypes.c_uint, ctypes.c_int]
        libcuda.cuCtxCreate_v2.restype = ctypes.c_int
        
        # Initialize CUDA
        result = libcuda.cuInit(0)
        if result != 0:  # CUDA_SUCCESS = 0
            return None, None
        
        # Get device handle
        device = ctypes.c_int()
        result = libcuda.cuDeviceGet(ctypes.byref(device), device_id)
        if result != 0:
            return None, None
        
        # Create context (required to query memory)
        ctx = ctypes.c_void_p()
        result = libcuda.cuCtxCreate_v2(ctypes.byref(ctx), 0, device)
        if result != 0:
            return None, None
        
        # Query memory info
        free = ctypes.c_size_t()
        total = ctypes.c_size_t()
        result = libcuda.cuMemGetInfo_v2(ctypes.byref(free), ctypes.byref(total))
        
        if result != 0:
            return None, None
        
        free_mb = free.value / 1024**2
        total_mb = total.value / 1024**2
        
        return free_mb, total_mb
        
    except (OSError, AttributeError, ValueError):
        # libcuda.so not found or function not available
        return None, None


def get_gpu_memory_sysfs():
    """
    Get GPU memory info from /sys filesystem (Jetson-specific, no CUDA/PyTorch needed).
    
    This reads from sysfs which may have GPU memory information on some systems.
    Note: This is system-specific and may not work on all platforms.
    
    Returns:
        tuple: (free_mb, total_mb) in MB, or (None, None) if not available
    """
    # Try common sysfs paths for GPU memory info
    # Note: These paths are platform-specific and may not exist
    sysfs_paths = [
        '/sys/class/nvidia-gpu/memory/total',
        '/sys/devices/gpu.0/memory/total',
    ]
    
    for path in sysfs_paths:
        try:
            with open(path, 'r') as f:
                total_bytes = int(f.read().strip())
                total_mb = total_bytes / 1024**2
                # Free memory is harder to get from sysfs, would need to query CUDA
                return None, total_mb  # Only total available
        except (FileNotFoundError, ValueError, PermissionError):
            continue
    
    return None, None


def measure_cuda_context_overhead():
    """
    Measure CUDA context initialization overhead.

    This initializes CUDA and measures the memory overhead before any model loading.
    Call this BEFORE loading any models to see how much memory CUDA runtime takes.

    Returns:
        tuple: (gpu_before_mb, gpu_after_cuda_init_mb, cuda_overhead_mb)
    """
    import subprocess

    # Get GPU memory before any CUDA operations (using tegrastats/system query)
    try:
        # Use tegrastats to get memory before CUDA init
        result = subprocess.run(
            ['cat', '/sys/devices/gpu.0/load'],  # Just to check GPU exists
            capture_output=True, text=True, timeout=1
        )
    except:
        pass

    # Get baseline GPU memory (before CUDA context)
    free_before, total = torch.cuda.mem_get_info()
    gpu_before_mb = (total - free_before) / 1024**2

    # Initialize CUDA context by creating a small tensor
    if not hasattr(measure_cuda_context_overhead, '_initialized'):
        # Force CUDA initialization
        _ = torch.zeros(1, device='cuda')
        torch.cuda.synchronize()
        measure_cuda_context_overhead._initialized = True

    # Measure after CUDA init
    free_after, _ = torch.cuda.mem_get_info()
    gpu_after_mb = (total - free_after) / 1024**2

    cuda_overhead_mb = gpu_after_mb - gpu_before_mb

    print(f"  [CUDA Context] Before init: {gpu_before_mb:.0f} MB, After init: {gpu_after_mb:.0f} MB, Overhead: {cuda_overhead_mb:.0f} MB", file=sys.stderr)

    return gpu_before_mb, gpu_after_mb, cuda_overhead_mb


def get_detailed_memory_stats():
    """
    Get detailed memory breakdown using torch.cuda.memory_stats().
    
    Returns a dictionary with detailed memory statistics including:
    - Allocated memory (current, peak, total)
    - Reserved memory (current, peak)
    - Active memory
    - Inactive memory
    - Non-releasable memory
    - Oversized allocations
    
    Returns:
        dict: Detailed memory statistics in MB, or None if CUDA unavailable
    """
    if not torch.cuda.is_available():
        return None
    
    stats = torch.cuda.memory_stats()
    
    # Convert bytes to MB
    def to_mb(bytes_val):
        return bytes_val / 1024**2 if bytes_val else 0
    
    return {
        # Current allocations
        'allocated_mb': to_mb(stats.get('allocated_bytes.all.current', 0)),
        'reserved_mb': to_mb(stats.get('reserved_bytes.all.current', 0)),
        'active_mb': to_mb(stats.get('active_bytes.all.current', 0)),
        'inactive_split_mb': to_mb(stats.get('inactive_split_bytes.all.current', 0)),
        
        # Peak values
        'allocated_peak_mb': to_mb(stats.get('allocated_bytes.all.peak', 0)),
        'reserved_peak_mb': to_mb(stats.get('reserved_bytes.all.peak', 0)),
        
        # Cumulative
        'allocated_total_mb': to_mb(stats.get('allocated_bytes.all.allocated', 0)),
        'freed_total_mb': to_mb(stats.get('allocated_bytes.all.freed', 0)),
        
        # Fragmentation and overhead
        'segment_pool_mb': to_mb(stats.get('segment_pool_bytes.all.current', 0)),
        'oversized_allocations': stats.get('oversized_allocations.all.current', 0),
        
        # OOM info
        'num_alloc_retries': stats.get('num_alloc_retries', 0),
        'num_ooms': stats.get('num_ooms', 0),
    }


def print_detailed_memory_stats(label: str = "Memory Stats"):
    """
    Print detailed memory breakdown using torch.cuda.memory_stats().
    
    This provides more detailed information than mem_get_info(), including:
    - Active vs inactive memory
    - Memory fragmentation
    - Allocation retries and OOMs
    - Oversized allocations
    
    Args:
        label: Label for the output
    """
    if not torch.cuda.is_available():
        print(f"  [{label}] CUDA not available", file=sys.stderr)
        return
    
    stats = get_detailed_memory_stats()
    if not stats:
        return
    
    print(f"\n  [{label}] Detailed PyTorch Memory Breakdown:", file=sys.stderr)
    print(f"  {'─' * 55}", file=sys.stderr)
    print(f"  Current Allocations:", file=sys.stderr)
    print(f"    Allocated (active tensors):     {stats['allocated_mb']:>8.1f} MB", file=sys.stderr)
    print(f"    Reserved (memory pool):          {stats['reserved_mb']:>8.1f} MB", file=sys.stderr)
    print(f"    Active (in use):                 {stats['active_mb']:>8.1f} MB", file=sys.stderr)
    print(f"    Inactive (cached):               {stats['inactive_split_mb']:>8.1f} MB", file=sys.stderr)
    print(f"    Pool overhead:                   {stats['reserved_mb'] - stats['allocated_mb']:>8.1f} MB", file=sys.stderr)
    print(f"  {'─' * 55}", file=sys.stderr)
    print(f"  Peak Values:", file=sys.stderr)
    print(f"    Peak allocated:                  {stats['allocated_peak_mb']:>8.1f} MB", file=sys.stderr)
    print(f"    Peak reserved:                  {stats['reserved_peak_mb']:>8.1f} MB", file=sys.stderr)
    print(f"  {'─' * 55}", file=sys.stderr)
    print(f"  Cumulative:", file=sys.stderr)
    print(f"    Total allocated:                {stats['allocated_total_mb']:>8.1f} MB", file=sys.stderr)
    print(f"    Total freed:                     {stats['freed_total_mb']:>8.1f} MB", file=sys.stderr)
    print(f"  {'─' * 55}", file=sys.stderr)
    print(f"  Fragmentation:", file=sys.stderr)
    print(f"    Oversized allocations:          {stats['oversized_allocations']:>8d}", file=sys.stderr)
    print(f"    Segment pool:                    {stats['segment_pool_mb']:>8.1f} MB", file=sys.stderr)
    if stats['num_ooms'] > 0 or stats['num_alloc_retries'] > 0:
        print(f"  {'─' * 55}", file=sys.stderr)
        print(f"  Warnings:", file=sys.stderr)
        if stats['num_ooms'] > 0:
            print(f"    OOM errors:                    {stats['num_ooms']:>8d}", file=sys.stderr)
        if stats['num_alloc_retries'] > 0:
            print(f"    Allocation retries:             {stats['num_alloc_retries']:>8d}", file=sys.stderr)
    print(f"  {'─' * 55}", file=sys.stderr)
    print("", file=sys.stderr)


def get_bnb_memory_usage(model):
    """
    Estimate memory usage from BitsAndBytes quantized parameters.

    BitsAndBytes allocates memory outside PyTorch's allocator, so this helps
    account for the "missing" memory in torch.cuda.memory_allocated().

    Args:
        model: A PyTorch model (potentially with BitsAndBytes quantized layers)

    Returns:
        dict with memory breakdown in MB, or None if no BitsAndBytes params found
    """
    total_4bit_params = 0
    total_8bit_params = 0
    total_fp16_params = 0
    quant_state_bytes = 0
    num_4bit_layers = 0
    num_8bit_layers = 0
    max_layer_params = 0  # Track largest layer for dequant buffer estimate

    for name, param in model.named_parameters():
        numel = param.numel()

        # Check for 4-bit quantization (has quant_state attribute)
        if hasattr(param, 'quant_state') and param.quant_state is not None:
            total_4bit_params += numel
            num_4bit_layers += 1
            max_layer_params = max(max_layer_params, numel)

            # Estimate quant_state memory (scales, zeros, etc.)
            qs = param.quant_state
            if hasattr(qs, 'absmax') and qs.absmax is not None:
                quant_state_bytes += qs.absmax.numel() * qs.absmax.element_size()
            if hasattr(qs, 'code') and qs.code is not None:
                quant_state_bytes += qs.code.numel() * qs.code.element_size()
            if hasattr(qs, 'state2') and qs.state2 is not None:
                # Double quantization state
                s2 = qs.state2
                if hasattr(s2, 'absmax') and s2.absmax is not None:
                    quant_state_bytes += s2.absmax.numel() * s2.absmax.element_size()

        # Check for 8-bit quantization
        elif hasattr(param, 'SCB') or (hasattr(param, 'CB') and param.CB is not None):
            total_8bit_params += numel
            num_8bit_layers += 1
            max_layer_params = max(max_layer_params, numel)
        else:
            # Regular parameter (fp16/fp32)
            total_fp16_params += numel

    # Calculate memory usage
    # 4-bit: 0.5 bytes per param
    # 8-bit: 1 byte per param
    # FP16: 2 bytes per param
    mem_4bit_mb = (total_4bit_params * 0.5) / 1024**2
    mem_8bit_mb = (total_8bit_params * 1.0) / 1024**2
    mem_fp16_mb = (total_fp16_params * 2.0) / 1024**2
    mem_quant_state_mb = quant_state_bytes / 1024**2

    # Estimate dequantization buffer: largest quantized layer × 2 bytes (FP16)
    # BnB needs to dequantize weights to FP16 for matmul
    dequant_buffer_mb = (max_layer_params * 2) / 1024**2

    is_quantized = total_4bit_params > 0 or total_8bit_params > 0

    return {
        'num_4bit_params': total_4bit_params,
        'num_8bit_params': total_8bit_params,
        'num_fp16_params': total_fp16_params,
        'num_4bit_layers': num_4bit_layers,
        'num_8bit_layers': num_8bit_layers,
        'mem_4bit_mb': mem_4bit_mb,
        'mem_8bit_mb': mem_8bit_mb,
        'mem_fp16_mb': mem_fp16_mb,
        'mem_quant_state_mb': mem_quant_state_mb,
        'dequant_buffer_mb': dequant_buffer_mb,
        'total_bnb_mb': mem_4bit_mb + mem_8bit_mb + mem_quant_state_mb,
        'total_weights_mb': mem_4bit_mb + mem_8bit_mb + mem_fp16_mb + mem_quant_state_mb,
        'is_quantized': is_quantized,
    }


def print_full_memory_breakdown(label: str = "Memory", baseline_mb: float = None):
    """
    Print comprehensive GPU memory breakdown.

    Uses actual measured GPU memory and classifies allocations into:
    - PyTorch tracked (via memory_allocated)
    - Outside PyTorch (measured delta - PyTorch tracked)

    Per PyTorch docs: memory_allocated() only tracks tensors created through
    PyTorch. External libraries allocate via raw CUDA calls and are NOT
    visible to PyTorch's memory profiler.

    Args:
        label: Label for the output
        baseline_mb: GPU memory used before model load (for delta calculation)
    """
    if not torch.cuda.is_available():
        print(f"  [{label}] CUDA not available", file=sys.stderr)
        return

    # Get GPU-level stats (same as nvidia-smi/tegrastats)
    # Try CUDA runtime first (no PyTorch dependency), fallback to PyTorch
    free_mb, total_mb = get_gpu_memory_cuda_runtime()
    used_mb = total_mb - free_mb
    # if free_mb is None or total_mb is None:
        # Fallback to PyTorch
    free_torch, total_torch = torch.cuda.mem_get_info()
    total_mb_torch = total_torch / 1024**2
    free_mb_torch = free_torch / 1024**2
    print("torch: ", free_mb_torch, total_mb_torch, ", cuda: ", free_mb, total_mb)
    

    # Get PyTorch stats
    pytorch_allocated = torch.cuda.memory_allocated() / 1024**2
    pytorch_reserved = torch.cuda.memory_reserved() / 1024**2

    print(f"\n  [{label}] GPU Memory Breakdown:", file=sys.stderr)
    print(f"  {'─' * 55}", file=sys.stderr)

    # GPU level bar
    bar_len = 30
    filled = int(bar_len * used_mb / total_mb)
    bar = "█" * filled + "░" * (bar_len - filled)
    print(f"  Total GPU Used: [{bar}] {used_mb:.0f}/{total_mb:.0f} MB ({used_mb/total_mb*100:.1f}%)", file=sys.stderr)
    print(f"  {'─' * 55}", file=sys.stderr)

    # Calculate model memory if baseline provided
    if baseline_mb is not None:
        model_memory = used_mb - baseline_mb
        outside_pytorch = model_memory - pytorch_allocated

        print(f"  MEASURED (from GPU):", file=sys.stderr)
        print(f"    Baseline (before load):       {baseline_mb:>8.1f} MB", file=sys.stderr)
        print(f"    Model memory (delta):         {model_memory:>8.1f} MB", file=sys.stderr)
        print(f"  {'─' * 55}", file=sys.stderr)
        print(f"  TRACKED BY PYTORCH:", file=sys.stderr)
        print(f"    memory_allocated():           {pytorch_allocated:>8.1f} MB", file=sys.stderr)
        print(f"    memory_reserved():            {pytorch_reserved:>8.1f} MB", file=sys.stderr)
        print(f"  {'─' * 55}", file=sys.stderr)
        print(f"  OUTSIDE PYTORCH (raw CUDA):     {outside_pytorch:>8.1f} MB", file=sys.stderr)
    else:
        # No baseline, just show current state
        outside_pytorch = used_mb - pytorch_allocated

        print(f"  TRACKED BY PYTORCH:", file=sys.stderr)
        print(f"    memory_allocated():           {pytorch_allocated:>8.1f} MB", file=sys.stderr)
        print(f"    memory_reserved():            {pytorch_reserved:>8.1f} MB", file=sys.stderr)
        print(f"  {'─' * 55}", file=sys.stderr)
        print(f"  OUTSIDE PYTORCH (raw CUDA):     {outside_pytorch:>8.1f} MB", file=sys.stderr)

    print(f"  {'─' * 55}", file=sys.stderr)
    print(f"  Free:                           {free_mb:>8.1f} MB", file=sys.stderr)
    print("", file=sys.stderr)

    return {
        'used_mb': used_mb,
        'free_mb': free_mb,
        'pytorch_allocated_mb': pytorch_allocated,
        'outside_pytorch_mb': outside_pytorch,
    }


def print_memory_status(label: str):
    """Print current memory status for debugging."""
    try:
        # System RAM
        with open('/proc/meminfo', 'r') as f:
            meminfo = f.read()
        for line in meminfo.split('\n'):
            if 'MemAvailable' in line:
                available_kb = int(line.split()[1])
                print(f"  [{label}] System RAM available: {available_kb/1024:.0f} MB", file=sys.stderr)
                break

        # GPU/CUDA memory
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**2
            reserved = torch.cuda.memory_reserved() / 1024**2
            print(f"  [{label}] CUDA allocated: {allocated:.0f} MB, reserved: {reserved:.0f} MB", file=sys.stderr)
    except Exception as e:
        print(f"  [{label}] Memory check failed: {e}", file=sys.stderr)
