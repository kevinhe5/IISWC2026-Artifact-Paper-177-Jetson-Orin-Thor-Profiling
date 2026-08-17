#!/usr/bin/env python3
"""
Memory analysis utilities for Jetson GPU profiling.
"""
import os
import subprocess
import torch


def identify_process(comm, cmdline):
    """Identify what a process is based on comm and cmdline."""
    # VS Code related node processes
    if comm == 'node' and '.vscode-server' in cmdline:
        if 'extensionHost' in cmdline:
            return 'vscode-extensions'
        elif 'ptyHost' in cmdline:
            return 'vscode-terminal'
        elif 'fileWatcher' in cmdline:
            return 'vscode-filewatcher'
        elif 'server-main' in cmdline:
            return 'vscode-server'
        else:
            return 'vscode-node'

    # Other common processes
    process_descriptions = {
        'containerd': 'container-runtime',
        'dockerd': 'docker-daemon',
        'snapd': 'snap-daemon',
        'tailscaled': 'tailscale-vpn',
        'NetworkManager': 'network-mgr',
        'systemd': 'init-system',
        'polkitd': 'policy-kit',
        'Xorg': 'x-server',
        'gnome-shell': 'gnome-desktop',
        'nvargus-daemon': 'jetson-camera',
        'jetson_clocks': 'jetson-clocks',
    }

    return process_descriptions.get(comm, '')


def analyze_gpu_memory_usage():
    """Analyze and explain what's consuming GPU memory at startup."""
    print("\n" + "=" * 60)
    print("  GPU Memory Analysis - Why is memory already used?")
    print("=" * 60)

    # Get memory info BEFORE any CUDA operations
    with open('/proc/meminfo', 'r') as f:
        meminfo = f.read()

    mem_total_kb = 0
    mem_avail_before_kb = 0
    for line in meminfo.split('\n'):
        if line.startswith('MemTotal:'):
            mem_total_kb = int(line.split()[1])
        if line.startswith('MemAvailable:'):
            mem_avail_before_kb = int(line.split()[1])

    mem_used_before_mb = (mem_total_kb - mem_avail_before_kb) / 1024

    print(f"\n[System Memory BEFORE CUDA init]")
    print(f"  Total RAM:     {mem_total_kb/1024:.0f} MB")
    print(f"  Available:     {mem_avail_before_kb/1024:.0f} MB")
    print(f"  Used:          {mem_used_before_mb:.0f} MB")

    # Detect if running inside Docker
    in_docker = False
    try:
        with open('/proc/1/cgroup', 'r') as f:
            in_docker = 'docker' in f.read() or 'containerd' in f.read()
    except:
        pass
    if not in_docker:
        try:
            in_docker = os.path.exists('/.dockerenv')
        except:
            pass

    # Try to get host processes from inside container
    host_proc_path = None

    if in_docker:
        print(f"\n[Running inside Docker container - attempting to read HOST processes]")

        # Method 1: Check if /host/proc is mounted
        if os.path.exists('/host/proc'):
            host_proc_path = '/host/proc'
            print(f"  Found /host/proc mount")

        # Method 2: Check if running with --pid=host (PID 1 would be systemd, not bash)
        elif os.path.exists('/proc/1/comm'):
            with open('/proc/1/comm', 'r') as f:
                init_name = f.read().strip()
            if init_name in ['systemd', 'init']:
                host_proc_path = '/proc'
                print(f"  Container using --pid=host (PID 1 is {init_name})")

        # Method 3: Try reading /proc directly and look for high PIDs (host processes)
        if not host_proc_path:
            # Check if we can see processes with PIDs > 1000 (likely host processes)
            try:
                pids = [int(p) for p in os.listdir('/proc') if p.isdigit()]
                if max(pids) > 10000:  # Host usually has many more PIDs
                    host_proc_path = '/proc'
                    print(f"  Detected shared PID namespace (max PID: {max(pids)})")
            except:
                pass

    if host_proc_path or not in_docker:
        proc_path = host_proc_path if host_proc_path else '/proc'
        print(f"\n[Top processes consuming memory (from {'host' if in_docker else 'system'})]")

        # Read process info directly from /proc
        processes = []
        try:
            for pid in os.listdir(proc_path):
                if not pid.isdigit():
                    continue
                try:
                    # Read process name
                    with open(f'{proc_path}/{pid}/comm', 'r') as f:
                        comm = f.read().strip()[:20]

                    # Read cmdline for more context (especially for node processes)
                    cmdline = ""
                    try:
                        with open(f'{proc_path}/{pid}/cmdline', 'r') as f:
                            cmdline = f.read().replace('\x00', ' ')
                    except:
                        pass

                    # Read memory info from status
                    with open(f'{proc_path}/{pid}/status', 'r') as f:
                        status = f.read()

                    rss_kb = 0
                    for line in status.split('\n'):
                        if line.startswith('VmRSS:'):
                            rss_kb = int(line.split()[1])
                            break

                    if rss_kb > 1024:  # Only show processes using > 1MB
                        processes.append((rss_kb, pid, comm, cmdline))
                except (FileNotFoundError, PermissionError, ProcessLookupError):
                    continue

            # Sort by RSS descending
            processes.sort(reverse=True)

            print(f"  {'PID':>7}  {'COMMAND':<20} {'RSS(MB)':>10}  {'TYPE':<18}")
            print(f"  {'-'*7}  {'-'*20} {'-'*10}  {'-'*18}")

            # Track VS Code total
            vscode_total_kb = 0

            for rss_kb, pid, comm, cmdline in processes[:15]:  # Top 15
                rss_mb = rss_kb / 1024
                proc_type = identify_process(comm, cmdline)
                if proc_type.startswith('vscode'):
                    vscode_total_kb += rss_kb
                print(f"  {pid:>7}  {comm:<20} {rss_mb:>10.1f}  {proc_type:<18}")

            # Sum up total
            total_rss = sum(p[0] for p in processes) / 1024
            print(f"  {'-'*7}  {'-'*20} {'-'*10}  {'-'*18}")
            print(f"  {'TOTAL':>7}  {'':<20} {total_rss:>10.1f}")

            # Show VS Code summary if present
            if vscode_total_kb > 0:
                print(f"\n  Note: VS Code Server processes: {vscode_total_kb/1024:.1f} MB total")

        except Exception as e:
            print(f"  Error reading /proc: {e}")

    else:
        # Can't access host processes
        print(f"\n[Cannot access host processes from container]")
        print(f"  Container is isolated. To see host processes either:")
        print(f"  1. Run container with: --pid=host")
        print(f"  2. Mount host proc:    -v /proc:/host/proc:ro")
        print(f"  3. Run on host:        ps aux --sort=-%mem | head -20")
        print(f"")
        print(f"  Common Jetson host memory consumers:")
        print(f"    - Xorg / gnome-shell (desktop: ~200-500 MB)")
        print(f"    - dockerd / containerd (~100-300 MB)")
        print(f"    - nvargus-daemon (camera services)")
        print(f"    - systemd and kernel (~200-400 MB)")

        # Show container processes
        print(f"\n[Container processes only]")
        try:
            result = subprocess.run(
                ["ps", "-eo", "pid,comm,%mem,rss", "--sort=-rss"],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                print(f"  {'PID':>7}  {'COMMAND':<20} {'%MEM':>6} {'RSS(MB)':>10}")
                print(f"  {'-'*7}  {'-'*20} {'-'*6} {'-'*10}")
                for line in lines[1:11]:
                    parts = line.split()
                    if len(parts) >= 4:
                        pid = parts[0]
                        comm = parts[1][:20]
                        mem_pct = parts[2]
                        rss_kb = int(parts[3])
                        rss_mb = rss_kb / 1024
                        print(f"  {pid:>7}  {comm:<20} {mem_pct:>6} {rss_mb:>10.1f}")
        except Exception as e:
            print(f"  Error: {e}")

    # Now initialize CUDA and measure the actual cost
    print(f"\n[CUDA Context Initialization]")

    # Get memory right before CUDA init
    with open('/proc/meminfo', 'r') as f:
        meminfo = f.read()
    for line in meminfo.split('\n'):
        if line.startswith('MemAvailable:'):
            mem_avail_pre_cuda_kb = int(line.split()[1])
            break

    # Force CUDA initialization
    torch.cuda.init()
    torch.cuda.synchronize()

    # Get memory right after CUDA init
    with open('/proc/meminfo', 'r') as f:
        meminfo = f.read()
    for line in meminfo.split('\n'):
        if line.startswith('MemAvailable:'):
            mem_avail_post_cuda_kb = int(line.split()[1])
            break

    cuda_init_cost_mb = (mem_avail_pre_cuda_kb - mem_avail_post_cuda_kb) / 1024

    print(f"  Memory before torch.cuda.init(): {mem_avail_pre_cuda_kb/1024:.0f} MB available")
    print(f"  Memory after torch.cuda.init():  {mem_avail_post_cuda_kb/1024:.0f} MB available")
    print(f"  CUDA init consumed:              {cuda_init_cost_mb:.0f} MB")
    print(f"")

    # Final summary
    print(f"\n[Memory Breakdown Summary]")
    with open('/proc/meminfo', 'r') as f:
        for line in f:
            if line.startswith('MemAvailable:'):
                mem_avail_now_kb = int(line.split()[1])
                break

    total_consumed = (mem_avail_before_kb - mem_avail_now_kb) / 1024
    pre_python_used = mem_used_before_mb

    free_now, total_unified = torch.cuda.mem_get_info()

    print(f"  ┌────────────────────────────────────────────────────┐")
    print(f"  │ Memory used BEFORE this script:     {pre_python_used:>6.0f} MB      │")
    print(f"  │ CUDA context init cost:             {cuda_init_cost_mb:>6.0f} MB      │")
    print(f"  ├────────────────────────────────────────────────────┤")
    print(f"  │ System RAM available now:           {mem_avail_now_kb/1024:>6.0f} MB      │")
    print(f"  │ PyTorch unified pool free:          {free_now/1024**2:>6.0f} MB      │")
    print(f"  └────────────────────────────────────────────────────┘")


    return total_consumed
