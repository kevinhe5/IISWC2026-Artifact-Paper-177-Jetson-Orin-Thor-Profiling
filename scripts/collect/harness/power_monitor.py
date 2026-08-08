#!/usr/bin/env python3
"""Tegrastats power monitoring for Jetson, with per-sample timestamps,
time-window slicing, AGX 4-rail breakdown (incl. LPDDR5 DRAM cell rail
read directly from hwmon since tegrastats omits it), and PowerTracer
stub for backward compat with the pytorch profiler.

This is the "post-powerfix" version of the API: each tegrastats sample
gets a wall-clock timestamp on arrival, and `get_power_breakdown()`
optionally filters to a [start_ns, end_ns) window so callers can compute
per-phase (idle / prefill / decode) means without conflating phases.
"""
import glob
import re
import subprocess
import sys
import threading
import time


# ---------------------------------------------------------------------------
# DRAM cell rail (VDDQ_VDD2_1V8AO) — AGX Orin only, from a SECOND INA3221
# that tegrastats does not poll. Located at /sys/class/hwmon/hwmon*/ where
# in2_label == "VDDQ_VDD2_1V8AO". On Orin Nano this rail does not exist.
# ---------------------------------------------------------------------------
def _find_dram_rail_paths():
    """Return (voltage_path, current_path) for VDDQ_VDD2_1V8AO, or (None, None)."""
    for label_path in glob.glob('/sys/class/hwmon/hwmon*/in*_label'):
        try:
            label = open(label_path).read().strip()
        except OSError:
            continue
        if label != 'VDDQ_VDD2_1V8AO':
            continue
        # Derive the matching curr*_input from the in*_label index
        m = re.search(r'in(\d+)_label$', label_path)
        if not m:
            continue
        idx = m.group(1)
        base = label_path.rsplit('/', 1)[0]
        v = f'{base}/in{idx}_input'
        c = f'{base}/curr{idx}_input'
        return v, c
    return None, None


_DRAM_V_PATH, _DRAM_C_PATH = _find_dram_rail_paths()


def _read_dram_mw():
    """Returns DRAM rail power in mW, or 0 if rail unavailable / unreadable."""
    if not _DRAM_V_PATH:
        return 0
    try:
        v_mv = int(open(_DRAM_V_PATH).read().strip())   # bus voltage, mV
        c_ma = int(open(_DRAM_C_PATH).read().strip())   # current, mA
        # mV * mA = µW; / 1000 = mW
        return (v_mv * c_ma) // 1000
    except (OSError, ValueError):
        return 0


class TegrastatsMonitor:
    """Background tegrastats sampler with timestamped samples and
    time-window aware aggregation.

    Each sample dict carries:
      ts_ns              : wall-clock time of this sample (time.time_ns())
      vdd_in             : VIN_SYS_5V0 (AGX) or VDD_IN (Nano), mW
      vdd_cpu_gpu_cv     : VDD_GPU_SOC (AGX) or VDD_CPU_GPU_CV (Nano), mW
      vdd_soc            : VDD_CPU_CV (AGX) or VDD_SOC (Nano), mW
      dram               : VDDQ_VDD2_1V8AO (AGX only, from hwmon), mW
      gpu_util, cpu_util, emc_util, emc_freq, emc_bw_gb_s
      gpu_temp, cpu_temp
      ram_used_mb, ram_total_mb, ram_use/total
    """

    def __init__(self, interval_ms: int = 1):
        # tegrastats clamps below ~100 ms; we accept any int and let it clamp.
        self.interval_ms = max(interval_ms, 1)
        self.process = None
        self.samples = []          # list of dicts with ts_ns + raw fields
        self._reader_thread = None
        self._stop_event = threading.Event()

    # -- Lifecycle ---------------------------------------------------------
    def __enter__(self):
        for cmd in ['tegrastats', '/usr/bin/tegrastats']:
            try:
                self.process = subprocess.Popen(
                    [cmd, '--readall', '--interval', str(self.interval_ms)],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    text=True, bufsize=1,
                )
                self._stop_event.clear()
                self._reader_thread = threading.Thread(target=self._read_output, daemon=True)
                self._reader_thread.start()
                # Brief warmup so tegrastats is producing output before we measure idle.
                time.sleep(0.1)
                self.samples.clear()
                return self
            except FileNotFoundError:
                continue
            except Exception as e:
                print(f"Warning: Could not start tegrastats: {e}", file=sys.stderr)
                return self
        if not hasattr(TegrastatsMonitor, '_warned'):
            print("Warning: tegrastats not found (not running on Jetson?)", file=sys.stderr)
            TegrastatsMonitor._warned = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.process:
            time.sleep(0.05)        # let final samples land
            self._stop_event.set()
            self.process.terminate()
            try:
                self.process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                self.process.kill()
            if self._reader_thread:
                self._reader_thread.join(timeout=1)
        return False

    # -- Reading & parsing -------------------------------------------------
    def _read_output(self):
        if not self.process:
            return
        try:
            for line in iter(self.process.stdout.readline, ''):
                if self._stop_event.is_set():
                    break
                if not line.strip():
                    continue
                sample = self._parse_line(line)
                if sample:
                    # Match the bench's clock: prefill_start_ns / decode_start_ns
                    # are taken with time.perf_counter_ns(), so we must use the
                    # same monotonic clock for sample timestamps to land in the
                    # right window.
                    sample['ts_ns'] = time.perf_counter_ns()
                    sample['dram'] = _read_dram_mw()
                    self.samples.append(sample)
        except Exception as e:
            if not self._stop_event.is_set():
                print(f"Warning: tegrastats reader error: {e}", file=sys.stderr)

    def _parse_line(self, line: str) -> dict:
        sample = {}
        power_patterns = [
            ('vdd_in',          r'(?:VDD_IN|VIN_SYS_5V0)\s+(\d+)mW'),
            # GPU rail: Orin=VDD_GPU_SOC (GPU+SOC fabric), Thor=VDD_GPU (GPU only),
            # Nano=VDD_CPU_GPU_CV. The whole Orin name VDD_GPU_SOC must precede the
            # Thor prefix VDD_GPU in the alternation so it is matched intact on Orin.
            ('vdd_cpu_gpu_cv',  r'(?:VDD_CPU_GPU_CV|VDD_GPU_SOC|VDD_GPU)\s+(\d+)mW'),
            # CPU rail: Orin=VDD_CPU_CV, Thor=VDD_CPU_SOC_MSS (CPU+SOC+mem subsystem),
            # Nano=VDD_SOC. The specific Thor name is listed first.
            ('vdd_soc',         r'(?:VDD_CPU_SOC_MSS|VDD_SOC|VDD_CPU_CV)\s+(\d+)mW'),
        ]
        for name, pattern in power_patterns:
            m = re.search(pattern, line)
            if m:
                sample[name] = int(m.group(1))
        m = re.search(r'GR3D_FREQ\s+(\d+)%', line)
        if m:
            sample['gpu_util'] = int(m.group(1))
        m = re.search(r'EMC(?:_FREQ)?\s+(\d+)%@(\d+)', line)
        if m:
            sample['emc_util'] = int(m.group(1))
            sample['emc_freq'] = int(m.group(2))
            sample['emc_bw_gb_s'] = sample['emc_util'] / 100.0 * (sample['emc_freq'] * 32 / 1000)
        m = re.search(r'CPU\s+\[([^\]]+)\]', line)
        if m:
            utils = [int(x.group(1)) for c in m.group(1).split(',')
                     for x in [re.search(r'(\d+)%', c)] if x]
            if utils:
                sample['cpu_util'] = sum(utils) / len(utils)
        m = re.search(r'gpu@([\d.]+)C', line, re.IGNORECASE)
        if m:
            sample['gpu_temp'] = float(m.group(1))
        m = re.search(r'cpu@([\d.]+)C', line, re.IGNORECASE)
        if m:
            sample['cpu_temp'] = float(m.group(1))
        m = re.search(r'RAM\s+(\d+)/(\d+)MB', line)
        if m:
            ru, rt = int(m.group(1)), int(m.group(2))
            sample['ram_used_mb'] = ru
            sample['ram_total_mb'] = rt
            sample['ram_use/total'] = (ru / rt) * 100 if rt > 0 else 0
        return sample if sample else None

    # -- Aggregation -------------------------------------------------------
    def _slice(self, start_ns=None, end_ns=None):
        if start_ns is None and end_ns is None:
            return self.samples
        s = self.samples
        if start_ns is not None:
            s = [x for x in s if x.get('ts_ns', 0) >= start_ns]
        if end_ns is not None:
            s = [x for x in s if x.get('ts_ns', float('inf')) <= end_ns]
        return s

    @staticmethod
    def _avg(samples, key):
        vs = [s[key] for s in samples if key in s]
        return sum(vs) / len(vs) if vs else 0

    @staticmethod
    def _max(samples, key):
        vs = [s[key] for s in samples if key in s]
        return max(vs) if vs else 0

    def get_power_breakdown(self, start_ns=None, end_ns=None) -> dict:
        """Mean/max of every recorded field, optionally restricted to a
        [start_ns, end_ns] window. Returns BOTH the legacy field names and
        the new named-rail fields the post-powerfix bench scripts expect."""
        samples = self._slice(start_ns, end_ns)
        n = len(samples)

        emc_freq = int(self._avg(samples, 'emc_freq')) or 3199
        peak_bw  = emc_freq * 32 / 1000   # GB/s

        # Tegrastats-mapped semantic rails:
        #   AGX Orin: vdd_in=VIN_SYS_5V0, vdd_cpu_gpu_cv=VDD_GPU_SOC, vdd_soc=VDD_CPU_CV,
        #             dram=VDDQ_VDD2_1V8AO (separate LPDDR5 cell rail via hwmon).
        #   AGX Thor: vdd_in=VIN_SYS_5V0, vdd_cpu_gpu_cv=VDD_GPU (GPU only),
        #             vdd_soc=VDD_CPU_SOC_MSS (CPU+SOC+mem subsystem). Thor exposes
        #             NO separate DRAM rail, so dram=0 and total4==total (the DRAM
        #             draw is folded into VDD_CPU_SOC_MSS).
        #   Orin Nano: vdd_in=VDD_IN (true total), other two as named.
        gpu_mw  = int(self._avg(samples, 'vdd_cpu_gpu_cv'))   # Orin: GPU+SOC fabric / Thor: GPU
        cpu_mw  = int(self._avg(samples, 'vdd_soc'))          # Orin: CPU cluster / Thor: CPU+SOC+mem
        soc_mw  = int(self._avg(samples, 'vdd_in'))           # VIN_SYS_5V0 (peripheral 5V) both
        dram_mw = int(self._avg(samples, 'dram'))             # Orin-only LPDDR5 cell rail (0 on Thor)
        total_mw  = gpu_mw + cpu_mw + soc_mw                  # 3 tegrastats rails
        total4_mw = total_mw + dram_mw                        # Full 4-rail board power

        # Heuristic: if no samples landed in the requested window (which can
        # happen with very short phases at coarse tegrastats interval), warn
        # so the bench can flag the row.
        samples_warning = (n == 0)

        return {
            # Named per-rail (post-powerfix names — what build_power_dict reads):
            'gpu_mw':    gpu_mw,
            'cpu_mw':    cpu_mw,
            'soc_mw':    soc_mw,
            'dram_mw':   dram_mw,
            'total_mw':  total_mw,
            'total4_mw': total4_mw,

            # Legacy aliases (older bench code paths still read these):
            'vdd_in_mw':         soc_mw,                                # was VIN_SYS_5V0 / VDD_IN
            'vdd_cpu_gpu_cv_mw': gpu_mw,                                # was VDD_GPU_SOC / VDD_CPU_GPU_CV
            'vdd_soc_mw':        cpu_mw,                                # was VDD_CPU_CV / VDD_SOC
            'vdd_in_max_mw':     int(self._max(samples, 'vdd_in')),

            # Utilization / EMC:
            'gpu_util_pct':      int(self._avg(samples, 'gpu_util')),
            'gpu_util_max_pct':  int(self._max(samples, 'gpu_util')),
            'cpu_util_pct':      int(self._avg(samples, 'cpu_util')),
            'cpu_util_max_pct':  int(self._max(samples, 'cpu_util')),
            'emc_util_pct':      int(self._avg(samples, 'emc_util')),
            'emc_util_max_pct':  int(self._max(samples, 'emc_util')),
            'emc_freq_mhz':      emc_freq,
            'emc_bw_gb_s':       round(self._avg(samples, 'emc_bw_gb_s'), 2),
            'emc_bw_max_gb_s':   round(self._max(samples, 'emc_bw_gb_s'), 2),
            'emc_peak_bw_gb_s':  round(peak_bw, 1),

            # Misc:
            'ram_use/total':     f"{int(self._avg(samples, 'ram_used_mb'))}/{int(self._avg(samples, 'ram_total_mb'))}MB",
            'gpu_temp_c':        round(self._avg(samples, 'gpu_temp'), 1),
            'cpu_temp_c':        round(self._avg(samples, 'cpu_temp'), 1),
            'samples':           n,
            'samples_warning':   samples_warning,
        }

    def get_avg_power(self) -> int:
        """Approximate total board power in mW (4 rails for AGX, single rail for Nano)."""
        bd = self.get_power_breakdown()
        return bd['total4_mw'] or bd['vdd_in_mw']


class PowerTracer:
    """No-op stub for compatibility with profiler_pytorch/bench_e2e.py
    (the pytorch bench drives a finer-grained per-token tracer; sweep mode
    doesn't need the trace, so the stub just absorbs the calls)."""

    def __init__(self, interval_ms: int = 1):
        self.interval_ms = interval_ms

    def start(self): pass
    def stop(self): pass
    def mark_phase(self, name: str): pass
    def save_csv(self, path: str): pass
    def get_phase_stats(self): return {}
    def get_phases(self): return []
    def plot(self, *a, **kw): pass
