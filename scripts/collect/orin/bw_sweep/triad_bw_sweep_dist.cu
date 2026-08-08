// triad_bw_sweep.cu
// Buffer-size sweep of achievable GPU DRAM bandwidth via a STREAM-style triad
// (a[i] = b[i] + scalar * c[i]) on Jetson (Orin sm_87 / Thor sm_110).
//
// Purpose: settle whether the paper's measured S_BW (Orin->Thor) is a real
// hardware ratio or an artifact of an undersized working set on Thor. We sweep
// the per-array size from below to well above the last-level cache and report
// the bandwidth plateau. The plateau value is the one to feed into Eq. (3).
//
// Bytes-moved accounting is reported under BOTH conventions:
//   * STREAM (3N): read b, read c, write a           -> the classic figure
//   * write-allocate (4N): the write to `a` also pulls the line into cache
// so the reader can see the up-to-1.33x ambiguity explicitly.
//
// Build (auto-targets the present GPU; CUDA >= 11.5):
//   nvcc -O3 -arch=native -o triad_bw_sweep triad_bw_sweep.cu
// Run:
//   ./triad_bw_sweep [--peak <GB/s>] [--max-mb <N>] [--trials <N>] [--csv]
//
// Notes:
//   * Lock clocks first:  sudo jetson_clocks   (and nvpmodel -m 0)
//   * float4 vectorized, grid-stride; timing via CUDA events, best-of-trials.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t _e = (call);                                                   \
    if (_e != cudaSuccess) {                                                   \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,            \
              cudaGetErrorString(_e));                                         \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

__global__ void fill_kernel(float4 *p, size_t n4, float v) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = (size_t)gridDim.x * blockDim.x;
  for (; i < n4; i += stride) p[i] = make_float4(v, v, v, v);
}

__global__ void triad_kernel(float4 *__restrict__ a,
                             const float4 *__restrict__ b,
                             const float4 *__restrict__ c,
                             float scalar, size_t n4) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = (size_t)gridDim.x * blockDim.x;
  for (; i < n4; i += stride) {
    float4 bb = b[i];
    float4 cc = c[i];
    float4 aa;
    aa.x = bb.x + scalar * cc.x;
    aa.y = bb.y + scalar * cc.y;
    aa.z = bb.z + scalar * cc.z;
    aa.w = bb.w + scalar * cc.w;
    a[i] = aa;
  }
}

int main(int argc, char **argv) {
  double peak = 0.0;       // spec peak GB/s, for %-of-peak (0 = unknown)
  size_t max_mb = 2048;    // max per-array size (MB)
  int trials = 7;          // outer repeats; we keep the best (min time)
  bool csv = false;

  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "--peak") && i + 1 < argc) peak = atof(argv[++i]);
    else if (!strcmp(argv[i], "--max-mb") && i + 1 < argc) max_mb = (size_t)atoll(argv[++i]);
    else if (!strcmp(argv[i], "--trials") && i + 1 < argc) trials = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--csv")) csv = true;
    else { fprintf(stderr, "unknown arg: %s\n", argv[i]); return 2; }
  }

  int dev = 0;
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDevice(&dev));
  CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));

  // theoretical peak from device attributes (2 = DDR), for cross-check with
  // --peak. NOTE: CUDA 13 removed cudaDeviceProp::memoryClockRate/memoryBusWidth,
  // so query via cudaDeviceGetAttribute (portable across CUDA 12 and 13).
  int mem_clock_khz = 0, mem_bus_bits = 0;
  cudaDeviceGetAttribute(&mem_clock_khz, cudaDevAttrMemoryClockRate, dev);
  cudaDeviceGetAttribute(&mem_bus_bits, cudaDevAttrGlobalMemoryBusWidth, dev);
  double prop_peak = 2.0 * mem_clock_khz * 1e3 /*kHz->Hz*/ *
                     (mem_bus_bits / 8.0) / 1e9;

  if (!csv) {
    fprintf(stderr, "# Device: %s (sm_%d%d), %d SMs, L2=%.1f MB\n",
            prop.name, prop.major, prop.minor, prop.multiProcessorCount,
            prop.l2CacheSize / (1024.0 * 1024.0));
    fprintf(stderr, "# memClock=%.0f MHz busWidth=%d bits -> prop peak=%.1f GB/s\n",
            mem_clock_khz / 1e3, mem_bus_bits, prop_peak);
    if (peak <= 0) peak = prop_peak;
    fprintf(stderr, "# using peak = %.1f GB/s for %%-of-peak\n#\n", peak);
  } else if (peak <= 0) {
    peak = prop_peak;
  }

  // per-array sizes (MB): span from below L2 to well above it
  // per-array sizes (MB). Note: total triad footprint = 3x this.
  // 85 MB/array => 255 MB total ~= the paper's "256 MB" if that label is the
  // aggregate 3-array working set rather than per-array.
  std::vector<size_t> sizes_mb = {2,  4,   8,   16,  24,  32,   48,   64,
                                  85, 96,  128, 170, 256, 341,  512,  1024,
                                  2048};

  const int threads = 256;
  const float scalar = 3.0f;

  if (csv)
    printf("per_array_mb,total_footprint_mb,n_kernels,"
           "bw_sustained_gbs,bw_mean_gbs,bw_median_gbs,bw_best_gbs,"
           "bw_worst_gbs,pct_peak_sustained\n");
  else
    printf("%8s %10s %5s %10s %10s %10s %10s %10s %8s\n", "arr_MB",
           "foot_MB", "nK", "SUSTAIN", "mean", "median", "best", "worst",
           "%pk(sus)");

  for (size_t mb : sizes_mb) {
    if (mb > max_mb) break;
    size_t bytes = mb * 1024ull * 1024ull;
    size_t N = bytes / sizeof(float);   // floats per array
    size_t n4 = N / 4;                  // float4 elements
    N = n4 * 4;                         // round to float4
    bytes = N * sizeof(float);

    float4 *a, *b, *c;
    CUDA_CHECK(cudaMalloc(&a, bytes));
    CUDA_CHECK(cudaMalloc(&b, bytes));
    CUDA_CHECK(cudaMalloc(&c, bytes));

    int blocks = (int)std::min<size_t>((n4 + threads - 1) / threads,
                                       (size_t)prop.multiProcessorCount * 512);

    // first-touch + init (UMA pages become resident here)
    fill_kernel<<<blocks, threads>>>(b, n4, 1.0f);
    fill_kernel<<<blocks, threads>>>(c, n4, 2.0f);
    fill_kernel<<<blocks, threads>>>(a, n4, 0.0f);
    CUDA_CHECK(cudaDeviceSynchronize());

    // warmup
    for (int w = 0; w < 8; w++)
      triad_kernel<<<blocks, threads>>>(a, b, c, scalar, n4);
    CUDA_CHECK(cudaDeviceSynchronize());

    // ONE back-to-back burst of M kernels, an event between every kernel so we
    // recover the per-kernel time distribution AND the sustained rate from the
    // same run. `trials` scales M (kept for CLI compat). Sustained = the honest
    // number to compare against the paper; best = the old optimistic estimator.
    int M = (int)std::min<size_t>(
        std::max<size_t>(2ull * 1000000000ull / bytes, (size_t)(20 * trials)),
        800ull);

    std::vector<cudaEvent_t> ev(M + 1);
    for (int j = 0; j <= M; j++) CUDA_CHECK(cudaEventCreate(&ev[j]));

    CUDA_CHECK(cudaEventRecord(ev[0]));
    for (int j = 0; j < M; j++) {
      triad_kernel<<<blocks, threads>>>(a, b, c, scalar, n4);
      CUDA_CHECK(cudaEventRecord(ev[j + 1]));
    }
    CUDA_CHECK(cudaEventSynchronize(ev[M]));

    std::vector<double> ms(M);
    for (int j = 0; j < M; j++) {
      float e;
      CUDA_CHECK(cudaEventElapsedTime(&e, ev[j], ev[j + 1]));
      ms[j] = e;
    }
    float burst_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&burst_ms, ev[0], ev[M]));
    for (int j = 0; j <= M; j++) cudaEventDestroy(ev[j]);

    std::vector<double> srt = ms;
    std::sort(srt.begin(), srt.end());
    double t_best = srt.front();            // min time  -> peak BW (optimistic)
    double t_worst = srt.back();            // max time  -> worst BW
    double t_med = srt[M / 2];               // median
    double t_mean = 0;
    for (double x : ms) t_mean += x;
    t_mean /= M;
    double t_sust = (double)burst_ms / M;    // whole-burst avg -> SUSTAINED

    // verify (also prevents dead-code elimination)
    float host_a;
    CUDA_CHECK(cudaMemcpy(&host_a, a, sizeof(float), cudaMemcpyDeviceToHost));
    double expect = 1.0 + scalar * 2.0;  // b + s*c = 1 + 3*2 = 7
    if (host_a != (float)expect) {
      fprintf(stderr, "WARN: verify failed at %zu MB: got %f want %f\n", mb,
              host_a, expect);
    }

    // all bandwidths under the STREAM 3N convention (2 read + 1 write)
    double bytes3 = 3.0 * (double)bytes;
    auto bw = [&](double t_ms) { return bytes3 / (t_ms / 1e3) / 1e9; };
    double bw_sust = bw(t_sust), bw_mean = bw(t_mean), bw_med = bw(t_med),
           bw_best = bw(t_best), bw_worst = bw(t_worst);
    double pct = peak > 0 ? 100.0 * bw_sust / peak : 0.0;  // headline=sustained

    if (csv)
      printf("%zu,%zu,%d,%.2f,%.2f,%.2f,%.2f,%.2f,%.1f\n", mb, 3 * mb, M,
             bw_sust, bw_mean, bw_med, bw_best, bw_worst, pct);
    else
      printf("%8zu %10zu %5d %10.2f %10.2f %10.2f %10.2f %10.2f %8.1f\n", mb,
             3 * mb, M, bw_sust, bw_mean, bw_med, bw_best, bw_worst, pct);
    fflush(stdout);

    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
  }

  if (!csv)
    fprintf(stderr,
            "#\n# Quote SUSTAIN (back-to-back burst avg) as S_BW, NOT best.\n"
            "# 'best' is the min-time / peak estimator (optimistic ~5-8%%); it is\n"
            "# shown only to expose the peak-vs-sustained gap. mean~=median~=\n"
            "# sustained on an uncontended, clock-locked box.\n");
  return 0;
}
