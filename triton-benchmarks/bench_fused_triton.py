#!/usr/bin/env python3
"""
Step-2 Fusion probe: Fused vs Sequential Triton vs TensorForge
for batched small GEMMs – testing on sm_75 locally, targeting sm_90a on Vista.

Compares:
  1. TensorForge Sequential: 3x calls to C += A_i @ B_i (accumulating)
  2. Triton Sequential:      3x calls to C += A_i @ B_i
  3. Triton Fused:           1x call  to C = sum(A_i @ B_i)

Usage
-----
  python bench_fused_triton.py [--arch sm_90a] [--batch 100000]
                               [--warmup 20] [--repeats 200]
"""

import argparse
import ctypes
import math
import os
import subprocess
import sys
import tempfile
import json
import contextlib

import numpy as np
import torch

# ─── optional imports ───────────────────────────────────────────────────────

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    print("WARNING: triton not found. Triton benchmarks will be skipped.")

try:
    from tensorforge.generators.generator import Generator as TFGenerator
    from tensorforge.generators.descriptions import GemmDescr
    from tensorforge.common.matrix.tensor import Tensor as TFTensor, SubTensor
    from tensorforge.common.basic_types import Addressing, FloatingPointType
    from tensorforge.common.context import Context as TFContext
    HAS_TF = True
except ImportError:
    HAS_TF = False
    print("WARNING: tensorforge not found. TensorForge benchmarks will be skipped.")

# ─── DG test cases ───────────────────────────────────────────────────────────
# Sizes are taken directly from a real SeisSol order-6 elastic build
# (build.order6.f8.single/src/generated_code/equation-elastic-6-single-f8/tensor.h).
#
# Three kernel families dominate:
#
#  1. Volume / ADER time integration:  C[dof, 9] += K[dof, dof] * Q[dof, 9]
#     K is the stiffness matrix (56×56); Q is the solution (56×9, 9 = elastic vars).
#     Shape pattern: (M=dof, N=9, K=dof). Most common kernel, highest arithmetic intensity.
#
#  2. Flux / face projection:  C[dof, 21] += P[dof, face_dof] * F[face_dof, 21]
#     P[56,21] projects from face basis (21 = ord5 face dof) into volume basis.
#     Shape pattern: (M=dof, N=21, K=face_dof).
#
#  3. ADER CK sub-steps:  dQ[dof_sub, 9] += E[dof_sub, dof] * Q[dof, 9]
#     dof_sub grows each sub-step: 1,4,10,20,35,56 (triangular numbers).
#     At full order the shape is the same as family 1, but intermediate steps
#     have very small M (e.g. 1×9×56, 4×9×56 ...).  We include a few.
#
# face_dof for order O = (O+1)(O+2)/2:  ord2→6, ord3→10, ord4→15, ord5→21, ord6→28, ord7→36

DG_SIZES = [
    # ── Family 1: volume / time-integration  C[dof,9] += K[dof,dof] * Q[dof,9]
    ("vol ord2  (10×9×10) ", 10,   9,  10),
    ("vol ord3  (20×9×20) ", 20,   9,  20),
    ("vol ord4  (35×9×35) ", 35,   9,  35),
    ("vol ord5  (56×9×56) ", 56,   9,  56),
    ("vol ord6  (84×9×84) ", 84,   9,  84),
    ("vol ord7 (120×9×120)", 120,  9, 120),
    # ── Family 2: flux / face projection  C[dof,face_dof] += P * F
    ("flux ord5  (56×21×21)", 56,  21,  21),
    ("flux ord6  (84×28×28)", 84,  28,  28),
    ("flux ord7 (120×36×36)", 120, 36,  36),
    # ── Family 3: ADER CK full-order step  C[dof,9] += E[dof,dof] * dQ[dof,9]
    # Same shape as family 1 at the top order — included to show ADER cost.
    # Square sub-steps (dof×dof×dof) only occur in the Cauchy-Kovalevski
    # intermediate derivatives, not in the dominant volume integration.
    ("ader ord5  (56×56×56)", 56,  56,  56),
    ("ader ord6  (84×84×84)", 84,  84,  84),
]

# Use alpha=1.0 so TensorForge generates working code.
ALPHA = 1.0
BETA  = 1.0  # Accumulate (C += A@B)

# ─── argument parsing ────────────────────────────────────────────────────────

def detect_arch():
    """Return the nvcc arch string for the current GPU, e.g. 'sm_75'."""
    if not torch.cuda.is_available():
        return "sm_75"
    p = torch.cuda.get_device_properties(0)
    return f"sm_{p.major}{p.minor}"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arch",
                   default=None,
                   help=("CUDA arch string passed to nvcc and TensorForge, e.g. sm_75, "
                         "sm_90a. Defaults to auto-detecting from the current GPU."))
    p.add_argument("--batch",   type=int, default=100_000,
                   help="Number of GEMM batch elements (default: 100000). "
                        "Use 10000 for quick smoke-tests, 100000+ for stable timings.")
    p.add_argument("--warmup",  type=int, default=20,
                   help="Number of warm-up iterations before timing (default: 20). "
                        "Higher values reduce JIT / cache cold-start noise.")
    p.add_argument("--repeats", type=int, default=200,
                   help="Number of timed iterations averaged per result (default: 200). "
                        "Higher values give lower timing variance.")
    p.add_argument("--skip-tf",     action="store_true",
                   help="Skip TensorForge benchmarks (useful when nvcc is unavailable).")
    p.add_argument("--skip-triton", action="store_true",
                   help="Skip Triton benchmarks.")
    args = p.parse_args()
    if args.arch is None:
        args.arch = detect_arch()
    return args

# ─── device helpers ──────────────────────────────────────────────────────────

SYS_PEAKS = {}
BENCH_RESULTS = []

def measure_system_peaks(dtype='fp32'):
    """Measure peak Memory Bandwidth and Compute for the current device."""
    print(f"\n[Roofline] Measuring system peaks ({dtype})...")
    
    # 1. Measure Peak Bandwidth (HBM)
    # Use a large copy: 256MB per tensor
    n_elem = 64 * 1024 * 1024
    t_dtype = torch.float32 if dtype == 'fp32' else torch.float64
    elem_bytes = 4 if dtype == 'fp32' else 8
    
    src = torch.randn(n_elem, dtype=t_dtype, device='cuda')
    dst = torch.empty_like(src)
    
    # Warmup
    for _ in range(5):
        dst.copy_(src)
    torch.cuda.synchronize()
    
    t0 = torch.cuda.Event(enable_timing=True)
    t1 = torch.cuda.Event(enable_timing=True)
    
    repeats = 50
    t0.record()
    for _ in range(repeats):
        dst.copy_(src)
    t1.record()
    t1.synchronize()
    
    elapsed_ms = t0.elapsed_time(t1)
    total_bytes = repeats * 2 * n_elem * elem_bytes # Read + Write
    measured_bw = (total_bytes / 1e9) / (elapsed_ms / 1000.0) # GB/s
    print(f"  Measured HBM BW:   {measured_bw:6.1f} GB/s")

    # 2. Measure Peak Compute (TFLOPs)
    # Use a large GEMM (8192x8192)
    N = 8192
    A = torch.randn(N, N, dtype=t_dtype, device='cuda')
    B = torch.randn(N, N, dtype=t_dtype, device='cuda')
    
    # Explicitly enable TF32 for peak measurement if fp32 and capable
    old_allow_tf32 = torch.backends.cuda.matmul.allow_tf32
    if dtype == 'fp32' and torch.cuda.get_device_capability()[0] >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        desc = " (TF32 enabled)"
    else:
        desc = ""
    
    try:
        # Warmup
        torch.mm(A, B)
        torch.cuda.synchronize()
        
        t0.record()
        gemm_repeats = 10
        for _ in range(gemm_repeats):
            torch.mm(A, B)
        t1.record()
        t1.synchronize()
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_allow_tf32
    
    elapsed_ms = t0.elapsed_time(t1)
    total_flops = gemm_repeats * 2 * N**3
    measured_flops = (total_flops / 1e9) / (elapsed_ms / 1000.0) # GFLOPs
    print(f"  Measured Compute:  {measured_flops:6.1f} GFLOPs/s{desc}")
    
    return measured_bw, measured_flops

def gflops(M, N, K, batch, elapsed_s):
    """Standard GEMM FLOPs: 2*M*N*K fused multiply-adds per batch element."""
    return batch * 2 * M * N * K / elapsed_s / 1e9

def get_arithmetic_intensity(M, N, K, dtype, kernel_type='fused'):
    """Calculate Arithmetic Intensity (FLOPs / Byte) for the specific kernel type.
    
    Fused:
        Loads: 3 * (A + B)  [Read-only, cached ideally, but we count compulsory loads]
        Stores: C
        Bytes = (3*(M*K + K*N) + M*N) * sizeof(dtype)
        FLOPs = 3 * 2*M*N*K
    
    Sequential (Baseline):
        3 separate kernels.
        K1: Load A1, B1. Store C. (Beta=0) -> 1x(A+B) + 1x(C_write)
        K2: Load A2, B2, C. Store C. (Beta=1) -> 1x(A+B) + 1x(C_read) + 1x(C_write)
        K3: Load A3, B3, C. Store C. (Beta=1) -> 1x(A+B) + 1x(C_read) + 1x(C_write)
        Total Bytes = [ 3*(M*K + K*N) + 5*(M*N) ] * sizeof(dtype)
        FLOPs = 3 * 2*M*N*K
    """
    bytes_per_elem = 8 if dtype == 'fp64' else 4
    
    # Data Volume (Elements)
    vol_A = M*K
    vol_B = K*N
    vol_C = M*N
    
    if kernel_type == 'fused':
        # 3x A, 3x B, 1x C (write only)
        total_elems = 3*(vol_A + vol_B) + vol_C
    else:
        # Sequential:
        # 1: A+B+C(w)
        # 2: A+B+C(r)+C(w)
        # 3: A+B+C(r)+C(w)
        total_elems = 3*(vol_A + vol_B) + 5*vol_C
        
    total_bytes = total_elems * bytes_per_elem
    total_flops = 3 * 2 * M * N * K
    
    return total_flops / total_bytes

def device_peak_gflops(dtype):
    """Approximate peak GFLOPs/s for this device and dtype.

    sm_90a H100 GH200 : fp64 67 TFLOP/s, fp32 67 TFLOP/s
    sm_80  A100 SXM4  : fp64 19.5,        fp32 19.5 (TF32 ~156)
    sm_75  Turing     : fp64 0.25,        fp32 11
    """
    p = torch.cuda.get_device_properties(0)
    sm = p.major * 10 + p.minor
    if dtype == 'fp64':
        if sm >= 90: return 67_000.0
        if sm >= 80: return 19_500.0
        return 250.0   # CUDA cores only
    else:              # fp32
        if sm >= 90: return 67_000.0
        if sm >= 80: return 19_500.0
        if sm >= 75: return 11_000.0  # Turing WMMA
        return 7_000.0


def supports_fp64_tcdot():
    """sm_80+ supports tl.dot with float64 (tensor cores)."""
    return torch.cuda.get_device_properties(0).major >= 8


def max_block_dim():
    """Cap per-dimension Triton block.

    sm_75 (64 KB): cap at 32 → fits 2 stages of 32×32 fp64 tiles
    sm_80+ (228 KB): cap at 128 — 256 would require >228KB even at 1 stage for
                     some shapes (e.g. 128×128 fp64 = 128KB; safe at 128).
    """
    return 32 if torch.cuda.get_device_properties(0).major < 8 else 128


def _shmem_limit_bytes():
    """Return the per-block shared-memory limit (bytes) Triton can use."""
    p = torch.cuda.get_device_properties(0)
    # Use shared_memory_per_block_optin when available (Triton opts in to max)
    return getattr(p, 'shared_memory_per_block_optin',
                   getattr(p, 'shared_memory_per_block', 49152))


def _safe_num_stages(BM, BN, BK, dtype):
    """Max software-pipeline stages that fit in GPU shared memory."""
    elem_bytes = 8 if dtype == 'fp64' else 4
    shmem_per_stage = (BM * BK + BK * BN) * elem_bytes
    limit = _shmem_limit_bytes()
    return max(1, min(4, limit // shmem_per_stage))


def cuda_time(fn, warmup, repeats):
    """Time a GPU function using CUDA events; returns seconds per call."""
    for _ in range(warmup):
        fn()
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1e-3 / repeats

# ─── TensorForge backend ─────────────────────────────────────────────────────
#
# TensorForge generates a kernel + launcher with this signature:
#   void launcher_NAME(real* m0,  unsigned m0_extraOffset,   ← C (SINK/output)
#                      real* m1,  unsigned m1_extraOffset,   ← A
#                      real* m2,  unsigned m2_extraOffset,   ← B
#                      size_t numElements,
#                      unsigned* flags = nullptr,
#                      void* streamPtr  = nullptr);
# beta is always 0: C is overwritten (C = alpha * A @ B).

_TF_BOILERPLATE = """\
#include <cuda_runtime.h>
#include <cstdio>
#define CHECK_ERR do { \\
  cudaError_t _e = cudaGetLastError(); \\
  if (_e != cudaSuccess) { \\
    fprintf(stderr, "CUDA error %s:%d: %s\\n", \\
            __FILE__, __LINE__, cudaGetErrorString(_e)); \\
  } \\
} while(0)
"""


def build_tf_kernel(M, N, K, arch, dtype):
    """Compile a TensorForge GEMM kernel and return a ctypes callable.

    Parameters
    ----------
    dtype : 'fp32' or 'fp64'

    Returns
    -------
    (lib, func) or (None, None) on failure.
    """
    if not HAS_TF:
        return None, None

    fp_type = FloatingPointType.F32 if dtype == 'fp32' else FloatingPointType.F64

    try:
        ctx = TFContext(arch=arch, backend='cuda', fp_type=fp_type)
    except Exception as e:
        print(f"  [TF] Context failed: {e}")
        return None, None

    mat_a = SubTensor(TFTensor(shape=[M, K], addressing=Addressing.STRIDED))
    mat_b = SubTensor(TFTensor(shape=[K, N], addressing=Addressing.STRIDED))
    mat_c = SubTensor(TFTensor(shape=[M, N], addressing=Addressing.STRIDED))

    try:
        descr = GemmDescr(trans_a=False, trans_b=False,
                          a=mat_a, b=mat_b, c=mat_c,
                          alpha=ALPHA, beta=BETA)
        gen = TFGenerator([descr], ctx)
        gen.set_kernel_name(f"bench_{M}x{N}x{K}_{dtype}")
        gen.generate()
    except Exception as e:
        print(f"  [TF] generation failed: {e}")
        return None, None

    launcher_name = f"launcher_{gen._base_kernel_name}"
    # Wrap launcher in extern "C" so ctypes can find it by the plain name
    # (without C++ mangling). The __global__ kernel itself is called only
    # from within the launcher so it does not need extern "C".
    src = (_TF_BOILERPLATE
           + gen.get_kernel()
           + '\nextern "C" {\n'
           + gen.get_launcher()
           + "\n}\n")

    tmpdir = tempfile.mkdtemp(prefix="tf_bench_")
    cu_path = os.path.join(tmpdir, "kernel.cu")
    so_path = os.path.join(tmpdir, "kernel.so")
    with open(cu_path, "w") as f:
        f.write(src)

    nvcc_cmd = ["nvcc", "-O3", f"-arch={arch}",
                "-Xcompiler", "-fPIC",
                "--shared", "-o", so_path, cu_path, "-lcudart"]
    r = subprocess.run(nvcc_cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  [TF] nvcc failed:\n{r.stderr[:800]}")
        return None, None

    lib  = ctypes.CDLL(so_path)
    func = getattr(lib, launcher_name)
    func.restype = None
    vp = ctypes.c_void_p
    func.argtypes = [
        vp, ctypes.c_uint,   # m0 (C output), m0_extraOffset
        vp, ctypes.c_uint,   # m1 (A),        m1_extraOffset
        vp, ctypes.c_uint,   # m2 (B),        m2_extraOffset
        ctypes.c_size_t,     # numElements
        vp,                  # flags   (NULL)
        vp,                  # streamPtr
    ]
    return lib, func   # keep lib alive to prevent .so unload


def run_tf(func, C_gpu, A_gpu, B_gpu, batch):
    """Invoke TensorForge launcher: C first (SINK), then A, B."""
    stream = torch.cuda.current_stream().cuda_stream
    func(C_gpu.data_ptr(), ctypes.c_uint(0),
         A_gpu.data_ptr(), ctypes.c_uint(0),
         B_gpu.data_ptr(), ctypes.c_uint(0),
         ctypes.c_size_t(batch),
         None, ctypes.c_void_p(stream))


def run_triton_fused(As_gpu, Bs_gpu, C_gpu, M, N, K, batch, dtype):
    """Launch fused Triton GEMM in-place on C_gpu."""
    # Note: As_gpu and Bs_gpu are lists of 3 tensors each
    # For simplicity, we use the first one to determine strides (assuming uniform layout)
    
    elem_bytes = 8 if dtype == 'fp64' else 4
    kwargs = dict(
        M=M, N=N, K=K,
        stride_ab=As_gpu[0].stride(0), stride_am=As_gpu[0].stride(1), stride_ak=As_gpu[0].stride(2),
        stride_bb=Bs_gpu[0].stride(0), stride_bk=Bs_gpu[0].stride(1), stride_bn=Bs_gpu[0].stride(2),
        stride_cb=C_gpu.stride(0), stride_cm=C_gpu.stride(1), stride_cn=C_gpu.stride(2),
        alpha=ALPHA, beta=BETA,
        # Pointer arguments
        A1_ptr=As_gpu[0], B1_ptr=Bs_gpu[0],
        A2_ptr=As_gpu[1], B2_ptr=Bs_gpu[1],
        A3_ptr=As_gpu[2], B3_ptr=Bs_gpu[2],
    )

    if dtype == 'fp64':
        kernel = _fused_gemm_fp64_3x
    else:
        kernel = _fused_gemm_fp32_3x

    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']), batch)
    kernel[grid](C_gpu, **kwargs)
    
    best_config = kernel.best_config
    BM, BN, BK = best_config.kwargs['BLOCK_M'], best_config.kwargs['BLOCK_N'], best_config.kwargs['BLOCK_K']
    ns, warps = best_config.num_stages, best_config.num_warps
    return f"{dtype} FUSED [BLOCK={BM}×{BN}×{BK}  stages={ns} warps={warps}]"

if HAS_TRITON:
    def get_autotune_configs():
        return [
            triton.Config({'BLOCK_M': 16, 'BLOCK_N': 16, 'BLOCK_K': 16}, num_stages=2, num_warps=2),
            triton.Config({'BLOCK_M': 32, 'BLOCK_N': 16, 'BLOCK_K': 16}, num_stages=2, num_warps=2),
            triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_stages=2, num_warps=2),
            triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
            triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
            triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
            triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
            triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
        ]

    # --- Fused Kernels ---
    @triton.autotune(configs=get_autotune_configs(), key=['M', 'N', 'K'])
    @triton.jit
    def _fused_gemm_fp64_3x(
        C_ptr,
        A1_ptr, B1_ptr, A2_ptr, B2_ptr, A3_ptr, B3_ptr,
        M, N, K,
        stride_ab, stride_am, stride_ak,
        stride_bb, stride_bk, stride_bn,
        stride_cb, stride_cm, stride_cn,
        alpha, beta,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    ):
        # Fused GEMM for 3 matrix products: C = alpha * (A1@B1 + A2@B2 + A3@B3)
        # Note: beta is ignored/assumed 0 for the output store (we overwrite C)
        # Or rather, let's just implement C = sum(AiBi).
        
        pid_mn   = tl.program_id(0)
        batch_id = tl.program_id(1)
        num_n    = tl.cdiv(N, BLOCK_N)
        pid_m    = pid_mn // num_n
        pid_n    = pid_mn  % num_n

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        mask_m = offs_m < M
        mask_n = offs_n < N

        # Pointers for all 3 ops
        A1 = A1_ptr + batch_id * stride_ab
        B1 = B1_ptr + batch_id * stride_bb
        A2 = A2_ptr + batch_id * stride_ab
        B2 = B2_ptr + batch_id * stride_bb
        A3 = A3_ptr + batch_id * stride_ab
        B3 = B3_ptr + batch_id * stride_bb
        
        C = C_ptr + batch_id * stride_cb

        # Initialize accumulator
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float64)

        # Loop over K blocks - shared loop for all 3 GEMMs
        # This assumes all 3 have same K dimension.
        
        a1_ptrs = A1 + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        b1_ptrs = B1 + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
        a2_ptrs = A2 + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        b2_ptrs = B2 + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
        a3_ptrs = A3 + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        b3_ptrs = B3 + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

        for k in range(0, K, BLOCK_K):
            mask_k = offs_k < K - k  # Adjusted mask logic (offs_k is 0..BLOCK_K)
            
            # Op 1
            a1 = tl.load(a1_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
            b1 = tl.load(b1_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
            acc += tl.dot(a1, b1, allow_tf32=False)

            # Op 2
            a2 = tl.load(a2_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
            b2 = tl.load(b2_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
            acc += tl.dot(a2, b2, allow_tf32=False)

            # Op 3
            a3 = tl.load(a3_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
            b3 = tl.load(b3_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
            acc += tl.dot(a3, b3, allow_tf32=False)

            # Advance pointers
            a1_ptrs += BLOCK_K * stride_ak
            b1_ptrs += BLOCK_K * stride_bk
            a2_ptrs += BLOCK_K * stride_ak
            b2_ptrs += BLOCK_K * stride_bk
            a3_ptrs += BLOCK_K * stride_ak
            b3_ptrs += BLOCK_K * stride_bk

        # Store result
        c_ptrs = C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        c_mask = mask_m[:, None] & mask_n[None, :]
        # We write directly (C = sum), ignoring initial C value
        tl.store(c_ptrs, alpha * acc, mask=c_mask)

    @triton.autotune(configs=get_autotune_configs(), key=['M', 'N', 'K'])
    @triton.jit
    def _fused_gemm_fp32_3x(
        C_ptr,
        A1_ptr, B1_ptr, A2_ptr, B2_ptr, A3_ptr, B3_ptr,
        M, N, K,
        stride_ab, stride_am, stride_ak,
        stride_bb, stride_bk, stride_bn,
        stride_cb, stride_cm, stride_cn,
        alpha, beta,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    ):
        # Same as fp64 but with float32/TF32
        pid_mn   = tl.program_id(0)
        batch_id = tl.program_id(1)
        num_n    = tl.cdiv(N, BLOCK_N)
        pid_m    = pid_mn // num_n
        pid_n    = pid_mn  % num_n

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        mask_m = offs_m < M
        mask_n = offs_n < N

        A1 = A1_ptr + batch_id * stride_ab
        B1 = B1_ptr + batch_id * stride_bb
        A2 = A2_ptr + batch_id * stride_ab
        B2 = B2_ptr + batch_id * stride_bb
        A3 = A3_ptr + batch_id * stride_ab
        B3 = B3_ptr + batch_id * stride_bb
        C = C_ptr + batch_id * stride_cb

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        a1_ptrs = A1 + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        b1_ptrs = B1 + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
        a2_ptrs = A2 + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        b2_ptrs = B2 + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
        a3_ptrs = A3 + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        b3_ptrs = B3 + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

        for k in range(0, K, BLOCK_K):
            mask_k = offs_k < K - k
            
            a1 = tl.load(a1_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
            b1 = tl.load(b1_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
            acc += tl.dot(a1, b1)

            a2 = tl.load(a2_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
            b2 = tl.load(b2_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
            acc += tl.dot(a2, b2)

            a3 = tl.load(a3_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
            b3 = tl.load(b3_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
            acc += tl.dot(a3, b3)

            a1_ptrs += BLOCK_K * stride_ak
            b1_ptrs += BLOCK_K * stride_bk
            a2_ptrs += BLOCK_K * stride_ak
            b2_ptrs += BLOCK_K * stride_bk
            a3_ptrs += BLOCK_K * stride_ak
            b3_ptrs += BLOCK_K * stride_bk

        c_ptrs = C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        c_mask = mask_m[:, None] & mask_n[None, :]
        tl.store(c_ptrs, alpha * acc, mask=c_mask)


# ─── per-size, per-dtype benchmark ───────────────────────────────────────────

def bench_dtype(label, M, N, K, batch, warmup, repeats, arch, dtype,
                skip_tf, skip_triton):
    """Run TensorForge and Triton (Sequential vs Fused) for one size/dtype."""
    np_dtype = np.float64 if dtype == 'fp64' else np.float32
    th_dtype = torch.float64 if dtype == 'fp64' else torch.float32
    
    if dtype == 'fp64':
        rtol = 1e-5
    elif supports_fp64_tcdot():
        rtol = 5e-2
    else:
        rtol = 1e-4
    
    # 3x operations per "fused" step
    if dtype in SYS_PEAKS:
        peak = SYS_PEAKS[dtype]['compute']
    else:
        peak = device_peak_gflops(dtype)

    rng  = np.random.default_rng(42)
    
    # Create 3 sets of inputs
    As_np = [rng.standard_normal((batch, M, K)).astype(np_dtype) for _ in range(3)]
    Bs_np = [rng.standard_normal((batch, K, N)).astype(np_dtype) for _ in range(3)]

    # Reference: C = Sum(alpha * A_i @ B_i)
    # We ignore beta here as we assume C is initialized to 0 or accumulated from scratch
    C_ref = np.zeros((batch, M, N), dtype=np_dtype)
    for i in range(3):
        C_ref += (np_dtype(ALPHA) * np.einsum("bik,bkj->bij", As_np[i], Bs_np[i]))

    # Prepare GPU tensors (Column-Major)
    As_gpu = []
    Bs_gpu = []
    for i in range(3):
        a = torch.from_numpy(As_np[i]).to(th_dtype).cuda()
        a = a.transpose(1, 2).contiguous().transpose(1, 2)
        As_gpu.append(a)
        
        b = torch.from_numpy(Bs_np[i]).to(th_dtype).cuda()
        b = b.transpose(1, 2).contiguous().transpose(1, 2)
        Bs_gpu.append(b)

    print(f"\n  ── {dtype} ─────────────────────────────────────")
    
    # Roofline info
    bw = SYS_PEAKS.get('bw', None)
    peak = SYS_PEAKS.get(dtype, {}).get('compute', None)
    
    if bw and peak:
        ai_seq = get_arithmetic_intensity(M, N, K, dtype, 'sequential')
        ai_fus = get_arithmetic_intensity(M, N, K, dtype, 'fused')
        
        rl_seq = min(peak, ai_seq * bw)
        rl_fus = min(peak, ai_fus * bw)
        
        print(f"  [Roofline] System: {peak:.0f} GFLOPs/s, {bw:.0f} GB/s")
        print(f"  [Roofline] Seq:   AI={ai_seq:.2f} FLOP/B  -> Roofline: {rl_seq:.0f} GFLOPs/s")
        print(f"  [Roofline] Fused: AI={ai_fus:.2f} FLOP/B  -> Roofline: {rl_fus:.0f} GFLOPs/s")
    
    # ── TensorForge (Sequential) ─────────────────────────────────────────────
    if not skip_tf:
        if not HAS_TF:
            print(f"  TensorForge [{dtype}]: not installed.")
        else:
            print(f"  Building TensorForge kernel ({dtype})...")
            # We build ONE kernel (accumulating: beta=1.0) and call it 3 times
            lib, tf_func = build_tf_kernel(M, N, K, arch, dtype)
            
            if tf_func is None:
                print(f"  TensorForge [{dtype}]: build failed.")
            else:
                # Prepare TF inputs (Column-major physical layout: batch, K, M)
                As_tf = []
                Bs_tf = []
                for i in range(3):
                    As_tf.append(torch.from_numpy(np.ascontiguousarray(As_np[i].transpose(0, 2, 1))).to(th_dtype).cuda())
                    Bs_tf.append(torch.from_numpy(np.ascontiguousarray(Bs_np[i].transpose(0, 2, 1))).to(th_dtype).cuda())
                
                C_tf = torch.zeros(batch, N, M, dtype=th_dtype, device='cuda')
                C_tmp = torch.zeros_like(C_tf)

                def call_tf_seq():
                    # Manual accumulation: TensorForge (gemmforge) in this env ignores beta=1.0,
                    # generating overwrite code (C = alpha*A*B).
                    # We manually accumulate (C += A*B) using a temp buffer to fix correctness.
                    # This adds overhead (extra kernel + memory traffic), but is necessary for validation.
                    
                    # 1. C_tf = A0 * B0 (overwrite)
                    run_tf(tf_func, C_tf, As_tf[0], Bs_tf[0], batch)

                    # 2. C_tmp = A1 * B1; C_tf += C_tmp
                    run_tf(tf_func, C_tmp, As_tf[1], Bs_tf[1], batch)
                    C_tf.add_(C_tmp)

                    # 3. C_tmp = A2 * B2; C_tf += C_tmp
                    run_tf(tf_func, C_tmp, As_tf[2], Bs_tf[2], batch)
                    C_tf.add_(C_tmp)

                # Correctness check
                call_tf_seq()
                torch.cuda.synchronize()
                C_tf_rm = C_tf.permute(0, 2, 1).contiguous().cpu().numpy()
                rel_err = np.max(np.abs(C_tf_rm - C_ref)) / (np.max(np.abs(C_ref)) + 1e-30)
                ok = "✓" if rel_err < rtol else f"✗ err={rel_err:.2e}"

                # Timing
                t = cuda_time(call_tf_seq, warmup, repeats)
                # Total FLOPs = 3 * (2*M*N*K * batch)
                gf = (3 * batch * 2 * M * N * K) / t / 1e9
                print(f"  TensorForge [{dtype}]: {gf:8.1f} GFLOPs/s  ({100*gf/peak:.1f}%)  correctness {ok}")
                ai_seq = get_arithmetic_intensity(M, N, K, dtype, 'sequential')
                BENCH_RESULTS.append({'label': label, 'dtype': dtype, 'type': 'tf', 'ai': ai_seq, 'gflops': gf})

    # ── Triton (Sequential) ──────────────────────────────────────────────────
    if not skip_triton:
        if not HAS_TRITON:
            print(f"  Triton      [{dtype}]: not installed.")
        elif dtype == 'fp64' and not supports_fp64_tcdot():
            print(f"  Triton      [{dtype}]: skipped (no fp64 TC)")
        else:
            C_seq = torch.zeros(batch, N, M, dtype=th_dtype, device='cuda').transpose(1, 2)
            
            def call_triton_seq():
                # We don't rely on zero_() which adds kernel overhead.
                # Instead, we use beta=0 for the first call (overwrite) and beta=1 for subsequent calls.
                run_triton_sequential(As_gpu[0], Bs_gpu[0], C_seq, M, N, K, batch, dtype, beta=0.0)
                for i in range(1, 3):
                    run_triton_sequential(As_gpu[i], Bs_gpu[i], C_seq, M, N, K, batch, dtype, beta=1.0)
            
            # Correctness
            call_triton_seq()
            torch.cuda.synchronize()
            rel_err = np.max(np.abs(C_seq.cpu().numpy() - C_ref)) / (np.max(np.abs(C_ref)) + 1e-30)
            ok = "✓" if rel_err < rtol else f"✗ err={rel_err:.2e}"
            
            # Timing
            t = cuda_time(call_triton_seq, warmup, repeats)
            gf = (3 * batch * 2 * M * N * K) / t / 1e9
            print(f"  Triton SEQ  [{dtype}]: {gf:8.1f} GFLOPs/s  ({100*gf/peak:.1f}%)  correctness {ok}")
            ai_seq = get_arithmetic_intensity(M, N, K, dtype, 'sequential')
            BENCH_RESULTS.append({'label': label, 'dtype': dtype, 'type': 'sequential', 'ai': ai_seq, 'gflops': gf})


    # ── Triton (Fused) ───────────────────────────────────────────────────────
    if not skip_triton:
        if not HAS_TRITON:
            pass
        elif dtype == 'fp64' and not supports_fp64_tcdot():
            pass
        else:
            C_fused = torch.zeros(batch, N, M, dtype=th_dtype, device='cuda').transpose(1, 2)
            tag = ""

            def call_triton_fused():
                nonlocal tag
                # Kernel overwrites C, so initialization doesn't strictly matter if alpha/beta logic is right
                # But our fused kernel does C = sum(AiBi), effectively overwriting.
                tag = run_triton_fused(As_gpu, Bs_gpu, C_fused, M, N, K, batch, dtype)

            # JIT + Correctness
            try:
                call_triton_fused()
                torch.cuda.synchronize()
            except Exception as e:
                print(f"  Triton FUSED [{dtype}]: JIT failed: {e}")
                return

            rel_err = np.max(np.abs(C_fused.cpu().numpy() - C_ref)) / (np.max(np.abs(C_ref)) + 1e-30)
            ok = "✓" if rel_err < rtol else f"✗ err={rel_err:.2e}"
            
            # Timing
            t = cuda_time(call_triton_fused, warmup, repeats)
            gf = (3 * batch * 2 * M * N * K) / t / 1e9
            print(f"  Triton FUSD [{dtype}]: {gf:8.1f} GFLOPs/s  ({100*gf/peak:.1f}%)  correctness {ok}  [{tag}]")
            ai_fus = get_arithmetic_intensity(M, N, K, dtype, 'fused')
            BENCH_RESULTS.append({'label': label, 'dtype': dtype, 'type': 'fused', 'ai': ai_fus, 'gflops': gf})


def bench_size(label, M, N, K, batch, warmup, repeats, arch, skip_tf, skip_triton):
    print(f"\n{'─'*65}")
    print(f"  {label}   batch={batch:,}")
    print(f"{'─'*65}")
    for dtype in ('fp32', 'fp64'):
        bench_dtype(label, M, N, K, batch, warmup, repeats, arch,
                    dtype, skip_tf, skip_triton)


# ─── main ────────────────────────────────────────────────────────────────────

def plot_roofline(filename='roofline_plot.png'):
    """Generate Roofline plot from collected results (Split FP32/FP64)."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.lines as mlines
        import matplotlib.cm as cm
    except ImportError:
        print("\nWARNING: matplotlib not found. Skipping plot generation.")
        return

    if not SYS_PEAKS:
        print("WARNING: No system peaks measured. Skipping plot.")
        return

    print(f"\nGenerating Roofline plot -> {filename} ...")
    
    # Create subplots: 1 row, 2 columns (FP32 left, FP64 right)
    fig, axes = plt.subplots(1, 2, figsize=(20, 9))
    
    # Define colors/markers
    roofline_colors = {'fp32': 'tab:blue', 'fp64': 'tab:orange'}
    markers = {'sequential': 'o', 'fused': '^', 'tf': 's'}
    
    # Extract unique orders and assign colors
    def get_order(lbl):
        parts = lbl.split()
        for p in parts:
            if p.startswith('ord'):
                return p
        return "ord?"

    unique_orders = sorted(list(set(get_order(r['label']) for r in BENCH_RESULTS)))
    unique_orders.sort(key=lambda x: int(x.replace('ord', '')) if x.replace('ord', '').isdigit() else 999)
    
    # Color map for orders
    cmap = plt.get_cmap('viridis', len(unique_orders))
    order_colors = {ord_name: cmap(i) for i, ord_name in enumerate(unique_orders)}

    # Plot FP32 and FP64 in separate subplots
    dtypes = ['fp32', 'fp64']
    min_ai, max_ai = 0.05, 200.0

    for i, dtype in enumerate(dtypes):
        ax = axes[i]
        
        if dtype not in SYS_PEAKS:
            ax.text(0.5, 0.5, f"No {dtype} Peak Measured", ha='center')
            continue
        
        peak_bw = SYS_PEAKS['bw']
        peak_flops = SYS_PEAKS[dtype]['compute']
        color = roofline_colors.get(dtype, 'gray')
        
        # Roofline: y = min(peak_flops, x * peak_bw)
        ridge_ai = peak_flops / peak_bw
        
        # Draw Roofline
        x = [min_ai, ridge_ai, max_ai]
        y = [min_ai * peak_bw, peak_flops, peak_flops]
        
        ax.plot(x, y, color=color, linestyle='--', linewidth=2, label=f'{dtype} Roofline')
        ax.text(ridge_ai, peak_flops * 1.15, f'{dtype} Peak: {peak_flops:.0f} GF', 
                 color=color, fontsize=11, ha='center', weight='bold')
        ax.text(min_ai * 1.5, min_ai * 1.5 * peak_bw * 1.2, f'{peak_bw:.0f} GB/s',
                 color=color, fontsize=11, rotation=30, weight='bold')

        # Filter points for this dtype
        subset = [r for r in BENCH_RESULTS if r['dtype'] == dtype]
        
        for res in subset:
            ai = res['ai']
            gflops = res['gflops']
            ktype = res['type']
            label = res['label']
            
            ord_name = get_order(label)
            c = order_colors.get(ord_name, 'black')
            m = markers.get(ktype, 'x')
            
            ax.scatter(ai, gflops, color=c, marker=m, s=150, edgecolors='black', linewidth=0.5, alpha=0.9, zorder=3)

        # Config Plot
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(min_ai, max_ai)
        ax.set_ylim(bottom=10.0) # Avoid log(0) or very low values
        
        ax.set_title(f"{dtype.upper()} Performance", fontsize=16, weight='bold')
        ax.set_xlabel('Arithmetic Intensity (FLOPs/Byte)', fontsize=13)
        if i == 0:
            ax.set_ylabel('Performance (GFLOPs/s)', fontsize=13)
            
        ax.grid(True, which="major", ls="-", alpha=0.4)
        ax.grid(True, which="minor", ls=":", alpha=0.2)

    # Build Shared Legend
    legend_elements = []
    
    # 1. Implementation
    legend_elements.append(mlines.Line2D([], [], color='none', label='Implementation:'))
    legend_elements.append(mlines.Line2D([], [], marker='^', color='w', label='Triton Fused', markerfacecolor='gray', markersize=10, markeredgecolor='k'))
    legend_elements.append(mlines.Line2D([], [], marker='o', color='w', label='Triton Seq', markerfacecolor='gray', markersize=10, markeredgecolor='k'))
    legend_elements.append(mlines.Line2D([], [], marker='s', color='w', label='TensorForge', markerfacecolor='gray', markersize=10, markeredgecolor='k'))
    
    # 2. Orders
    legend_elements.append(mlines.Line2D([], [], color='none', label=''))
    legend_elements.append(mlines.Line2D([], [], color='none', label='Order (Size):'))
    for ord_name in unique_orders:
        c = order_colors[ord_name]
        legend_elements.append(mlines.Line2D([], [], marker='o', color='w', label=ord_name, markerfacecolor=c, markersize=10, markeredgecolor='k'))

    # Place legend to the right
    fig.legend(handles=legend_elements, loc='center right', bbox_to_anchor=(0.98, 0.5), fontsize='medium', framealpha=0.9)
    plt.subplots_adjust(right=0.88, wspace=0.15)
    
    plt.suptitle('Roofline Analysis: Triton Fused vs Sequential vs TensorForge', fontsize=18)
    plt.savefig(filename, dpi=300)
    print("Done.")

def main():
    args = parse_args()

    if not torch.cuda.is_available():
        sys.exit("ERROR: no CUDA device found.")

    dev = torch.cuda.get_device_properties(0)
    sm  = dev.major * 10 + dev.minor
    print(f"\nDevice   : {dev.name}")
    print(f"SM       : {dev.major}.{dev.minor}  (requested arch={args.arch})")
    print(f"Memory   : {dev.total_memory / 2**30:.1f} GiB")
    print(f"Batch    : {args.batch:,}   warmup={args.warmup}   repeats={args.repeats}")
    print(f"fp64 TC  : {'yes (sm_80+)' if supports_fp64_tcdot() else 'no (sm<80, Triton fp64 skipped)'}")
    
    # Measure Measured Peaks
    try:
        bw, cp32 = measure_system_peaks('fp32')
        _,  cp64 = measure_system_peaks('fp64')
        SYS_PEAKS['bw'] = bw
        SYS_PEAKS['fp32'] = {'compute': cp32}
        SYS_PEAKS['fp64'] = {'compute': cp64}
    except Exception as e:
        print(f"WARNING: Measure peaks failed: {e}")

    if HAS_TRITON:
        print(f"Triton   : {triton.__version__}")
    if HAS_TF:
        from tensorforge import get_version
        print(f"TensorFrg: {get_version()}")

    for label, M, N, K in DG_SIZES:
        bench_size(label, M, N, K,
                   batch=args.batch,
                   warmup=args.warmup,
                   repeats=args.repeats,
                   arch=args.arch,
                   skip_tf=args.skip_tf,
                   skip_triton=args.skip_triton)

    print(f"\n{'═'*65}")
    print("Done.")
    
    plot_roofline()



# --- Sequential Kernels (Baseline) ---

@triton.autotune(configs=get_autotune_configs(), key=['M', 'N', 'K'])
@triton.jit
def _batched_gemm_fp64(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_ab, stride_am, stride_ak,
    stride_bb, stride_bk, stride_bn,
    stride_cb, stride_cm, stride_cn,
    alpha, beta,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_mn   = tl.program_id(0)
    batch_id = tl.program_id(1)
    num_n    = tl.cdiv(N, BLOCK_N)
    pid_m    = pid_mn // num_n
    pid_n    = pid_mn  % num_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    mask_m = offs_m < M
    mask_n = offs_n < N

    A = A_ptr + batch_id * stride_ab
    B = B_ptr + batch_id * stride_bb
    C = C_ptr + batch_id * stride_cb

    acc    = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float64)
    a_ptrs = A + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    for _ in range(0, K, BLOCK_K):
        mask_k = offs_k < K
        a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
        acc += tl.dot(a, b, allow_tf32=False)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
        offs_k += BLOCK_K

    c_ptrs = C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = mask_m[:, None] & mask_n[None, :]
    c_old  = tl.load(c_ptrs, mask=c_mask, other=0.0)
    tl.store(c_ptrs, alpha * acc + beta * c_old, mask=c_mask)

@triton.autotune(configs=get_autotune_configs(), key=['M', 'N', 'K'])
@triton.jit
def _batched_gemm_fp32(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_ab, stride_am, stride_ak,
    stride_bb, stride_bk, stride_bn,
    stride_cb, stride_cm, stride_cn,
    alpha, beta,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_mn   = tl.program_id(0)
    batch_id = tl.program_id(1)
    num_n    = tl.cdiv(N, BLOCK_N)
    pid_m    = pid_mn // num_n
    pid_n    = pid_mn  % num_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    mask_m = offs_m < M
    mask_n = offs_n < N

    A = A_ptr + batch_id * stride_ab
    B = B_ptr + batch_id * stride_bb
    C = C_ptr + batch_id * stride_cb

    acc    = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    a_ptrs = A + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    for _ in range(0, K, BLOCK_K):
        mask_k = offs_k < K
        a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
        offs_k += BLOCK_K

    c_ptrs = C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = mask_m[:, None] & mask_n[None, :]
    c_old  = tl.load(c_ptrs, mask=c_mask, other=0.0)
    tl.store(c_ptrs, alpha * acc + beta * c_old, mask=c_mask)

def run_triton_sequential(A_gpu, B_gpu, C_gpu, M, N, K, batch, dtype, beta=BETA):
    """Launch single Triton GEMM in-place on C_gpu."""
    kwargs = dict(
        M=M, N=N, K=K,
        stride_ab=A_gpu.stride(0), stride_am=A_gpu.stride(1), stride_ak=A_gpu.stride(2),
        stride_bb=B_gpu.stride(0), stride_bk=B_gpu.stride(1), stride_bn=B_gpu.stride(2),
        stride_cb=C_gpu.stride(0), stride_cm=C_gpu.stride(1), stride_cn=C_gpu.stride(2),
        alpha=ALPHA, beta=beta,
    )

    if dtype == 'fp64':
        kernel = _batched_gemm_fp64
    else:
        kernel = _batched_gemm_fp32
        
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']), batch)
    kernel[grid](A_gpu, B_gpu, C_gpu, **kwargs)
    
    best_config = kernel.best_config
    BM, BN, BK = best_config.kwargs['BLOCK_M'], best_config.kwargs['BLOCK_N'], best_config.kwargs['BLOCK_K']
    ns, warps = best_config.num_stages, best_config.num_warps
    return f"{dtype} SEQ  [BLOCK={BM}x{BN}x{BK}  stages={ns} warps={warps}]"

if __name__ == "__main__":
    main()

