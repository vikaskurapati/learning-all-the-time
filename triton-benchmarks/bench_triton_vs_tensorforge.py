#!/usr/bin/env python3
"""
Step-1 feasibility probe: Triton JIT vs TensorForge-generated CUDA kernels
for batched small GEMMs – testing on sm_75 locally, targeting sm_90a on Vista.

For each (M, N, K) representative of SeisSol DG polynomial orders 2-7,
and for each precision (fp32, fp64):

  1. CPU reference:  C = alpha * A @ B  (beta=0, so C is overwritten)
  2. TensorForge: generate CUDA source, compile with nvcc, load via ctypes.
  3. Triton JIT: batched GEMM kernel.
     On sm_75, fp64 Triton is skipped (fp64 tensor cores require sm_80+).
  4. Correctness check against CPU reference.
  5. GFLOPs/s and % of device peak.

Usage
-----
  python bench_triton_vs_tensorforge.py [--arch sm_90a] [--batch 100000]
                                        [--warmup 20] [--repeats 200]
                                        [--skip-tf] [--skip-triton]

Requirements
------------
  pip install torch triton
  pip install git+https://github.com/SeisSol/TensorForge
  nvcc on PATH
"""

import argparse
import ctypes
import math
import os
import subprocess
import sys
import tempfile

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
# (The alpha != 1.0 path in TF's GemmDescr is currently broken: it emits a
# comment stub for the fused multiply instead of actual code, leaving the
# accumulator at zero.)
ALPHA = 1.0
BETA  = 0.0

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

def gflops(M, N, K, batch, elapsed_s):
    """Standard GEMM FLOPs: 2*M*N*K fused multiply-adds per batch element."""
    return batch * 2 * M * N * K / elapsed_s / 1e9


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


# ─── Triton kernels ──────────────────────────────────────────────────────────
#
# Two kernels:
#   _batched_gemm_fp64  – sm_80+ only; uses fp64 tensor cores via tl.dot
#   _batched_gemm_fp32  – sm_75+; pure fp32, uses WMMA on Turing
#
# Grid: (ceil(M/BM)*ceil(N/BN), batch)
# allow_tf32 is left at its default (True for fp32) to avoid a broken
# non-tensor-core path on sm_75 that produces zeros when set to False.
# For fp64 (which has no TF32 analog) allow_tf32=False is explicit and correct.

if HAS_TRITON:
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
        """fp64 GEMM – requires sm_80+ for tl.dot with float64."""
        pid_mn   = tl.program_id(0)
        batch_id = tl.program_id(1)
        num_n    = tl.cdiv(N, BLOCK_N)
        pid_m    = pid_mn // num_n
        pid_n    = pid_mn  % num_n

        A = A_ptr + batch_id * stride_ab
        B = B_ptr + batch_id * stride_bb
        C = C_ptr + batch_id * stride_cb

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        mask_m = offs_m < M
        mask_n = offs_n < N

        acc    = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float64)
        a_ptrs = A + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        b_ptrs = B + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

        for _ in range(0, K, BLOCK_K):
            # offs_k tracks absolute K indices; compare against K for masking
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
        """fp32 GEMM – works on sm_75+ via WMMA (Turing) or TF32 (Ampere+)."""
        pid_mn   = tl.program_id(0)
        batch_id = tl.program_id(1)
        num_n    = tl.cdiv(N, BLOCK_N)
        pid_m    = pid_mn // num_n
        pid_n    = pid_mn  % num_n

        A = A_ptr + batch_id * stride_ab
        B = B_ptr + batch_id * stride_bb
        C = C_ptr + batch_id * stride_cb

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        mask_m = offs_m < M
        mask_n = offs_n < N

        acc    = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        a_ptrs = A + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        b_ptrs = B + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

        for _ in range(0, K, BLOCK_K):
            mask_k = offs_k < K
            a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
            b = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
            acc += tl.dot(a, b)   # default allow_tf32; do not set False on sm_75
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk
            offs_k += BLOCK_K

        c_ptrs = C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        c_mask = mask_m[:, None] & mask_n[None, :]
        c_old  = tl.load(c_ptrs, mask=c_mask, other=0.0)
        tl.store(c_ptrs, alpha * acc + beta * c_old, mask=c_mask)


def _next_pow2(x):
    return 1 if x <= 1 else 2 ** math.ceil(math.log2(x))


def run_triton(A_gpu, B_gpu, C_gpu, M, N, K, batch, dtype):
    """Launch Triton GEMM in-place on C_gpu. Returns a descriptive tag string.

    dtype : 'fp32' or 'fp64'
    On sm_75 fp64 is not supported; caller should skip or catch the error.
    """
    elem_bytes = 8 if dtype == 'fp64' else 4
    cap = max_block_dim()
    BM  = min(max(16, _next_pow2(M)), cap)
    BN  = min(max(16, _next_pow2(N)), cap)
    BK  = min(max(16, _next_pow2(K)), cap)

    # Shrink BM/BK until at least 1 pipeline stage fits in shared memory.
    # BN is kept fixed (it covers the narrow N dimension and is usually small).
    limit = _shmem_limit_bytes()
    while BM > 16 and BK > 16 and (BM * BK + BK * BN) * elem_bytes > limit:
        BM = max(16, BM // 2)
        BK = max(16, BK // 2)

    ns   = _safe_num_stages(BM, BN, BK, dtype)
    grid = (math.ceil(M / BM) * math.ceil(N / BN), batch)

    kwargs = dict(
        M=M, N=N, K=K,
        stride_ab=A_gpu.stride(0), stride_am=A_gpu.stride(1), stride_ak=A_gpu.stride(2),
        stride_bb=B_gpu.stride(0), stride_bk=B_gpu.stride(1), stride_bn=B_gpu.stride(2),
        stride_cb=C_gpu.stride(0), stride_cm=C_gpu.stride(1), stride_cn=C_gpu.stride(2),
        alpha=ALPHA, beta=BETA,
        BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK,
    )

    if dtype == 'fp64':
        _batched_gemm_fp64[grid](A_gpu, B_gpu, C_gpu, **kwargs, num_stages=ns)
        return f"fp64 TC  [BLOCK={BM}×{BN}×{BK}  stages={ns}]"
    else:
        _batched_gemm_fp32[grid](A_gpu, B_gpu, C_gpu, **kwargs, num_stages=ns)
        return f"fp32 TC  [BLOCK={BM}×{BN}×{BK}  stages={ns}]"


# ─── per-size, per-dtype benchmark ───────────────────────────────────────────

def bench_dtype(label, M, N, K, batch, warmup, repeats, arch, dtype,
                skip_tf, skip_triton):
    """Run TensorForge and Triton for one (M,N,K) size and one dtype."""
    np_dtype = np.float64 if dtype == 'fp64' else np.float32
    th_dtype = torch.float64 if dtype == 'fp64' else torch.float32
    # Relative-error tolerances.  C values are O(sqrt(K)) for N(0,1) inputs.
    # fp64: allow_tf32=False in kernel → near machine-eps; 1e-5 relative is generous.
    # fp32 sm_75: true fp32 path → ~1e-4 relative.
    # fp32 sm_80+: tl.dot uses TF32 (10-bit mantissa, ~1e-3 relative per op);
    #   accumulated over K steps ~ sqrt(K)*1e-3 ≈ 1e-2 relative → use 5e-2.
    if dtype == 'fp64':
        rtol = 1e-5
    elif supports_fp64_tcdot():   # sm_80+ → TF32
        rtol = 5e-2
    else:
        rtol = 1e-4
    peak     = device_peak_gflops(dtype)

    rng  = np.random.default_rng(42)
    A_np = rng.standard_normal((batch, M, K)).astype(np_dtype)
    B_np = rng.standard_normal((batch, K, N)).astype(np_dtype)

    # Reference: C = alpha * A @ B  (beta=0)
    C_ref = (np_dtype(ALPHA) * np.einsum("bik,bkj->bij", A_np, B_np)).astype(np_dtype)

    # Row-major GPU tensors used by Triton
    A_gpu = torch.from_numpy(A_np).to(th_dtype).cuda().contiguous()
    B_gpu = torch.from_numpy(B_np).to(th_dtype).cuda().contiguous()

    print(f"\n  ── {dtype} ─────────────────────────────────────")

    # ── TensorForge ──────────────────────────────────────────────────────────
    # TensorForge follows the column-major (Fortran/BLAS) convention used by
    # SeisSol.  Our NumPy/PyTorch data is row-major, so we must reinterpret it:
    #
    #   Row-major  A [batch, M, K]  stored as  col-major [M, K]
    #   ↔  Present as  [batch, K, M]  C-contiguous  (same bytes, different shape).
    #   element [b,m,k]: row-major offset b*M*K + m*K + k
    #                  = col-major offset b*M*K + m + k*M  (same when k and m
    #                    are interchanged in storage → transpose inner dims).
    #   → A_tf = A_np.transpose(0,2,1) made C-contiguous.
    #
    #   Same logic for B: [batch,K,N] row-major → [batch,N,K] C-contiguous.
    #
    #   Output C col-major [M,N] stored in [batch,N,M] C-contiguous tensor.
    #   After TF writes: C_tf.permute(0,2,1) gives row-major [batch,M,N] = C_ref.
    if not skip_tf:
        if not HAS_TF:
            print(f"  TensorForge [{dtype}]: not installed.")
        else:
            print(f"  Building TensorForge kernel ({dtype})...")
            lib, tf_func = build_tf_kernel(M, N, K, arch, dtype)
            if tf_func is None:
                print(f"  TensorForge [{dtype}]: build failed.")
            else:
                # Column-major transposed copies for TF
                A_tf = torch.from_numpy(
                    np.ascontiguousarray(A_np.transpose(0, 2, 1))
                ).to(th_dtype).cuda()   # shape [batch, K, M]
                B_tf = torch.from_numpy(
                    np.ascontiguousarray(B_np.transpose(0, 2, 1))
                ).to(th_dtype).cuda()   # shape [batch, N, K]
                C_tf = torch.zeros(batch, N, M, dtype=th_dtype, device='cuda')

                def call_tf():
                    run_tf(tf_func, C_tf, A_tf, B_tf, batch)

                call_tf()
                torch.cuda.synchronize()
                # TF wrote col-major [M,N] in [batch,N,M] storage; permute back
                C_tf_rm  = C_tf.permute(0, 2, 1).contiguous().cpu().numpy()
                ref_norm = np.max(np.abs(C_ref)) + 1e-30
                rel_err  = np.max(np.abs(C_tf_rm - C_ref)) / ref_norm
                ok = "✓" if rel_err < rtol else f"✗ rel_err={rel_err:.2e} (rtol={rtol:.0e})"
                t  = cuda_time(call_tf, warmup, repeats)
                gf = gflops(M, N, K, batch, t)
                print(f"  TensorForge [{dtype}]: {gf:8.1f} GFLOPs/s  "
                      f"({100*gf/peak:.1f}% of peak)  correctness {ok}")

    # ── Triton ────────────────────────────────────────────────────────────────
    if not skip_triton:
        if not HAS_TRITON:
            print(f"  Triton      [{dtype}]: not installed.")
        elif dtype == 'fp64' and not supports_fp64_tcdot():
            print(f"  Triton      [{dtype}]: skipped (fp64 tl.dot requires sm_80+, "
                  f"this GPU is sm_{torch.cuda.get_device_properties(0).major}"
                  f"{torch.cuda.get_device_properties(0).minor})")
        else:
            # BETA=0 so C's initial value doesn't matter; start from zeros
            C_triton = torch.zeros(batch, M, N, dtype=th_dtype, device='cuda')
            triton_tag = ""

            def call_triton():
                nonlocal triton_tag
                triton_tag = run_triton(A_gpu, B_gpu, C_triton, M, N, K, batch, dtype)

            # First call: JIT compile + compute result used for correctness check
            try:
                call_triton()
                torch.cuda.synchronize()
            except Exception as e:
                print(f"  Triton      [{dtype}]: JIT failed: {e}")
                return

            max_err  = np.max(np.abs(C_triton.cpu().numpy() - C_ref))
            ref_norm = np.max(np.abs(C_ref)) + 1e-30
            rel_err  = max_err / ref_norm
            ok = "✓" if rel_err < rtol else f"✗ rel_err={rel_err:.2e} (rtol={rtol:.0e})"
            t  = cuda_time(call_triton, warmup, repeats)
            gf = gflops(M, N, K, batch, t)
            print(f"  Triton      [{dtype}]: {gf:8.1f} GFLOPs/s  "
                  f"({100*gf/peak:.1f}% of peak)  correctness {ok}  [{triton_tag}]")


def bench_size(label, M, N, K, batch, warmup, repeats, arch, skip_tf, skip_triton):
    print(f"\n{'─'*65}")
    print(f"  {label}   batch={batch:,}")
    print(f"{'─'*65}")
    for dtype in ('fp32', 'fp64'):
        bench_dtype(label, M, N, K, batch, warmup, repeats, arch,
                    dtype, skip_tf, skip_triton)


# ─── main ────────────────────────────────────────────────────────────────────

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
    print(f"fp32 peak: {device_peak_gflops('fp32'):.0f} GFLOPs/s")
    print(f"fp64 peak: {device_peak_gflops('fp64'):.0f} GFLOPs/s")
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


if __name__ == "__main__":
    main()
