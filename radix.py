import numpy as np
import time

# ---------------- helpers ---------------- #

def _bit_reverse_indices(n: int) -> np.ndarray:
    """Bit‑reversal permutation for power‑of‑two *n* (vectorised, 32‑bit)."""
    bits = n.bit_length() - 1
    rev = np.arange(n, dtype=np.uint32)
    rev = ((rev & 0x55555555) << 1) | ((rev & 0xAAAAAAAA) >> 1)
    rev = ((rev & 0x33333333) << 2) | ((rev & 0xCCCCCCCC) >> 2)
    rev = ((rev & 0x0F0F0F0F) << 4) | ((rev & 0xF0F0F0F0) >> 4)
    rev = ((rev & 0x00FF00FF) << 8) | ((rev & 0xFF00FF00) >> 8)
    rev = (rev << 16) | (rev >> 16)
    rev >>= 32 - bits
    return rev.astype(np.intp)


# ---------------- radix‑2 (DIT) FFT ---------------- #

def radix2_fft_iter(x: np.ndarray) -> np.ndarray:
    """Non‑recursive, in‑place radix‑2 Cooley–Tukey FFT.

    Parameters
    ----------
    x : array_like
        Complex input whose *length must be an exact power of two*.

    Returns
    -------
    ndarray (complex)
        The discrete Fourier transform of *x* (same buffer when possible).
    """
    x = np.asarray(x, dtype=complex)
    N = x.size
    if N & (N - 1):
        raise ValueError("Input length must be a power of two for radix‑2 FFT.")

    # 1. Bit‑reverse permutation
    x = x[_bit_reverse_indices(N)]

    # 2. Iterative Danielson–Lanczos stages (DIT)
    m = 2
    while m <= N:
        half = m // 2
        W = np.exp(-2j * np.pi * np.arange(half) / m)  # twiddle table for this stage
        for start in range(0, N, m):
            idx1 = slice(start, start + half)
            idx2 = slice(start + half, start + m)

            u = x[idx1].copy()       # **copy!** – avoid overwriting while still in use
            t = x[idx2] * W          # twiddled second half

            x[idx1] = u + t          # butterfly
            x[idx2] = u - t
        m <<= 1  # ×2 per stage
    return x


# ---------------- quick benchmark / accuracy check ---------------- #

def test_large_radix(N):
    print(f"\n🚀 测试大规模 Radix‑2 FFT, 长度 N = {N}")
    x = np.random.randn(N) + 1j * np.random.randn(N)

    start = time.perf_counter()
    X_r = radix2_fft_iter(x.copy())
    elapsed = time.perf_counter() - start
    print(f"✅ 完成 Radix‑2 FFT，耗时：{elapsed:.2f} 秒")

    X_np = np.fft.fft(x)
    rel_err = np.max(np.abs(X_r - X_np) / np.maximum(np.abs(X_np), 1e-12))
    print(f"最大相对误差 = {rel_err:.2e}")


if __name__ == "__main__":
    for N in (1 << 20, 1 << 22, 1 << 24):  # 1 M, 4 M, 16 M
        test_large_radix(N)