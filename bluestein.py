import numpy as np
import time

# def bluestein_fft(x: np.ndarray) -> np.ndarray:
#     x = np.asarray(x, dtype=complex)
#     N = x.size
#     M = 1 << (2 * N - 1).bit_length()
#     n = np.arange(N)
#     w = np.exp(-1j * np.pi * n * n / N)

#     a = np.zeros(M, dtype=complex)
#     a[:N] = x * w

#     b = np.zeros(M, dtype=complex)
#     b[:N] = np.exp(1j * np.pi * n * n / N)
#     if N > 1:
#         b[-(N - 1):] = b[1:N][::-1]

#     A = np.fft.fft(a)
#     B = np.fft.fft(b)
#     C = A * B
#     c = np.fft.ifft(C)

#     return c[:N] * w

def bluestein_fft(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=complex)
    N = x.size
    M = 1 << (2 * N - 1).bit_length()
    n = np.arange(N)
    
    w = np.exp(-1j * np.pi * n * n / N)  # Chirp
    w_conj = np.conj(w)  # Instead of recomputing exp(+j...)

    a = np.zeros(M, dtype=complex)
    a[:N] = x * w

    b = np.zeros(M, dtype=complex)
    b[:N] = w_conj
    if N > 1:
        b[-(N - 1):] = w_conj[1:N][::-1]  # mirror of w_conj[1:]

    A = np.fft.fft(a)
    B = np.fft.fft(b)
    C = A * B
    c = np.fft.ifft(C)

    return c[:N] * w


def test_large_bluestein(N):
    print(f"\n🚀 测试大规模 Bluestein FFT, 长度 N = {N}")
    x = np.random.randn(N) + 1j * np.random.randn(N)

    start = time.perf_counter()
    X_b = bluestein_fft(x)
    elapsed = time.perf_counter() - start

    print(f"✅ 完成 Bluestein FFT，耗时：{elapsed:.2f} 秒")

    try:
        X_np = np.fft.fft(x)
        # 最大相对误差（避免除以零）
        relative_err = np.max(np.abs(X_b - X_np) / np.maximum(np.abs(X_np), 1e-12))
        print(f"最大相对误差 = {relative_err:.2e}")
    except MemoryError:
        print("⚠️ 无法验证 NumPy FFT（可能内存不足）")

if __name__ == "__main__":
    test_large_bluestein(1_000_007)
    test_large_bluestein(5_000_001)
    test_large_bluestein(100_000_003)  # 可根据内存情况打开