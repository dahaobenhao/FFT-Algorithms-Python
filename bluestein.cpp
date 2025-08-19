// g++ -O2 -std=c++17 bluestein.cpp -o bluestein
#include <cmath>
#include <iostream>
#include <random>
#include <vector>
#include <limits>
#include <cassert>

namespace {
constexpr double PI() { return 3.141592653589793238462643383279502884; }

// 计算 M: 最小的 2^k >= 2*N - 1
int calc_bluestein_M(int N) {
    int M = 1;
    while (M < 2 * N - 1) M <<= 1;
    return M;
}

// ---- 小函数拆分：chirp + 前缀填充 + 对称填充 ----
inline void chirp(int n, int N, double& wr, double& wi) {
    // w[n] = exp(-i * pi * n^2 / N)
    double angle = -PI() * (double)n * (double)n / (double)N;
    wr = std::cos(angle);
    wi = std::sin(angle);
}

inline void fill_ab_prefix(const double* x, int N, double* a, double* b) {
    for (int n = 0; n < N; ++n) {
        double wr, wi; chirp(n, N, wr, wi);
        // a[n] = x[n] * w[n]
        a[2*n]   = x[n] * wr;
        a[2*n+1] = x[n] * wi;
        // b[n] = conj(w[n])
        b[2*n]   = wr;
        b[2*n+1] = -wi;
    }
}

inline void mirror_b_tail(double* b, int N, int M) {
    // b[M - i] = b[i], for i=1..N-1
    for (int i = 1; i < N; ++i) {
        int k = M - i;
        double br = b[2*i], bi = b[2*i+1];
        b[2*k]   = br;
        b[2*k+1] = bi;
    }
}

// 朴素 DFT（复数交织）
void naive_dft(const double* in, int M, bool inverse, double* out) {
    const double scale = inverse ? (1.0 / M) : 1.0;
    const double sgn = inverse ? +1.0 : -1.0; // exp(i*sgn*2πnk/M)
    for (int k = 0; k < M; ++k) {
        double sumRe = 0.0, sumIm = 0.0;
        for (int n = 0; n < M; ++n) {
            double xr = in[2*n];
            double xi = in[2*n+1];
            double angle = (2.0 * PI()) * (double)n * (double)k / (double)M;
            double wr = std::cos(angle);
            double wi = std::sin(angle);
            // x[n] * (wr + i*sgn*wi)
            double tr = xr * wr - xi * (sgn * wi);
            double ti = xr * (sgn * wi) + xi * wr;
            sumRe += tr;
            sumIm += ti;
        }
        out[2*k]   = scale * sumRe;
        out[2*k+1] = scale * sumIm;
    }
}

// 点乘：C = A .* B（逐点复数乘法，交织格式）
void pointwise_mul(const double* A, const double* B, int M, double* C) {
    for (int k = 0; k < M; ++k) {
        double ar = A[2*k], ai = A[2*k+1];
        double br = B[2*k], bi = B[2*k+1];
        C[2*k]   = ar*br - ai*bi;
        C[2*k+1] = ar*bi + ai*br;
    }
}

// Bluestein：对实数输入 x（长度 N）计算 N 点 DFT，输出 out（复数交织，长度 N）
// 所有缓冲由外部分配；a、b 需在外部（main）先清零。
void bluestein_dft_real(
    const double* x, int N, int M,
    double* out,     // 2*N
    double* a,       // 2*M (cleared outside)
    double* b,       // 2*M (cleared outside)
    double* A,       // 2*M
    double* B,       // 2*M
    double* C,       // 2*M
    double* c        // 2*M
) {
    // 直接调用两个小函数：写前缀 + 对称填充
    fill_ab_prefix(x, N, a, b);
    mirror_b_tail(b, N, M);

    // DFT_M(a), DFT_M(b)
    naive_dft(a, M, /*inverse=*/false, A);
    naive_dft(b, M, /*inverse=*/false, B);

    // 频域点乘 -> 卷积
    pointwise_mul(A, B, M, C);

    // 逆变换 -> 时域卷积
    naive_dft(C, M, /*inverse=*/true, c);

    // 取前 N 点，并乘 w[k]
    for (int k = 0; k < N; ++k) {
        double wr, wi; chirp(k, N, wr, wi); // w[k]
        double cr = c[2*k], ci = c[2*k+1];
        out[2*k]   = cr*wr - ci*wi;
        out[2*k+1] = cr*wi + ci*wr;
    }
}

// 直接 DFT（用于对拍）：实数输入 x[N] -> 复数输出 out[N]
void direct_dft_real(const double* x, int N, double* out) {
    for (int k = 0; k < N; ++k) {
        double sumRe = 0.0, sumIm = 0.0;
        for (int n = 0; n < N; ++n) {
            double angle = (2.0 * PI()) * (double)n * (double)k / (double)N;
            double wr = std::cos(angle);
            double wi = std::sin(angle);
            sumRe += x[n] * wr;
            sumIm += -x[n] * wi; // exp(-i*angle)
        }
        out[2*k]   = sumRe;
        out[2*k+1] = sumIm;
    }
}

// 计算最大绝对误差（复数向量，交织格式）
double max_abs_err(const double* A, const double* B, int N) {
    double maxe = 0.0;
    for (int i = 0; i < N; ++i) {
        double er = A[2*i]   - B[2*i];
        double ei = A[2*i+1] - B[2*i+1];
        double e = std::hypot(er, ei);
        if (e > maxe) maxe = e;
    }
    return maxe;
}

} // namespace

int main(int argc, char** argv) {
    int N = 17; // 任意长度
    if (argc >= 2) N = std::max(1, std::atoi(argv[1]));
    int M = calc_bluestein_M(N);

    std::mt19937_64 rng(12345);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    std::vector<double> x(N);
    for (int i = 0; i < N; ++i) x[i] = dist(rng);

    // 在 main 中一次性分配所有缓冲区
    std::vector<double> a(2*M), b(2*M), A(2*M), B(2*M), C(2*M), c(2*M);
    std::vector<double> X_blu(2*N), X_dir(2*N);

    // 只在 main 清零 a、b（其余缓冲无需清零）
    std::fill(a.begin(), a.end(), 0.0);
    std::fill(b.begin(), b.end(), 0.0);

    bluestein_dft_real(x.data(), N, M,
                       X_blu.data(),
                       a.data(), b.data(), A.data(), B.data(), C.data(), c.data());

    direct_dft_real(x.data(), N, X_dir.data());

    double err = max_abs_err(X_blu.data(), X_dir.data(), N);

    std::cout << "N = " << N << ", M = " << M
              << ", max |Δ| = " << err << "\n";

    // 打印前若干点，便于目视检查
    int show = std::min(N, 8);
    std::cout << "k :  Bluestein(real,imag)    Direct(real,imag)\n";
    for (int k = 0; k < show; ++k) {
        double br = X_blu[2*k], bi = X_blu[2*k+1];
        double dr = X_dir[2*k], di = X_dir[2*k+1];
        std::cout << k << ": (" << br << ", " << bi << ")    ("
                  << dr << ", " << di << ")\n";
    }
    return 0;
}