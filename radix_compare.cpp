#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <chrono>
#include <random>
#include <algorithm>
#include <stdexcept>
#include <fftw3.h>

using namespace std;
using Complex = complex<double>;
using Vec = vector<Complex>;

// --------- Bit-Reversal --------- //
vector<size_t> bit_reverse_indices(size_t n) {
    size_t bits = static_cast<size_t>(log2(n));
    vector<size_t> rev(n);
    for (size_t i = 0; i < n; ++i) {
        size_t x = i;
        size_t r = 0;
        for (size_t j = 0; j < bits; ++j) {
            r = (r << 1) | (x & 1);
            x >>= 1;
        }
        rev[i] = r;
    }
    return rev;
}

// --------- Radix-2 Iterative FFT --------- //
void radix2_fft_iter(Vec& x) {
    size_t N = x.size();
    if ((N & (N - 1)) != 0) {
        throw invalid_argument("Input size must be a power of 2.");
    }

    // Bit-reversal permutation
    auto rev = bit_reverse_indices(N);
    Vec x_copy = x;
    for (size_t i = 0; i < N; ++i) {
        x[i] = x_copy[rev[i]];
    }

    // Iterative Cooley–Tukey FFT
    for (size_t m = 2; m <= N; m <<= 1) {
        size_t half = m / 2;
        Vec W(half);
        for (size_t k = 0; k < half; ++k) {
            double angle = -2.0 * M_PI * k / m;
            W[k] = Complex(cos(angle), sin(angle));
        }

        for (size_t start = 0; start < N; start += m) {
            for (size_t k = 0; k < half; ++k) {
                Complex u = x[start + k];
                Complex t = x[start + k + half] * W[k];
                x[start + k] = u + t;
                x[start + k + half] = u - t;
            }
        }
    }
}

// --------- Test & FFTW Comparison --------- //
void test_large_radix(size_t N) {
    cout << "\n🚀 测试 Radix‑2 FFT, 长度 N = " << N << endl;

    // 随机输入
    random_device rd;
    mt19937 gen(rd());
    normal_distribution<> dist(0.0, 1.0);
    Vec x(N);
    for (size_t i = 0; i < N; ++i) {
        x[i] = Complex(dist(gen), dist(gen));
    }

    // 自写 FFT
    Vec x_fft = x;
    auto start_radix = chrono::high_resolution_clock::now();
    radix2_fft_iter(x_fft);
    auto end_radix = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed_radix = end_radix - start_radix;
    cout << "✅ 完成 Radix‑2 FFT，耗时：" << elapsed_radix.count() << " 秒" << endl;

    // FFTW 准备输入
    vector<fftw_complex> in(N), out(N);
    for (size_t i = 0; i < N; ++i) {
        in[i][0] = x[i].real();
        in[i][1] = x[i].imag();
    }

    // 创建 plan
    auto start_plan = chrono::high_resolution_clock::now();
    fftw_plan p = fftw_plan_dft_1d(
        static_cast<int>(N), in.data(), out.data(), FFTW_FORWARD, FFTW_ESTIMATE);
    auto end_plan = chrono::high_resolution_clock::now();
    chrono::duration<double> plan_time = end_plan - start_plan;

    // 执行 FFTW
    auto start_fftw = chrono::high_resolution_clock::now();
    fftw_execute(p);
    auto end_fftw = chrono::high_resolution_clock::now();
    fftw_destroy_plan(p);
    chrono::duration<double> elapsed_fftw = end_fftw - start_fftw;

    cout << "✅ 完成 FFTW FFT，耗时：" << elapsed_fftw.count() << " 秒 (不含 plan: plan 耗时 " << plan_time.count() << " 秒)" << endl;

    // 误差对比
    double max_rel_err = 0.0;
    for (size_t i = 0; i < N; ++i) {
        Complex ref(out[i][0], out[i][1]);
        double denom = abs(ref);
        if (denom < 1e-12) denom = 1e-12;
        double rel_err = abs(x_fft[i] - ref) / denom;
        max_rel_err = max(max_rel_err, rel_err);
    }

    cout << "🎯 最大相对误差 = " << max_rel_err << endl;

    // 速度对比
    double speedup = elapsed_fftw.count() / elapsed_radix.count();
    cout << "⚡️ 速度比 (Radix2 / FFTW) = " << speedup << "x\n";
}


// --------- Main --------- //
int main() {
    vector<size_t> sizes = { 1 << 20, 1 << 22, 1 << 24 };  // 1M, 4M, 16M
    for (size_t N : sizes) {
        test_large_radix(N);
    }
    return 0;
}