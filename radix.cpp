/*********************************************************************
 *  Radix-2 FFT (无内部 malloc/free) — 单文件实现
 *
 *  编译:
 *      g++ -std=c++17 -O3 radix.cpp -lm -o radix
 *********************************************************************/

#include <iostream>
#include <cmath>
#include <random>
#include <chrono>
#include <iomanip>
#include <cstring>
#include <cstdlib>

using namespace std;

/*======================================================================
  build_bit_reverse : 生成位逆序表
  ----------------------------------------------------------------------
  - rev 必须指向至少 N 个 size_t 的缓冲区
  - N 必须是 2 的幂
  - 无动态分配
======================================================================*/
void build_bit_reverse(size_t* rev, size_t N)
{
    size_t bits = static_cast<size_t>(log2(N));   // log2(N) 为整数
    for (size_t i = 0; i < N; ++i) {
        size_t x = i, r = 0;
        for (size_t j = 0; j < bits; ++j) {
            r = (r << 1) | (x & 1);
            x >>= 1;
        }
        rev[i] = r;
    }
}

/*======================================================================
  radix2_fft : 核心 FFT，不做任何内存分配
  ----------------------------------------------------------------------
  - x    : 长度 2N 的复数数组  [Re0, Im0, Re1, Im1, ...]
  - N    : 复数个数 (2 的幂)
  - rev  : 事先计算好的位逆序表，长度 N
  - tmp  : 临时缓冲，长度 2N (double)
======================================================================*/
void radix2_fft(double* x, size_t N,
                const size_t* rev,
                double* tmp)
{
    if (N == 0 || (N & (N - 1)))
        throw invalid_argument("Length must be a power of two.");

    /* ---------- 1) bit-reverse 重排 (利用外部 tmp) ---------- */
    std::memcpy(tmp, x, 2 * N * sizeof(double));
    for (size_t i = 0; i < N; ++i) {
        size_t r = rev[i];
        x[2*i]     = tmp[2*r];
        x[2*i + 1] = tmp[2*r + 1];
    }

    /* ---------- 2) 逐级蝴蝶运算 (不分配内存) --------------- */
    for (size_t m = 2; m <= N; m <<= 1) {
        size_t half = m >> 1;
        double theta_unit = -2.0 * M_PI / m;
        for (size_t start = 0; start < N; start += m) {
            for (size_t k = 0; k < half; ++k) {
                double wr = std::cos(theta_unit * k);
                double wi = std::sin(theta_unit * k);

                size_t i = start + k;
                size_t j = i + half;

                double ur = x[2*i];
                double ui = x[2*i + 1];
                double tr = x[2*j];
                double ti = x[2*j + 1];

                double tr_wr = tr * wr - ti * wi;
                double ti_wr = tr * wi + ti * wr;

                x[2*i]     = ur + tr_wr;
                x[2*i + 1] = ui + ti_wr;
                x[2*j]     = ur - tr_wr;
                x[2*j + 1] = ui - ti_wr;
            }
        }
    }
}

/*======================================================================
  benchmark_single_N : 基准测试（不触及 radix2_fft 内部分配）
======================================================================*/
void benchmark_single_N(size_t N,
                        mt19937& gen,
                        normal_distribution<>& dist)
{
    cout << "N = " << setw(9) << N << " : ";

    /* ---- 外部分配所有所需内存 ---- */
    double* data = static_cast<double*>(malloc(2 * N * sizeof(double)));  // 输入/输出
    double* tmp  = static_cast<double*>(malloc(2 * N * sizeof(double)));  // 临时缓冲
    size_t* rev  = static_cast<size_t*>(malloc(N * sizeof(size_t)));      // bit-reverse 表
    if (!data || !tmp || !rev) throw bad_alloc();

    /* ---- 填充随机数据 ---- */
    for (size_t i = 0; i < N; ++i) {
        data[2*i]     = dist(gen);
        data[2*i + 1] = dist(gen);
    }

    build_bit_reverse(rev, N);                 // 预计算位逆序表

    /* ---- 计时 FFT ---- */
    auto t0 = chrono::high_resolution_clock::now();
    radix2_fft(data, N, rev, tmp);
    auto t1 = chrono::high_resolution_clock::now();

    chrono::duration<double> dt = t1 - t0;
    cout << fixed << setprecision(4)
         << "t_fft = " << setw(8) << dt.count() * 1000 << " ms\n";

    free(data);
    free(tmp);
    free(rev);
}

/*======================================================================
  main : 从 2^0 到 2^max_pow 做基准
======================================================================*/
int main(int argc, char** argv)
{
    int max_pow = 24;                      // 默认测试到 2^24
    if (argc > 1) max_pow = std::stoi(argv[1]);

    random_device rd;
    mt19937 gen(rd());
    normal_distribution<> dist(0.0, 1.0);

    cout << "\nRadix-2 FFT benchmark (no internal malloc/free)\n";
    cout << "------------------------------------------------\n";

    for (int p = 0; p <= max_pow; ++p) {
        size_t N = 1ull << p;
        try {
            benchmark_single_N(N, gen, dist);
        }
        catch (const bad_alloc&) {
            cerr << "Out of memory @ N = " << N << " — terminating\n";
            break;
        }
    }
    return 0;
}
