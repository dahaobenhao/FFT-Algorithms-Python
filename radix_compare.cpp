/*********************************************************************
 *  Radix-2 FFT (无内部 malloc/free) vs FFTW 3
 *
 *  编译:
 *      g++ -std=c++17 -O3 radix_compare.cpp -lfftw3 -lm -o radix_compare
 *********************************************************************/

#include <iostream>
#include <cmath>
#include <random>
#include <chrono>
#include <iomanip>
#include <cstring>
#include <cstdlib>
#include <fftw3.h>

using namespace std;

/*======================================================================
  build_bit_reverse : 预计算位逆序表（外部缓冲）
======================================================================*/
void build_bit_reverse(size_t* rev, size_t N)
{
    size_t bits = static_cast<size_t>(log2(N));
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
  radix2_fft : 核心 FFT（无内部分配）
======================================================================*/
void radix2_fft(double* x, size_t N,
                const size_t* rev,
                double* tmp)
{
    if (N == 0 || (N & (N - 1)))
        throw invalid_argument("Length must be a power of two.");

    /* ---------- 1) bit-reverse 置换 ---------- */
    memcpy(tmp, x, 2 * N * sizeof(double));
    for (size_t i = 0; i < N; ++i) {
        size_t r = rev[i];
        x[2*i]     = tmp[2*r];
        x[2*i + 1] = tmp[2*r + 1];
    }

    /* ---------- 2) 迭代蝴蝶 ---------- */
    for (size_t m = 2; m <= N; m <<= 1) {
        size_t half = m >> 1;
        double theta_unit = -2.0 * M_PI / m;
        for (size_t start = 0; start < N; start += m) {
            for (size_t k = 0; k < half; ++k) {
                double wr = cos(theta_unit * k);
                double wi = sin(theta_unit * k);

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
  benchmark_N : 比较自定义 FFT 与 FFTW (最大相对误差 & 速度)
======================================================================*/
void benchmark_N(size_t N,
                 mt19937& gen,
                 normal_distribution<>& dist)
{
    cout << "N = " << setw(9) << N << " : ";

    /* ---- 外部分配 ---- */
    double* data  = static_cast<double*>(malloc(2 * N * sizeof(double))); // 原始数据
    double* mine  = static_cast<double*>(malloc(2 * N * sizeof(double))); // 自定义 FFT 输入/输出
    double* tmp   = static_cast<double*>(malloc(2 * N * sizeof(double))); // 临时缓冲
    size_t* rev   = static_cast<size_t*>(malloc(N * sizeof(size_t)));     // bit-reverse
    if (!data || !mine || !tmp || !rev) throw bad_alloc();

    /* ---- 随机填充 ---- */
    for (size_t i = 0; i < N; ++i) {
        double re = dist(gen), im = dist(gen);
        data[2*i] = mine[2*i] = re;
        data[2*i+1] = mine[2*i+1] = im;
    }
    build_bit_reverse(rev, N);

    /* ---- 自定义 FFT ---- */
    auto t0 = chrono::high_resolution_clock::now();
    radix2_fft(mine, N, rev, tmp);
    auto t1 = chrono::high_resolution_clock::now();

    /* ---- FFTW 参考 ---- */
    fftw_complex* in  = (fftw_complex*)fftw_malloc(sizeof(fftw_complex)*N);
    fftw_complex* out = (fftw_complex*)fftw_malloc(sizeof(fftw_complex)*N);
    for (size_t i = 0; i < N; ++i) {
        in[i][0] = data[2*i];
        in[i][1] = data[2*i+1];
    }
    fftw_plan plan = fftw_plan_dft_1d(static_cast<int>(N), in, out,
                                      FFTW_FORWARD, FFTW_ESTIMATE);
    auto t2 = chrono::high_resolution_clock::now();
    fftw_execute(plan);
    auto t3 = chrono::high_resolution_clock::now();

    chrono::duration<double> dt_my   = t1 - t0;
    chrono::duration<double> dt_fftw = t3 - t2;

    /* ---- 误差 ---- */
    double max_rel_err = 0.0;
    for (size_t i = 0; i < N; ++i) {
        double ref_r = out[i][0], ref_i = out[i][1];
        double ref_mag = hypot(ref_r, ref_i);
        if (ref_mag < 1e-12) ref_mag = 1e-12;
        double diff = hypot(mine[2*i]-ref_r, mine[2*i+1]-ref_i);
        double rel_err = diff / ref_mag;
        if (rel_err > max_rel_err) max_rel_err = rel_err;
    }

    fftw_destroy_plan(plan);
    fftw_free(in);  fftw_free(out);
    free(data); free(mine); free(tmp); free(rev);

    /* ---- 输出结果 ---- */
    cout << fixed << setprecision(4)
         << "t_my = "   << setw(8) << dt_my.count()*1000 << " ms, "
         << "t_fftw = " << setw(8) << dt_fftw.count()*1000 << " ms, "
         << "speedup = " << setprecision(2) << dt_fftw.count()/dt_my.count() << "x, "
         << scientific << "maxErr = " << max_rel_err << '\n';
}

/*======================================================================
  main
======================================================================*/
int main(int argc, char** argv)
{
    int max_pow = 24;                  // 默认到 2^24
    if (argc > 1) max_pow = stoi(argv[1]);

    random_device rd;
    mt19937 gen(rd());
    normal_distribution<> dist(0.0, 1.0);

    cout << "\nRadix-2 FFT vs FFTW (no internal malloc in custom)\n";
    cout << "----------------------------------------------------\n";

    for (int p = 0; p <= max_pow; ++p) {
        size_t N = 1ull << p;
        try {
            benchmark_N(N, gen, dist);
        } catch (const bad_alloc&) {
            cerr << "Out of memory @ N = " << N << " — terminating\n";
            break;
        }
    }
    return 0;
}