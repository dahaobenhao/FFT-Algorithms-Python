// g++ -std=c++17 -O3 radix_compare.cpp -lfftw3 -lm -o radix_compare

#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <chrono>
#include <iomanip>
#include <fftw3.h>
#include <stdexcept>
#include <algorithm>

using namespace std;

// ------------- Bit-reversal permutation table ------------- //
static vector<size_t> bit_reverse_indices(size_t n) {
    size_t bits = static_cast<size_t>(log2(n));
    vector<size_t> rev(n);
    for (size_t i = 0; i < n; ++i) {
        size_t x = i, r = 0;
        for (size_t j = 0; j < bits; ++j) {
            r = (r << 1) | (x & 1);
            x >>= 1;
        }
        rev[i] = r;
    }
    return rev;
}

// ------------- Radix-2 iterative FFT (in-place) ------------ //
void radix2_fft_iter_real(vector<double>& x) {
    size_t N = x.size() / 2;            // complex length
    if (N == 0 || (N & (N - 1))) {
        throw invalid_argument("Length must be a positive power of two.");
    }

    // Bit-reversal copy
    static vector<size_t> rev;          // cached between calls
    if (rev.size() != N) rev = bit_reverse_indices(N);

    vector<double> tmp = x;
    for (size_t i = 0; i < N; ++i) {
        size_t r = rev[i];
        x[2*i]     = tmp[2*r];
        x[2*i + 1] = tmp[2*r + 1];
    }

    // Cooley–Tukey iterations
    for (size_t m = 2; m <= N; m <<= 1) {
        size_t half = m >> 1;
        double theta = -2.0 * M_PI / m;
        for (size_t start = 0; start < N; start += m) {
            for (size_t k = 0; k < half; ++k) {
                double wr = cos(theta * k);
                double wi = sin(theta * k);

                size_t i = start + k;
                size_t j = i + half;

                double ur = x[2*i];
                double ui = x[2*i + 1];
                double tr = x[2*j];
                double ti = x[2*j + 1];

                // t * W
                double tr_wr = tr * wr - ti * wi;
                double ti_wr = tr * wi + ti * wr;

                // butterfly
                x[2*i]     = ur + tr_wr;
                x[2*i + 1] = ui + ti_wr;
                x[2*j]     = ur - tr_wr;
                x[2*j + 1] = ui - ti_wr;
            }
        }
    }
}

// ------------- One size test (against FFTW) --------------- //
void test_size(size_t N, mt19937& gen, normal_distribution<>& dist) {
    cout << "N = " << setw(10) << N << " : ";

    vector<double> data(2*N);
    for (size_t i = 0; i < N; ++i) {
        data[2*i]     = dist(gen);
        data[2*i + 1] = dist(gen);
    }

    vector<double> mine = data;

    auto t0 = chrono::high_resolution_clock::now();
    radix2_fft_iter_real(mine);
    auto t1 = chrono::high_resolution_clock::now();
    chrono::duration<double> dt_mine = t1 - t0;

    // ---- FFTW reference ----- //
    vector<fftw_complex> in(N), out(N);
    for (size_t i = 0; i < N; ++i) {
        in[i][0] = data[2*i];
        in[i][1] = data[2*i + 1];
    }
    fftw_plan plan = fftw_plan_dft_1d(static_cast<int>(N), in.data(), out.data(), FFTW_FORWARD, FFTW_ESTIMATE);
    auto t2 = chrono::high_resolution_clock::now();
    fftw_execute(plan);
    auto t3 = chrono::high_resolution_clock::now();
    fftw_destroy_plan(plan);
    chrono::duration<double> dt_fftw = t3 - t2;

    // ---- accuracy ----- //
    double max_rel_err = 0.0;
    for (size_t i = 0; i < N; ++i) {
        double ref_r = out[i][0];
        double ref_i = out[i][1];
        double ref_mag = hypot(ref_r, ref_i);
        if (ref_mag < 1e-12) ref_mag = 1e-12;
        double diff = hypot(mine[2*i] - ref_r, mine[2*i + 1] - ref_i);
        max_rel_err = max(max_rel_err, diff / ref_mag);
    }

    cout << fixed << setprecision(4)
         << "t_my = " << setw(9) << dt_mine.count()*1000 << " ms, "
         << "t_fftw = " << setw(9) << dt_fftw.count()*1000 << " ms, "
         << "speedup = " << setprecision(2) << dt_fftw.count()/dt_mine.count() << "x, "
         << scientific << "maxErr = " << max_rel_err << '\n';
}

// ------------- Main driver -------------------------------- //
int main(int argc, char** argv) {
    int max_pow = 24;              // default 2^0 .. 2^24 (16,777,216)
    if (argc > 1) max_pow = stoi(argv[1]);

    random_device rd;
    mt19937 gen(rd());
    normal_distribution<> dist(0.0, 1.0);

    cout << "\nRadix-2 FFT (double array) vs FFTW — accuracy & speed\n";
    cout << "---------------------------------------------------\n";
    for (int p = 0; p <= max_pow; ++p) {
        size_t N = 1ull << p;
        try {
            test_size(N, gen, dist);
        } catch (const bad_alloc&) {
            cerr << "Out of memory for N=" << N << " — stopping tests\n";
            break;
        }
    }
    return 0;
}