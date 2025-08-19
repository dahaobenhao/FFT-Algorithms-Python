// g++ -O2 -std=c++17 test_fill_ab_prefix.cpp -o test_fill_ab_prefix
#include <cmath>
#include <vector>
#include <cassert>
#include <iostream>
#include <algorithm>

// ==== 被测函数：fill_ab_prefix ====
// 输入：x[N]（实数）
// 输出：a[2*N], b[2*N]（复数交织：re,im）
// 其中 w[n] = exp(-i * pi * n^2 / N)
// a[n] = x[n] * w[n]
// b[n] = conj(w[n])
inline void fill_ab_prefix(const double* x, int N, double* a, double* b) {
    for (int n = 0; n < N; ++n) {
        const double angle = -(std::atan2(0.0, -1.0)) * (double)n * (double)n / (double)N; // -π n^2 / N
        const double wr = std::cos(angle);
        const double wi = std::sin(angle);
        a[2*n]   = x[n] * wr;
        a[2*n+1] = x[n] * wi;
        b[2*n]   = wr;
        b[2*n+1] = -wi;
    }
}

// ==== 测试工具 ====
bool nearly_equal(double a, double b, double tol = 1e-12) {
    return std::abs(a - b) <= tol;
}
void expect_array_near(const std::vector<double>& got,
                       const std::vector<double>& expect,
                       double tol = 1e-12) {
    assert(got.size() == expect.size());
    for (size_t i = 0; i < got.size(); ++i) {
        if (!nearly_equal(got[i], expect[i], tol)) {
            std::cerr << "Mismatch at idx " << i
                      << ": got " << got[i]
                      << ", expect " << expect[i] << "\n";
            assert(false);
        }
    }
}

// ==== 单元测试 ====
void test_fill_ab_prefix_basic() {
    // 造数据：固定输入，便于回归
    // N=5, x = [1.0, -0.5, 0.25, 0.0, -1.25]
    const int N = 5;
    const double x_arr[N] = { 1.0, -0.5, 0.25, 0.0, -1.25 };
    std::vector<double> a(2*N, 0.0), b(2*N, 0.0);

    fill_ab_prefix(x_arr, N, a.data(), b.data());

    // 期望值（双精度预计算，π 用 atan2 方式推导）：
    // a[n] = x[n] * exp(-i*pi*n^2/N)
    // b[n] = conj(exp(-i*pi*n^2/N))
    const std::vector<double> a_expect = {
        // n=0
        1.0, 0.0,
        // n=1
        -0.4045084971874737,  0.29389262614623657,
        // n=2
        -0.20225424859373684, -0.1469463130731183,
        // n=3
        0.0, 0.0,
        // n=4
        1.0112712429686845,  -0.734731565365591
    };
    const std::vector<double> b_expect = {
        // n=0
        1.0, 0.0,
        // n=1
        0.8090169943749475,   0.5877852522924731,
        // n=2
        -0.8090169943749473,  0.5877852522924732,
        // n=3
        0.8090169943749473,  -0.5877852522924734,
        // n=4
        -0.8090169943749477, -0.5877852522924728
    };

    expect_array_near(a, a_expect);
    expect_array_near(b, b_expect);
}

void test_fill_ab_prefix_edge_N1() {
    // 边界：N=1，只会用到 n=0，要求 w[0]=1
    const int N = 1;
    const double x_arr[N] = { 2.5 };
    std::vector<double> a(2*N, 0.0), b(2*N, 0.0);

    fill_ab_prefix(x_arr, N, a.data(), b.data());

    assert(nearly_equal(a[0], 2.5));
    assert(nearly_equal(a[1], 0.0));
    assert(nearly_equal(b[0], 1.0));
    assert(nearly_equal(b[1], 0.0));
}

int main() {
    test_fill_ab_prefix_basic();
    test_fill_ab_prefix_edge_N1();
    std::cout << "[fill_ab_prefix] All tests passed.\n";
    return 0;
}