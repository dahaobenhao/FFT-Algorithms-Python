// g++ -O2 -std=c++17 test_mirror_b_tail.cpp -o test_mirror_b_tail
#include <vector>
#include <cassert>
#include <iostream>
#include <algorithm>

// ==== 被测函数：mirror_b_tail ====
// 输入/输出：b 为复数交织缓冲（长度至少 2*M）
// 约定：b 的前 N 个复数位置（下标 0..N-1）已写入；函数执行：
// for i=1..N-1: b[M - i] = b[i]
inline void mirror_b_tail(double* b, int N, int M) {
    for (int i = 1; i < N; ++i) {
        int k = M - i;
        double br = b[2*i], bi = b[2*i+1];
        b[2*k]   = br;
        b[2*k+1] = bi;
    }
}

// ==== 测试工具 ====
bool nearly_equal(double a, double b, double tol = 1e-12) {
    return std::abs(a - b) <= tol;
}

// ==== 单元测试 ====
void test_mirror_b_tail_basic() {
    // 造一个容易核对的数据集：
    // N=5, M=16（典型 Bluestein 会取 >= 2*N-1 的 2 次幂）
    // 先把 b[0..N-1] 写入明显的模式值：br = 10*i+1, bi = -10*i-2
    // 其它全 0。执行后应有：b[M-i] == b[i] 对 i=1..N-1。
    const int N = 5;
    const int M = 16;
    std::vector<double> b(2*M, 0.0);

    for (int i = 0; i < N; ++i) {
        double br = 10.0*i + 1.0;
        double bi = -10.0*i - 2.0;
        b[2*i]   = br;
        b[2*i+1] = bi;
    }

    mirror_b_tail(b.data(), N, M);

    for (int i = 1; i < N; ++i) {
        int k = M - i;
        // 实部 / 虚部分别镜像
        assert(nearly_equal(b[2*k],   10.0*i + 1.0));
        assert(nearly_equal(b[2*k+1], -10.0*i - 2.0));
    }

    // 额外检查：未被镜像覆盖且此前为 0 的位置仍保持 0
    // 例如 k= N..(M-N) 中大部分点
    for (int idx = N; idx < M - N; ++idx) {
        // 跳过可能与 k=M-i 重合的镜像目标
        bool is_mirror_target = false;
        for (int i = 1; i < N; ++i) if (idx == M - i) is_mirror_target = true;
        if (!is_mirror_target) {
            assert(nearly_equal(b[2*idx],   0.0));
            assert(nearly_equal(b[2*idx+1], 0.0));
        }
    }

    // b[0] 保持原值（未参与镜像）
    assert(nearly_equal(b[0],   1.0));
    assert(nearly_equal(b[1],  -2.0));
}

void test_mirror_b_tail_N2() {
    // 最小非平凡镜像：N=2
    // 期望：只会做 i=1 一次，把 b[M-1] 设为 b[1]
    const int N = 2;
    const int M = 8;
    std::vector<double> b(2*M, 0.0);

    // b[0] = (3, 4), b[1] = (5, 6)
    b[0] = 3.0; b[1] = 4.0;
    b[2] = 5.0; b[3] = 6.0;

    mirror_b_tail(b.data(), N, M);

    // 检查镜像目标 M-1 = 7
    assert(nearly_equal(b[2*(M-1)],   5.0));
    assert(nearly_equal(b[2*(M-1)+1], 6.0));

    // 其它未动
    assert(nearly_equal(b[0], 3.0));
    assert(nearly_equal(b[1], 4.0));
}

int main() {
    test_mirror_b_tail_basic();
    test_mirror_b_tail_N2();
    std::cout << "[mirror_b_tail] All tests passed.\n";
    return 0;
}