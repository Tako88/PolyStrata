// src/poly/mamba_parity.cu - mamba_conv(), mamba_scan() and mamba_norm() against the same arithmetic on the host:
// the block's convolution output, its post-scan y and its normed output, the state it leaves, rows=3 against three
// rows=1 calls with the snapshots restored in between, and a 9-token call against nine rows=1 calls (the property
// a --window 8 --lookup 3 replay depends on: a token comes out the same whatever tokens the call has beside it).
//
// The reference is the formulas of llama.cpp's mamba-base graph and its CPU scan, with the model's own shapes:
// dim 64 over 128 heads, state 128, 8 groups, 4 taps (gguf-facts.txt).
//
// usage: mamba_parity   (exit 0: everything within a few units in the last place)
#include "strata/poly/kernels.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

namespace {

using namespace strata::poly;

constexpr int kDim = 64, kHead = 128, kState = 128, kGroup = 8;
constexpr int kInner = kDim * kHead;                                        // 8192
constexpr int kConv = kInner + 2 * kGroup * kState;                         // 10240
constexpr int kProj = 2 * kInner + 2 * kGroup * kState + kHead;             // 18560
constexpr int kGroups = kInner / kGroup;                                    // 1024
constexpr float kEps = 1e-5f;

float sigmoidf(float x) { return 1.f / (1.f + expf(-x)); }
float softplus(float x) { return x > 20.f ? x : logf(1.f + expf(x)); }

std::vector<float>& fbuf(std::vector<float>& v, size_t n) {
    v.assign(n, 0.f);
    return v;
}

void host_block(const std::vector<float>& proj, const std::vector<float>& w, const std::vector<float>& bias,
                const std::vector<float>& dt_bias, const std::vector<float>& a, const std::vector<float>& d,
                const std::vector<float>& norm, std::vector<float>& conv_state, std::vector<float>& scan_state,
                int rows, std::vector<float>& conv_out, std::vector<float>& y_add_d, std::vector<float>& y,
                std::vector<float>* snap_conv, std::vector<float>* snap_scan, unsigned snaps) {
    const int dt_off = 2 * kInner + 2 * kGroup * kState;
    fbuf(conv_out, (size_t) rows * kConv);
    fbuf(y_add_d, (size_t) rows * kInner);
    fbuf(y, (size_t) rows * kInner);
    int snap_at = 0;
    for (int row = 0; row < rows; ++row) {
        const float* p = proj.data() + (size_t) row * kProj;
        for (int c = 0; c < kConv; ++c) {   // the depthwise convolution, its bias and its SiLU
            float* st = conv_state.data() + (size_t) c * 3;
            const float s0 = st[0], s1 = st[1], s2 = st[2];
            const float v = p[kInner + c];
            const float yc = fmaf(s0, w[(size_t) c * 4], fmaf(s1, w[(size_t) c * 4 + 1],
                             fmaf(s2, w[(size_t) c * 4 + 2], fmaf(v, w[(size_t) c * 4 + 3], bias[c]))));
            st[0] = st[1], st[1] = st[2], st[2] = v;
            conv_out[(size_t) row * kConv + c] = yc * (float) sigmoidf(yc);
        }
        const float* xbc = conv_out.data() + (size_t) row * kConv;
        for (int h = 0; h < kHead; ++h) {   // the selective scan, y += x * ssm_d folded in
            const float dt = p[dt_off + h] + dt_bias[h];
            const float dts = (float) softplus(dt);
            const float dA = expf(dts * a[h]);
            const int g = h / (kHead / kGroup);
            for (int i = 0; i < kDim; ++i) {
                const float xv = xbc[h * kDim + i];
                const float x_dt = xv * dts;
                float* s = scan_state.data() + (size_t) (h * kDim + i) * kState;
                float sum = 0.f;
                for (int j = 0; j < kState; ++j) {
                    const float st = fmaf(s[j], dA, xbc[kInner + g * kState + j] * x_dt);
                    s[j] = st;
                    sum = fmaf(st, xbc[kInner + kGroup * kState + g * kState + j], sum);
                }
                y_add_d[(size_t) row * kInner + h * kDim + i] = sum + d[h] * xv;
            }
        }
        if (snap_conv && (snaps >> row & 1)) {
            std::copy(conv_state.begin(), conv_state.end(), snap_conv->begin() + (size_t) snap_at * 3 * kConv);
            std::copy(scan_state.begin(), scan_state.end(), snap_scan->begin() + (size_t) snap_at * kInner * kState);
            ++snap_at;
        }
        for (int g = 0; g < kGroup; ++g) {   // silu(z) * y, then the grouped RMS norm
            float ss = 0.f;
            for (int i = 0; i < kGroups; ++i) {
                const float v = p[g * kGroups + i] * (float) sigmoidf(p[g * kGroups + i]) *
                                y_add_d[(size_t) row * kInner + g * kGroups + i];
                y[(size_t) row * kInner + g * kGroups + i] = v;
                ss += v * v;
            }
            const float scale = 1.f / sqrtf(ss / kGroups + kEps);
            for (int i = 0; i < kGroups; ++i)
                y[(size_t) row * kInner + g * kGroups + i] *= scale * norm[g * kGroups + i];
        }
    }
}

template <typename T> T* upload(const std::vector<T>& v) {
    T* d = nullptr;
    if (cudaMalloc(&d, v.size() * sizeof(T)) != cudaSuccess) std::abort();
    cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice);
    return d;
}

std::vector<float> download(const float* d, size_t n) {
    std::vector<float> v(n);
    cudaMemcpy(v.data(), d, n * sizeof(float), cudaMemcpyDeviceToHost);
    return v;
}

float worst(const std::vector<float>& a, const std::vector<float>& b) {
    float w = 0.f;
    for (size_t i = 0; i < a.size(); ++i) w = std::fmax(w, std::fabs(a[i] - b[i]) / (1.f + std::fabs(b[i])));
    return w;
}

// Everything a check needs: the random inputs, their host reference and the device buffers.
struct Case {
    std::vector<float> proj, w, bias, dt_bias, a, d, norm, conv_state0, scan_state0;
    float *d_proj = nullptr, *d_w = nullptr, *d_bias = nullptr, *d_dt_bias = nullptr, *d_a = nullptr,
          *d_d = nullptr, *d_norm = nullptr, *d_dt = nullptr, *d_conv_state = nullptr, *d_scan_state = nullptr,
          *d_conv_out = nullptr, *d_y = nullptr, *d_snap_conv = nullptr, *d_snap_scan = nullptr;

    Case(int rows, std::mt19937& rng) {
        std::normal_distribution<float> nd(0.f, 1.f);
        std::uniform_real_distribution<float> au(-2.f, -0.01f);
        auto fill = [&](std::vector<float>& v, size_t n, auto gen) {
            v.resize(n);
            for (float& x : v) x = gen();
        };
        fill(proj, (size_t) rows * kProj, [&] { return nd(rng); });
        fill(w, (size_t) kConv * 4, [&] { return 0.5f * nd(rng); });
        fill(bias, kConv, [&] { return nd(rng); });
        fill(dt_bias, kHead, [&] { return nd(rng); });
        fill(a, kHead, [&] { return au(rng); });
        fill(d, kHead, [&] { return nd(rng); });
        fill(norm, kInner, [&] { return 1.f + 0.1f * nd(rng); });
        fill(conv_state0, (size_t) kConv * 3, [&] { return nd(rng); });
        fill(scan_state0, (size_t) kInner * kState, [&] { return nd(rng); });
        d_proj = upload(proj), d_w = upload(w), d_bias = upload(bias), d_dt_bias = upload(dt_bias);
        d_a = upload(a), d_d = upload(d), d_norm = upload(norm);
        cudaMalloc(&d_dt, (size_t) rows * kHead * sizeof(float));
        cudaMalloc(&d_conv_out, (size_t) rows * kConv * sizeof(float));
        cudaMalloc(&d_y, (size_t) rows * kInner * sizeof(float));
        cudaMalloc(&d_conv_state, conv_state0.size() * sizeof(float));
        cudaMalloc(&d_scan_state, scan_state0.size() * sizeof(float));
        cudaMalloc(&d_snap_conv, (size_t) 9 * 3 * kConv * sizeof(float));
        cudaMalloc(&d_snap_scan, (size_t) 9 * kInner * kState * sizeof(float));
    }
    ~Case() {
        for (void* p : {(void*) d_proj, (void*) d_w, (void*) d_bias, (void*) d_dt_bias, (void*) d_a, (void*) d_d,
                        (void*) d_norm, (void*) d_dt, (void*) d_conv_state, (void*) d_scan_state,
                        (void*) d_conv_out, (void*) d_y, (void*) d_snap_conv, (void*) d_snap_scan})
            if (p) cudaFree(p);
    }
    Case(const Case&) = delete;
    Case& operator=(const Case&) = delete;

    void reset() {
        cudaMemcpy(d_conv_state, conv_state0.data(), conv_state0.size() * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_scan_state, scan_state0.data(), scan_state0.size() * sizeof(float), cudaMemcpyHostToDevice);
    }
    // The block's kernels for `rows` tokens from row `first` of the inputs.  d_y holds the post-scan y; call
    // normed() to finish the block.
    void run(int first, int rows, unsigned snaps) {
        const int dt_off = 2 * kInner + 2 * kGroup * kState;
        const float* p = d_proj + (size_t) first * kProj;
        cudaMemcpy2D(d_dt, (size_t) kHead * sizeof(float), p + dt_off, (size_t) kProj * sizeof(float),
                     (size_t) kHead * sizeof(float), (size_t) rows, cudaMemcpyDeviceToDevice);
        add_bias(d_dt_bias, d_dt, kHead, rows, nullptr);
        mamba_conv(p + kInner, kProj, d_conv_state, d_w, d_bias, d_conv_out, kConv, rows, nullptr,
                   snaps ? d_snap_conv : nullptr, 3 * (size_t) kConv, snaps);
        mamba_scan(d_conv_out, d_conv_out + kInner, d_conv_out + kInner + kGroup * kState, d_dt, d_a, d_d, d_scan_state,
                   d_y, kDim, kHead, kState, kGroup, rows, nullptr, snaps ? d_snap_scan : nullptr,
                   (size_t) kInner * kState, snaps);
    }
    void normed(int first, int rows) {
        mamba_norm(d_y, d_proj + (size_t) first * kProj, kProj, d_norm, kGroups, kGroup, rows, kEps, nullptr);
    }
};

// (a) the three kernels against the host reference, for one call of `rows` tokens.
int check_formulas(int rows, std::mt19937& rng) {
    Case c(rows, rng);
    std::vector<float> h_conv_state = c.conv_state0, h_scan_state = c.scan_state0;
    std::vector<float> h_conv, h_yd, h_y;
    host_block(c.proj, c.w, c.bias, c.dt_bias, c.a, c.d, c.norm, h_conv_state, h_scan_state, rows, h_conv, h_yd, h_y,
               nullptr, nullptr, 0u);
    c.reset();
    c.run(0, rows, 0u);
    const float e_yd = worst(download(c.d_y, h_yd.size()), h_yd);
    c.normed(0, rows);
    const float e_conv = worst(download(c.d_conv_out, h_conv.size()), h_conv);
    const float e_y = worst(download(c.d_y, h_y.size()), h_y);
    const float e_cs = worst(download(c.d_conv_state, h_conv_state.size()), h_conv_state);
    const float e_ss = worst(download(c.d_scan_state, h_scan_state.size()), h_scan_state);
    // The block's own output and the convolution are f32 over few terms, so 1e-6; the raw scan y is a sum of
    // d_state (128) f32 products and the differently-compiled scalar reference cannot stay under 1e-6 of it
    // (~5e-6, the f32 accumulation itself); the scan state is within ~1e-6.  The row-splitting checks below are
    // exact, which is what pins the recurrence.
    const bool ok = e_conv < 1e-6f && e_y < 1e-6f && e_cs == 0.f && e_ss < 2e-6f && e_yd < 1e-5f;
    std::printf("%s rows %2d: conv %.2g, y_add_d %.2g, y_norm %.2g, conv state %.2g, scan state %.2g\n",
                ok ? "ok  " : "FAIL", rows, e_conv, e_yd, e_y, e_cs, e_ss);
    return ok ? 0 : 1;
}

// (b) rows=3 against three rows=1 calls, with the state restored from the snapshots the first call left.
int check_rows3(std::mt19937& rng) {
    Case c(3, rng);
    std::vector<float> want_conv((size_t) 3 * kConv), want_yd((size_t) 3 * kInner), want_y((size_t) 3 * kInner);
    std::vector<float> snap_conv((size_t) 3 * 3 * kConv), snap_scan((size_t) 3 * kInner * kState);
    std::vector<float> h_cs = c.conv_state0, h_ss = c.scan_state0;
    host_block(c.proj, c.w, c.bias, c.dt_bias, c.a, c.d, c.norm, h_cs, h_ss, 3, want_conv, want_yd, want_y,
               &snap_conv, &snap_scan, 0x7u);
    c.reset();
    c.run(0, 3, 0x7u);
    const std::vector<float> got_conv = download(c.d_conv_out, want_conv.size());
    const std::vector<float> got_yd = download(c.d_y, want_yd.size());
    float e_conv = worst(got_conv, want_conv), e_yd = worst(got_yd, want_yd);

    // now three one-token calls, the state restored from the snapshots between them
    c.reset();
    c.run(0, 1, 0u);
    float e_seq_conv = 0.f, e_seq_yd = 0.f;
    // the state after row 0 and 1 is in the snapshots; the first call starts from the initial state
    for (int k = 0; k < 3; ++k) {
        if (k > 0) {
            cudaMemcpy(c.d_conv_state, c.d_snap_conv + (size_t) (k - 1) * 3 * kConv, (size_t) 3 * kConv * sizeof(float),
                       cudaMemcpyDeviceToDevice);
            cudaMemcpy(c.d_scan_state, c.d_snap_scan + (size_t) (k - 1) * kInner * kState,
                       (size_t) kInner * kState * sizeof(float), cudaMemcpyDeviceToDevice);
            c.run(k, 1, 0u);
        }
        const auto co = download(c.d_conv_out, kConv), yd = download(c.d_y, kInner);
        for (int i = 0; i < kConv; ++i)
            e_seq_conv = std::fmax(e_seq_conv, std::fabs(co[i] - got_conv[(size_t) k * kConv + i]) / (1.f + std::fabs(got_conv[(size_t) k * kConv + i])));
        for (int i = 0; i < kInner; ++i)
            e_seq_yd = std::fmax(e_seq_yd, std::fabs(yd[i] - got_yd[(size_t) k * kInner + i]) / (1.f + std::fabs(got_yd[(size_t) k * kInner + i])));
    }
    // The property is the sequential equality (exact); the host comparison is a sanity line, already covered by
    // check_formulas(3).
    const bool ok = e_seq_conv == 0.f && e_seq_yd == 0.f;
    std::printf("%s rows 3 vs 1+1+1: host conv %.2g y %.2g, sequential conv %.2g y %.2g\n", ok ? "ok  " : "FAIL",
                e_conv, e_yd, e_seq_conv, e_seq_yd);
    return ok ? 0 : 1;
}

// (c) a window's replay: nine tokens in one call, then the state restored from the snapshot after the fifth
// token and the rest run again, which must reproduce the tokens after it and the final state.  Also the nine
// tokens in one call against nine rows=1 calls carrying the state.
int check_window(std::mt19937& rng) {
    Case c(9, rng);
    c.reset();
    c.run(0, 9, 0x1ffu);   // a snapshot after every token
    const std::vector<float> want_yd = download(c.d_y, (size_t) 9 * kInner);
    const std::vector<float> want_ss = download(c.d_scan_state, (size_t) kInner * kState);
    const std::vector<float> want_cs = download(c.d_conv_state, (size_t) 3 * kConv);

    // restore the state after token 4 and run the four tokens behind it
    cudaMemcpy(c.d_conv_state, c.d_snap_conv + (size_t) 4 * 3 * kConv, (size_t) 3 * kConv * sizeof(float),
               cudaMemcpyDeviceToDevice);
    cudaMemcpy(c.d_scan_state, c.d_snap_scan + (size_t) 4 * kInner * kState, (size_t) kInner * kState * sizeof(float),
               cudaMemcpyDeviceToDevice);
    c.run(5, 4, 0u);
    const std::vector<float> replay = download(c.d_y, (size_t) 4 * kInner);
    float e = 0.f;
    for (int k = 0; k < 4; ++k)
        for (int i = 0; i < kInner; ++i)
            e = std::fmax(e, std::fabs(replay[(size_t) k * kInner + i] - want_yd[(size_t) (k + 5) * kInner + i]) /
                                 (1.f + std::fabs(want_yd[(size_t) (k + 5) * kInner + i])));
    e = std::fmax(e, worst(download(c.d_scan_state, want_ss.size()), want_ss));
    e = std::fmax(e, worst(download(c.d_conv_state, want_cs.size()), want_cs));
    const bool ok_restore = e == 0.f;

    c.reset();
    std::vector<float> got((size_t) 9 * kInner);
    for (int k = 0; k < 9; ++k) {
        c.run(k, 1, 0u);
        const auto yd = download(c.d_y, kInner);
        std::copy(yd.begin(), yd.end(), got.begin() + (size_t) k * kInner);
    }
    const float e_seq = std::fmax(worst(got, want_yd),
                                  std::fmax(worst(download(c.d_scan_state, want_ss.size()), want_ss),
                                            worst(download(c.d_conv_state, want_cs.size()), want_cs)));
    const bool ok = ok_restore && e_seq == 0.f;
    std::printf("%s window replay: restore %.2g, 9 tokens vs 9 x 1 %.2g\n", ok ? "ok  " : "FAIL", e, e_seq);
    return ok ? 0 : 1;
}

}  // namespace

int main() {
    std::mt19937 rng(11);
    int bad = 0;
    for (int rows : {1, 3, 8}) bad += check_formulas(rows, rng);
    bad += check_rows3(rng);
    bad += check_window(rng);
    if (const cudaError_t e = cudaGetLastError(); e != cudaSuccess) {
        std::printf("CUDA: %s\n", cudaGetErrorString(e));
        return 1;
    }
    std::printf(bad ? "%d failed\n" : "all passed\n", bad);
    return bad ? 1 : 0;
}
