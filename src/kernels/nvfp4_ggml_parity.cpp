// src/kernels/nvfp4_ggml_parity.cpp - the NVFP4 (GGML type 40) matvec against ggml's own dequantization.
//
//   nvfp4_ggml_parity
//
// The engine multiplies a dense NVFP4 weight with iq_mmvq (native_mmvq.cu hands type 40 to it) over the GGUF
// blocks, on a q8_1 activation.  This compares that product, and the same product times the file's per-tensor
// scale, with ggml_get_type_traits(GGML_TYPE_NVFP4)->to_float() of the very same blocks followed by a
// double-precision dot, for n_out 1 and 8, n_in 64 and 18560 (64, one NVFP4 block's width, and 18560, the
// largest axis of the dense NVFP4 tensors the Nemotron file keeps: ssm_in [4096, 18560] and ssm_out [8192, 4096]),
// and ncols 1 and 8 (the engine's lin() launches up to 8 tokens at once, iq_kernels.cu's mmvq_multi_kernel).
//
// The reference dot reads the *quantized* activation, dequantized from the q8_1 the kernel itself was given, so
// what is left between the two is the kernel's arithmetic and not q8_1's own loss.
//
// The reference sets an NVFP4 mul_mat's accumulator to BF16 (ref/src/llama-graph.cpp:1626-1628:
// `if (w->type == GGML_TYPE_NVFP4) ggml_prec_set_acc(res, GGML_PREC_BF16);`) while the poly MMVQ accumulates
// f32.  A BF16 *kernel* is not written here (that is a later milestone); instead every row's result is also
// rounded to BF16 and its error printed, so the size of the gap between the two modes is a measured number.
//
// Needs a card, so it is run by hand, like poly_route_test and poly_norm_q8_test.
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/poly/kernels.hpp"

#include "ggml.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <vector>

namespace {

void ck(cudaError_t e) {
    if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}

// GGML's q8_1 block: fp16 d and s, then 32 int8.  native_quantize_q8_1() writes it; the kernel reads d for the
// product and s for an MoE sum we do not use here.
struct Q8_1 {
    uint16_t d, s;
    int8_t qs[32];
};
static_assert(sizeof(Q8_1) == 36, "a q8_1 block is 36 bytes per 32 values");

// A float rounded to BF16 and back, as a BF16 accumulator would hold it.
float to_bf16(float v) {
    uint32_t u;
    std::memcpy(&u, &v, sizeof u);
    u += 0x7FFFu + ((u >> 16) & 1u);   // round to nearest, ties to even
    u &= 0xFFFF0000u;
    std::memcpy(&v, &u, sizeof v);
    return v;
}

double rel_error(const float* got, const std::vector<double>& want) {
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < want.size(); ++i) {
        num += ((double) got[i] - want[i]) * ((double) got[i] - want[i]);
        den += want[i] * want[i];
    }
    return std::sqrt(num / den);
}

int g_ok = 0;

// One shape and one column count: random block_nvfp4 weights and a random activation of ncols columns.
void check(int n_in, int n_out, int ncols) {
    const int block = 64, block_bytes = 36;
    const size_t refs = (size_t) n_out * n_in;
    const size_t wbytes = (size_t) n_out * (n_in / block) * block_bytes;
    const size_t cols = (size_t) ncols;

    std::mt19937 rng(20261015u + (unsigned) n_in * 8u + (unsigned) n_out);
    std::normal_distribution<float> dist(0.f, 1.f);

    // Random floats, then ggml's own NVFP4 quantizer makes the blocks: the layout, the UE4M3 scales and the
    // E2M1 codes are ggml's, so to_float() and iq_mmvq() must read them the same way.  Columns are contiguous, as
    // native_mmvq.hpp defines them: activation column c is x + c * n_in, output column c is y + c * n_out.
    std::vector<float> wf(refs), x(cols * (size_t) n_in);
    for (float& v : wf) v = dist(rng);
    for (float& v : x) v = dist(rng);
    const ggml_type_traits* traits = ggml_get_type_traits(GGML_TYPE_NVFP4);
    if (!traits || !traits->to_float || !traits->from_float_ref) throw std::runtime_error("ggml has no NVFP4 traits");
    std::vector<uint8_t> wblocks(wbytes);
    traits->from_float_ref(wf.data(), (void*) wblocks.data(), (int64_t) refs);
    std::vector<float> wdeq(refs);
    traits->to_float(wblocks.data(), wdeq.data(), (int64_t) refs);

    // The activation on the card, as the engine makes it, and the same values back on the host for the reference.
    cudaStream_t st = nullptr;
    ck(cudaStreamCreate(&st));
    float* d_x = nullptr;
    void* d_xq = nullptr;
    uint8_t* d_w = nullptr;
    float* d_y = nullptr;
    const size_t xq_bytes = strata::kernels::native_q8_1_bytes(n_in, ncols);
    ck(cudaMalloc(&d_x, x.size() * 4));
    ck(cudaMalloc(&d_xq, xq_bytes));
    ck(cudaMalloc((void**) &d_w, wbytes));
    ck(cudaMalloc(&d_y, cols * (size_t) n_out * 4));
    ck(cudaMemcpy(d_x, x.data(), x.size() * 4, cudaMemcpyHostToDevice));
    ck(cudaMemcpy(d_w, wblocks.data(), wbytes, cudaMemcpyHostToDevice));
    strata::kernels::native_quantize_q8_1(d_x, d_xq, n_in, ncols, st);
    std::vector<uint8_t> q8(xq_bytes);
    ck(cudaMemcpyAsync(q8.data(), d_xq, xq_bytes, cudaMemcpyDeviceToHost, st));
    ck(cudaStreamSynchronize(st));
    // The columns' q8_1 blocks are back to back with no row padding, so decode in linear order: element i of
    // column c is block (c * n_in + i) / 32, position (c * n_in + i) % 32.
    std::vector<double> xq(cols * (size_t) n_in);
    for (size_t b = 0; b < xq.size() / 32; ++b) {
        const Q8_1* blk = reinterpret_cast<const Q8_1*>(q8.data() + b * sizeof(Q8_1));
        const double d = (double) ggml_fp16_to_fp32(blk->d);
        for (int j = 0; j < 32; ++j) xq[b * 32 + j] = d * (double) blk->qs[j];
    }

    // The reference: a double dot of the dequantized weights and the dequantized activation, per column and row.
    std::vector<double> want(cols * (size_t) n_out);
    for (int c = 0; c < ncols; ++c) {
        for (int r = 0; r < n_out; ++r) {
            double acc = 0.0;
            for (int i = 0; i < n_in; ++i)
                acc += (double) wdeq[(size_t) r * n_in + i] * xq[(size_t) c * n_in + i];
            want[(size_t) c * n_out + r] = acc;
        }
    }

    // The card, and the same call through native_mmvq()'s new type-40 dispatch: the two must be bitwise equal.
    const size_t yn = cols * (size_t) n_out;
    std::vector<float> got(yn), native_on_card(yn);
    strata::kernels::iq_mmvq(40, d_w, d_xq, d_y, n_in, n_out, ncols, st);
    ck(cudaMemcpyAsync(got.data(), d_y, yn * 4, cudaMemcpyDeviceToHost, st));
    ck(cudaStreamSynchronize(st));
    strata::kernels::native_mmvq(40, d_w, d_xq, d_y, n_in, n_out, ncols, st);
    ck(cudaMemcpyAsync(native_on_card.data(), d_y, yn * 4, cudaMemcpyDeviceToHost, st));
    ck(cudaStreamSynchronize(st));
    const bool dispatch_same = std::memcmp(got.data(), native_on_card.data(), yn * 4) == 0;

    const double e_f32 = rel_error(got.data(), want);

    // The file's per-tensor multiplier, applied on the card exactly as lin() applies it (axpy, add=false).
    const float scale = 1.337e-4f;   // ssm_in.scale of the Nemotron file, a representative per-tensor scale
    strata::poly::axpy(d_y, scale, d_y, (int) yn, false, st);
    std::vector<float> scaled(yn);
    ck(cudaMemcpyAsync(scaled.data(), d_y, yn * 4, cudaMemcpyDeviceToHost, st));
    ck(cudaStreamSynchronize(st));
    std::vector<double> want_scaled(want.size());
    for (size_t i = 0; i < want.size(); ++i) want_scaled[i] = want[i] * (double) scale;
    const double e_scaled = rel_error(scaled.data(), want_scaled);

    // The same result held in BF16, as the reference's accumulator asks for.
    std::vector<float> bf16(yn);
    for (size_t i = 0; i < yn; ++i) bf16[i] = to_bf16(got[i]);
    const double e_bf16 = rel_error(bf16.data(), want);

    const bool shape_ok = strata::kernels::native_mmvq_weight_bytes(40, n_in, n_out) == wbytes &&
                          strata::kernels::native_mmvq_ids_supported(40) && dispatch_same &&
                          e_f32 < 1e-3 && std::fabs(e_scaled - e_f32) < 1e-6;
    g_ok += shape_ok ? 1 : 0;
    std::printf("n_in %5d  n_out %2d  ncols %d  f32 acc %.3e  bf16 acc %.3e  gap %.3e  scaled %.3e  %s\n",
                n_in, n_out, ncols, e_f32, e_bf16, e_bf16 - e_f32, e_scaled, shape_ok ? "ok" : "FAIL");

    ck(cudaFree(d_y));
    ck(cudaFree(d_xq));
    ck(cudaFree(d_w));
    ck(cudaFree(d_x));
    ck(cudaStreamDestroy(st));
}

}  // namespace

int main() {
    try {
        std::printf("NVFP4 (type 40) vs ggml_get_type_traits(NVFP4)->to_float + a double dot\n");
        g_ok = 0;
        for (int ncols : {1, 8}) {
            check(64, 1, ncols);
            check(64, 8, ncols);
            check(18560, 1, ncols);
            check(18560, 8, ncols);
        }
        std::printf("%d of 8 shapes ok\n", g_ok);
        return g_ok == 8 ? 0 : 1;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "nvfp4_ggml_parity: %s\n", e.what());
        return 1;
    }
}
