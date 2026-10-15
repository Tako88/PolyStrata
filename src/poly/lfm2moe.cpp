// src/poly/lfm2moe.cpp - the layers of LFM2.5 with routed experts.  The reference for the math is llama.cpp's
// lfm2.cpp / lfm2moe.cpp graph.
#include "lfm2moe.hpp"

#include "runtime.hpp"
#include "strata/poly/kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <stdexcept>

namespace strata::poly {
namespace {

constexpr float kMinWeightSum = 6.103515625e-5f;   // the floor ggml's MoE graph puts under the gate sum
constexpr size_t kScoreFloats = size_t(16) << 20;  // the attention's scores of the tokens it takes at a time

std::mutex g_route_mutex;   // STRATA_DUMP_ROUTE: the host's routing threads share one line at a time

// A value a file gives once for the model or once a layer.
int64_t of_layer(const MetaValue& v, int il) {
    if (v.type != MetaType::ARRAY) return (int64_t) v.u;
    if ((size_t) il >= v.items.size())
        throw std::runtime_error("a per-layer value of the GGUF ends before its layers");
    return (int64_t) v.items[(size_t) il].u;
}

}  // namespace

// llama.cpp reads the output norm of this architecture from `token_embd_norm.weight` (its
// LLM_TENSOR_OUTPUT_NORM_LFM2, src/llama-arch.cpp:427).  The engine looks for the bare name first, so every other
// architecture's lookup and its error message are unchanged.
const Names* Lfm2Moe::out_norm_alias() const {
    static const Names kOutNorm{"output_norm.weight", {"token_embd_norm.weight", nullptr}};
    return &kOutNorm;
}

void Lfm2Moe::begin_pass() {
    blocks_.assign(layers_.size(), Block{});
    n_conv_ = n_attn_ = 0;
    for (int il = 0; il < (int) blocks_.size(); ++il) {
        const bool conv = il < (int) kv_.size() && kv_[(size_t) il] == 0;
        blocks_[(size_t) il].conv = conv;
        (conv ? n_conv_ : n_attn_)++;
    }
}

void Lfm2Moe::load_geometry() {
    load_shape();
    auto find = [&](const char* key) { return meta_in(*gg_, key); };
    auto need = [](bool ok, const char* what) {
        if (!ok) throw std::runtime_error(std::string("unsupported model: ") + what);
    };
    auto must = [&](const char* key) -> const MetaValue& {
        const MetaValue* v = find(key);
        need(v != nullptr, key);
        return *v;
    };

    n_ff_ = shape_i("feed_forward_length");
    const MetaValue* lead = find("leading_dense_block_count");
    leading_dense_ = lead ? (int) lead->u : 0;   // llama.cpp's get_key(..., false): absent means no dense layer
    // taps >= 2 is llama.cpp's own GGML_ASSERT(n_shortconv_l_cache > 1).  There is no upper bound: the tensor's
    // own shape is the guard, upload() throws with the tensor's name and shape when the file disagrees.
    taps_ = shape_i("shortconv.l_cache");
    need(taps_ >= 2, "shortconv.l_cache is below 2");
    n_head_ = shape_i("attention.head_count");
    need(n_head_ > 0 && n_head_ <= 256 && n_embd_ % n_head_ == 0, "bad attention.head_count");

    // A layer's attention.head_count_kv of 0 makes it a short-convolution layer; the others have keys of their
    // own.  The walk, and the copy of the scalar-or-array helper, follow gemma4.cpp:20-24 and :61-72.
    const MetaValue& kv = must("attention.head_count_kv");
    kv_.resize((size_t) n_layer_);
    n_conv_ = n_attn_ = 0;
    n_kv_ = 0;
    for (int il = 0; il < n_layer_; ++il) {
        const int k = (int) of_layer(kv, il);
        kv_[(size_t) il] = k;
        if (k == 0) {
            ++n_conv_;
            continue;
        }
        need(k > 0 && n_head_ % k == 0, "bad attention.head_count_kv");
        if (n_kv_ == 0) n_kv_ = k;
        // one cache row width is sized for every attention layer (Block), so a file that varies them is refused
        // rather than sized for one of them
        need(k == n_kv_, "attention.head_count_kv differs between layers");
        ++n_attn_;
    }
    need(n_conv_ + n_attn_ == n_layer_, "a layer is neither a short convolution nor an attention");
    need(leading_dense_ >= 0 && leading_dense_ <= n_layer_, "bad leading_dense_block_count");

    const MetaValue* kl = find("attention.key_length");
    head_dim_ = kl ? (int) kl->u : n_embd_ / n_head_;
    const MetaValue* rot = find("rope.dimension_count");
    n_rot_ = rot ? (int) rot->u : head_dim_;
    const MetaValue* base = find("rope.freq_base");
    theta_scale_ = std::pow(base ? (float) base->f : 10000.f, -2.f / (float) n_rot_);
    need(head_dim_ > 0 && head_dim_ <= 256, "bad attention.key_length");
    need(n_rot_ > 0 && n_rot_ % 2 == 0 && n_rot_ <= head_dim_, "bad rope.dimension_count");
    // the sliding-window variant keeps a ring instead of a cache a position; no file here has one.  The value is
    // read a layer at a time so a file that gives it per layer is refused rather than read as a scalar.
    const MetaValue* swa = find("attention.sliding_window");
    if (swa)
        for (int il = 0; il < n_layer_; ++il) need(of_layer(*swa, il) == 0, "a sliding-window LFM2");
    // the whole routing design is the sigmoid path.  expert_gating_func is one value for the model, not a layer
    // at a time (unlike attention.head_count_kv), so the scalar read is the whole value.
    const MetaValue* gating = find("expert_gating_func");
    need(gating && gating->u == 2, "expert_gating_func is not 2 (sigmoid)");

    // this architecture puts no limit on a SwiGLU's inputs, and has no shared expert to size a limit for
    clamp_exp_.assign((size_t) n_layer_ + 1, INFINITY);
    clamp_shexp_ = clamp_exp_;
    bias_.assign((size_t) n_layer_ * n_expert_, 0.f);

    // Opt-in: print the router's ids, weights and biased scores, for Step 17's comparison with llama.cpp.
    const char* dump = std::getenv("STRATA_DUMP_ROUTE");
    dump_route_ = dump && *dump && std::strcmp(dump, "0") != 0;
}

void Lfm2Moe::load_layer(int il) {
    Layer& L = layers_[(size_t) il];
    Block& B = blocks_[(size_t) il];
    auto f32 = [&](const char* n, int64_t a, int64_t b = 1) { return upload(tensor(il, n), blk(il, n), a, b, kF32); };
    auto mat = [&](const char* n, int64_t a, int64_t b) { return upload(tensor(il, n), blk(il, n), a, b, kAnyMat); };

    L.attn_norm = f32("attn_norm.weight", n_embd_);
    L.ffn_norm = f32("ffn_norm.weight", n_embd_);
    L.fixed = true;   // the attention's launches read the position on the card when they are recorded

    if (B.conv) {
        B.in_proj = mat("shortconv.in_proj.weight", n_embd_, 3 * n_embd_);
        B.conv_w = f32("shortconv.conv.weight", taps_, n_embd_);   // upload() converts any type to F32
        B.out_proj = mat("shortconv.out_proj.weight", n_embd_, n_embd_);
    } else {
        B.wq = mat("attn_q.weight", n_embd_, (int64_t) n_head_ * head_dim_);
        B.wk = mat("attn_k.weight", n_embd_, (int64_t) n_kv_ * head_dim_);
        B.wv = mat("attn_v.weight", n_embd_, (int64_t) n_kv_ * head_dim_);
        B.wo = mat("attn_output.weight", (int64_t) n_head_ * head_dim_, n_embd_);
        B.q_norm = f32("attn_q_norm.weight", head_dim_);
        B.k_norm = f32("attn_k_norm.weight", head_dim_);
    }
    if (il < leading_dense_) {
        B.ffn_gate = mat("ffn_gate.weight", n_embd_, n_ff_);
        B.ffn_up = mat("ffn_up.weight", n_embd_, n_ff_);
        B.ffn_down = mat("ffn_down.weight", n_ff_, n_embd_);
        return;
    }
    load_moe(il, /*shared=*/false);   // no shared expert: moe_shared() only zeroes the sum's buffer
    const TensorInfo* s = tensor(il, "exp_probs_b.bias");
    if (!s || s->type != kTypeF32 || (int64_t) s->elements() != n_expert_ ||
        home(*s).payload(*s) != (size_t) n_expert_ * sizeof(float))
        throw std::runtime_error((s ? s->name : blk(il, "exp_probs_b.bias")) + " is " +
                                 (s ? shape_of(*s) : std::string("missing")) + ", expected F32 [" +
                                 std::to_string(n_expert_) + "]");
    std::memcpy(bias_.data() + (size_t) il * n_expert_, home(*s).data(*s), (size_t) n_expert_ * sizeof(float));
}

size_t Lfm2Moe::recurrent_floats() const { return size_t(n_conv_) * size_t(taps_ - 1) * n_embd_; }

int Lfm2Moe::widest() const { return std::max(n_embd_, n_ff_); }

// The attention layers' caches, and each short-convolution layer's state in state_.  The state lies in layer
// order, so save_layer() finds it at (B.past - state_).
void Lfm2Moe::alloc_caches() {
    const size_t past_n = size_t(taps_ - 1) * n_embd_;
    const size_t row = (size_t) n_kv_ * head_dim_ * sizeof(uint16_t);
    float* st = state_;
    for (int il = 0; il < (int) blocks_.size(); ++il) {
        Block& B = blocks_[(size_t) il];
        if (!runs(il)) continue;
        if (B.conv) {
            B.past = st;
            st += past_n;
        } else {
            B.keys = (uint16_t*) kv_alloc(row, 1);
            B.vals = (uint16_t*) kv_alloc(row, 1);
        }
    }
    chunk_ = (int) std::clamp<size_t>(kScoreFloats / ((size_t) n_head_ * opt_.max_context), 1, (size_t) rows_);
}

// Per-token buffers hold rows_ tokens, row after row.
void Lfm2Moe::assign_scratch() {
    const size_t n = (size_t) rows_, heads = n * n_head_ * head_dim_, kv = n * n_kv_ * head_dim_;
    proj_ = fbuf(n * 3 * n_embd_);
    conv_ = fbuf(n * n_embd_);
    qraw_ = fbuf(heads);
    q_ = fbuf(heads);
    kraw_ = fbuf(kv);
    k_ = fbuf(kv);
    v_ = fbuf(kv);
    att_ = fbuf(heads);
    score_ = fbuf((size_t) chunk_ * n_head_ * opt_.max_context);
    if (rows_ >= kStageRows) awork_ = fbuf(heads + heads / 2 + 1 + 3 * n * n_head_);   // a prompt batch's attention
}

// A short-convolution layer's state lies one behind the other in state_ (alloc_caches()).
void Lfm2Moe::save_layer(int il, int64_t, uint8_t* dst) {
    const Block& B = blocks_[(size_t) il];
    if (!B.past) return;
    const size_t n = size_t(taps_ - 1) * n_embd_;
    ck(cudaMemcpy(dst + (size_t) (B.past - state_) * sizeof(float), B.past, n * sizeof(float),
                  cudaMemcpyDeviceToHost),
       "save state");
}

size_t Lfm2Moe::card_row_bytes() const { return (size_t) n_attn_ * 2 * n_kv_ * head_dim_ * sizeof(uint16_t); }

void Lfm2Moe::save_card_rows(int64_t first, int64_t n, uint8_t* dst) {
    if (first < 0 || n < 0 || first + n > opt_.max_context)
        throw std::logic_error("save_rows: the rows are outside the context");
    ck(cudaStreamSynchronize(cs(st_)), "save rows");
    const size_t row = (size_t) n_kv_ * head_dim_ * sizeof(uint16_t), at = (size_t) first * row, bytes = (size_t) n * row;
    for (const Block& B : blocks_) {
        if (!B.keys || bytes == 0) continue;
        for (const uint16_t* cache : {B.keys, B.vals}) {
            ck(cudaMemcpy(dst, (const uint8_t*) cache + at, bytes, cudaMemcpyDeviceToHost), "save rows");
            dst += bytes;
        }
    }
}

void Lfm2Moe::load_card_rows(const uint8_t* src, int64_t from, int64_t count, int64_t first, int64_t n) {
    if (from < 0 || first < from || n < 0 || first + n > from + count || first + n > opt_.max_context)
        throw std::logic_error("load_rows: the rows are outside what was saved");
    cover(first + n);
    ck(cudaStreamSynchronize(cs(st_)), "load rows");
    const size_t row = (size_t) n_kv_ * head_dim_ * sizeof(uint16_t), at = (size_t) first * row, bytes = (size_t) n * row;
    const size_t skip = (size_t) (first - from) * row, all = (size_t) count * row;
    for (const Block& B : blocks_) {
        if (!B.keys || bytes == 0) continue;
        for (uint16_t* cache : {B.keys, B.vals}) {
            ck(cudaMemcpy((uint8_t*) cache + at, src + skip, bytes, cudaMemcpyHostToDevice), "load rows");
            src += all;
        }
    }
}

// A layer from its input to its router's copy (a layer without routed experts: to its end).  A prompt batch's
// MoE (moe_rows) does its own routing.
void Lfm2Moe::layer_head(Layer& L, int il, int n) {
    Block& B = blocks_[(size_t) il];
    rms_norm(r_, fp(L.attn_norm), xn_, n_embd_, n, rms_eps_, st_);
    kernels::native_quantize_q8_1(xn_, xq_, n_embd_ * n, 1, st_);
    dbg("attn_in", il, xn_, n_embd_);
    if (B.conv) {
        // b*x through the taps, then c, then out_proj
        lin(B.in_proj, xn_, proj_, n, xq_);
        dbg("conv_in", il, proj_, 3 * n_embd_);
        shortconv(proj_, B.past, fp(B.conv_w), conv_, n_embd_, taps_, n, st_,
                  snap_bits_ ? snap_at_ + (B.past - state_) : nullptr, rec_n_, snap_bits_);
        dbg("conv", il, conv_, n_embd_);
        lin(B.out_proj, conv_, ao_, n);
        dbg("conv_out", il, ao_, n_embd_);
    } else {
        const long long* at = recording_ ? at_ : nullptr;   // a recorded launch: the tokens from what at_ holds
        lin(B.wq, xn_, qraw_, n, xq_);
        head_norm_rope(qraw_, head_dim_, fp(B.q_norm), q_, head_dim_, n_head_, n_rot_, theta_scale_, pos_, n, rms_eps_,
                       st_, nullptr, nullptr, 1, at);
        lin(B.wk, xn_, kraw_, n, xq_);
        head_norm_rope(kraw_, head_dim_, fp(B.k_norm), k_, head_dim_, n_kv_, n_rot_, theta_scale_, pos_, n, rms_eps_,
                       st_, nullptr, nullptr, 1, at);
        lin(B.wv, xn_, v_, n, xq_);
        if (at) {
            store_f16_at(k_, B.keys, n_kv_ * head_dim_, at, n, 0, st_);
            store_f16_at(v_, B.vals, n_kv_ * head_dim_, at, n, 0, st_);
        } else {
            store_f16(k_, B.keys, n_kv_ * head_dim_, pos_, n, st_);
            store_f16(v_, B.vals, n_kv_ * head_dim_, pos_, n, st_);
        }
        const float scale = 1.f / std::sqrt((float) head_dim_);
        // a prompt batch through cuBLAS; a few tokens with the kernels that give a token the same bits in any
        // company
        if (n >= kStageRows)
            attend_batch(blas_, q_, B.keys, B.vals, pos_, n, score_, (size_t) chunk_ * n_head_ * opt_.max_context,
                         awork_, att_, head_dim_, n_head_, n_kv_, scale, st_);
        else
            attend(q_, B.keys, B.vals, pos_, n, score_, chunk_, att_, head_dim_, n_head_, n_kv_, scale, st_, 0, 0,
                   false, at, at ? room(n) : 0);
        lin(B.wo, att_, ao_, n);
        dbg("attn", il, ao_, n_embd_);
    }
    axpy(ao_, 1.f, r_, n * n_embd_, true, st_);

    rms_norm(r_, fp(L.ffn_norm), xn_, n_embd_, n, rms_eps_, st_);
    kernels::native_quantize_q8_1(xn_, xq_, n_embd_ * n, 1, st_);
    dbg("ffn_in", il, xn_, n_embd_);
    if (!L.moe) {
        lin(B.ffn_gate, xn_, g_, n, xq_);
        lin(B.ffn_up, xn_, u_, n, xq_);
        swiglu_clamp(g_, u_, h_, n * n_ff_, INFINITY, st_);
        lin(B.ffn_down, h_, cur_, n);
        dbg("ffn", il, cur_, n_embd_);
        axpy(cur_, 1.f, r_, n * n_embd_, true, st_);
        return;
    }
    if (!(mmq_ && n >= kStageRows)) moe_router(L, n);
}

void Lfm2Moe::layer_tail(Layer&, int, int n) { axpy(fo_, 1.f, r_, n * n_embd_, true, st_); }

void Lfm2Moe::output_rows(int first, int rows, float* h) {
    rms_norm(r_ + (size_t) first * n_embd_, fp(out_norm_), h, n_embd_, rows, rms_eps_, st_);
}

// No expert is shared: the routed experts' sum starts from nothing.  add_scaled_kernel, moe_sum_kernel and
// moe_add() all accumulate into fo_, so it must be cleared first.  Gemma 4 does the same (gemma4.cpp:495-498),
// and its memset runs inside moe_shared() under a replayed graph (engine.cpp:2634), which is what makes this one
// capturable too.
void Lfm2Moe::moe_shared(const Layer&, int, int n) {
    ck(cudaMemsetAsync(fo_, 0, (size_t) n * n_embd_ * sizeof(float), cs(st_)), "experts");
}

// What ggml's MoE graph does for this model: sigmoid gates, the n_used largest of sigmoid + exp_probs_b, and the
// weights from the bare sigmoids, normalized.  The router's matrix is read-only (h_rl_); tmp is a thread's own.
void Lfm2Moe::route(int il, const float* logits, int* ids, float* w, float* tmp) const {
    const float* bias = bias_.data() + (size_t) il * n_expert_;
    float* probs = tmp;
    float* pick = tmp + n_expert_;
    for (int e = 0; e < n_expert_; ++e) {
        probs[e] = 1.f / (1.f + std::exp(-logits[e]));
        pick[e] = probs[e] + bias[(size_t) e];   // the bias moves the selection alone
    }
    float sum = 0.f;
    for (int k = 0; k < n_used_; ++k) {
        int best = 0;
        for (int e = 1; e < n_expert_; ++e)
            if (pick[e] > pick[best]) best = e;   // strict: the first index wins a tie
        pick[best] = -INFINITY;
        ids[k] = best;
        w[k] = probs[best];   // unbiased
        sum += w[k];
    }
    const float denom = std::max(sum, kMinWeightSum);
    for (int k = 0; k < n_used_; ++k) w[k] /= denom;

    // Opt-in (STRATA_DUMP_ROUTE): the ids, the normalized weights and every biased selection score, so a top-1
    // mismatch against llama.cpp's ffn_moe_probs_biased / ffn_moe_topk can be explained before anything is tuned.
    if (dump_route_) {
        std::lock_guard<std::mutex> lock(g_route_mutex);
        std::fprintf(stderr, "route il=%d ids=", il);
        for (int k = 0; k < n_used_; ++k) std::fprintf(stderr, "%s%d", k ? "," : "", ids[k]);
        std::fprintf(stderr, " w=");
        for (int k = 0; k < n_used_; ++k) std::fprintf(stderr, "%s%.6g", k ? "," : "", w[k]);
        std::fprintf(stderr, " biased=");
        for (int e = 0; e < n_expert_; ++e)
            std::fprintf(stderr, "%s%.6g", e ? "," : "", probs[e] + bias[(size_t) e]);
        std::fprintf(stderr, "\n");
    }
}

// The recordings of an attention layer's head are numbered by the rows they are sized for; a short-convolution
// layer's never reads a position.
int Lfm2Moe::head_key(int il, int n) const { return blocks_[(size_t) il].conv ? 0 : room_key(n); }

}  // namespace strata::poly
