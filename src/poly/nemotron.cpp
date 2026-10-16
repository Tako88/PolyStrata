// src/poly/nemotron.cpp - the layers of NVIDIA-Nemotron-3.5-Bolt-74B-A9B (llama.cpp architecture
// "nemotron_h_moe").  The reference for the geometry and the load is llama.cpp's nemotron-h loader and its
// mamba-base graph (ref/src/models/mamba-base.cpp, ref-nemotron-h.cpp); the Mamba-2 block's forward is M4, and
// the attention, the latent MoE with its shared expert and the routing are M5 (forward pass complete).
#include "nemotron.hpp"

#include "runtime.hpp"
#include "strata/poly/kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace strata::poly {
namespace {

constexpr size_t kScoreFloats = size_t(16) << 20;  // the attention's scores of the tokens it takes at a time
constexpr float kMinWeightSum = 6.103515625e-5f;   // the floor ggml's MoE graph puts under the gate sum

// M4's independent check of the decay path.  `--dump-layers` forces window = rows = 1, so Engine::dbg() ever sees
// only a call's first row, and pos = 1 of a 2-token call (where the scan's `s[j] = s0[j] * dA + B[j] * x_dt` has
// read `dA = expf(dt_sp * a[h])`) is never printed.  STRATA_NEMOTRON_DUMP=1 prints every row here, in dbg()'s own
// format and at pos + row, so a 2-token call can be held against the reference's mamba2_y_add_d-0 {64,128,2,1}.
// Arch-local: with the variable unset this is a no-op and no other architecture's dump output changes.
void dump_rows(int64_t pos, const char* name, int il, const float* dev, int width, int rows, cudaStream_t st) {
    const char* e = std::getenv("STRATA_NEMOTRON_DUMP");
    if (!e || !*e || *e == '0') return;
    std::vector<float> h((size_t) width * rows);
    ck(cudaStreamSynchronize(st), "dump");
    ck(cudaMemcpy(h.data(), dev, h.size() * sizeof(float), cudaMemcpyDeviceToHost), "dump");
    for (int t = 0; t < rows; ++t) {
        const float* r = h.data() + (size_t) t * width;
        double sum = 0, abs = 0;
        for (int i = 0; i < width; ++i) {
            sum += r[i];
            abs += std::fabs(r[i]);
        }
        std::fprintf(stderr, "dump pos=%lld blk=%d %-8s n=%d sum=%.6f abs=%.6f first=%.6f %.6f %.6f\n",
                     (long long) (pos + t), il, name, width, sum, abs, r[0], r[1], r[2]);
    }
}

}  // namespace

void Nemotron::begin_pass() {
    blocks_.assign(layers_.size(), Block{});
    n_ssm_ = n_attn_ = 0;
}

void Nemotron::check_draft_block() {
    const MetaValue* nextn = meta_in(*gg_, "nextn_predict_layers");
    if ((nextn && nextn->u != 0) || tensor(n_layer_, "nextn.eh_proj.weight"))
        throw std::runtime_error("nemotron_h_moe: a NextN/MTP head is not one this engine runs");
}

void Nemotron::load_geometry() {
    load_shape();
    auto find = [&](const char* key) { return meta_in(*gg_, key); };
    auto need = [](bool ok, const std::string& what) {
        if (!ok) throw std::runtime_error(std::string("unsupported model: ") + what);
    };

    n_head_ = shape_i("attention.head_count");
    head_dim_ = shape_i("attention.key_length");
    need(n_head_ > 0 && n_head_ <= 256, "bad attention.head_count");
    need(head_dim_ > 0, "bad attention.key_length");
    need(shape_i("attention.value_length") == head_dim_, "the attention's keys and values differ in length");

    d_conv_ = shape_i("ssm.conv_kernel");
    d_inner_ = shape_i("ssm.inner_size");
    d_state_ = shape_i("ssm.state_size");
    n_ssm_head_ = shape_i("ssm.time_step_rank");
    n_group_ = shape_i("ssm.group_count");
    need(d_conv_ == 4, "ssm.conv_kernel is not 4");
    need(d_inner_ > 0 && d_state_ > 0 && n_ssm_head_ > 0 && n_group_ > 0, "bad ssm sizes");
    d_in_proj_ = 2 * d_inner_ + 2 * n_group_ * d_state_ + n_ssm_head_;
    conv_dim_ = d_inner_ + 2 * n_group_ * d_state_;
    // The scan's per-head width d_inner_ / n_ssm_head_ is derived here, not stored (64 in this file), and the
    // reference asserts both divisions (ref/src/models/mamba-base.cpp:164,183).
    need(d_inner_ % n_ssm_head_ == 0, "ssm.inner_size is not a multiple of ssm.time_step_rank");
    need(d_inner_ % n_group_ == 0, "ssm.inner_size is not a multiple of ssm.group_count");

    const MetaValue* lat = find("moe_latent_size");
    moe_latent_ = lat ? (int) lat->u : 0;
    if (moe_latent_ == 0) moe_latent_ = n_embd_;
    need(moe_latent_ > 0, "bad moe_latent_size");
    // The routed experts' wider side is the latent, both around them and in the MoE's launches.
    moe_in_w_ = moe_out_w_ = moe_latent_;

    need(shape_i("expert_gating_func") == 2, "the expert gate is not a sigmoid");
    const MetaValue* norm = find("expert_weights_norm");
    w_norm_ = norm ? norm->u != 0 : true;
    const MetaValue* scale = find("expert_weights_scale");
    w_scale_ = scale ? (float) scale->num() : 0.f;

    // NoPE.  LLM_ARCH_NEMOTRON_H(_MOE) maps to LLAMA_ROPE_TYPE_NONE (ref/src/llama-model.cpp:3467-3472), so no
    // attention block turns anything: rope.dimension_count (128) and rope.scaling.finetuned (False) are read into
    // evidence and used for nothing, and no rope launch exists in the forward.

    kv_heads_.assign((size_t) n_layer_, 0);
    used_.assign((size_t) n_layer_, 0);
    ff_exp_.assign((size_t) n_layer_, 0);
    kind_.assign((size_t) n_layer_, 0);
    int first_recr = -1;
    for (int il = 0; il < n_layer_; ++il) {
        const int h = shape_layer("attention.head_count_kv", il);
        need(h == 0 || n_head_ % h == 0, "a layer's key heads do not divide its query heads");
        kv_heads_[(size_t) il] = h;
        if (h) {
            need(n_kv_ == 0 || n_kv_ == h, "the attention layers have different key heads");
            n_kv_ = h;
        }
        const int ff = shape_layer("feed_forward_length", il);
        const int used = shape_layer("expert_used_count", il);
        const int ff_exp = shape_layer("expert_feed_forward_length", il);
        // The reference's classification: recurrent iff both are 0, else attention iff n_ff is 0, else MoE
        // (ref-nemotron-h.cpp:15-17).
        const bool recr = h == 0 && ff == 0;
        const bool attn = !recr && ff == 0;
        kind_[(size_t) il] = recr ? 0 : attn ? 1 : 2;
        used_[(size_t) il] = used;
        ff_exp_[(size_t) il] = ff_exp;
        // ... cross-checked against the tensors the file actually has (ref-nemotron-h.cpp:112-131).
        const bool ssm = tensor(il, "ssm_in.weight") != nullptr;
        const bool q = tensor(il, "attn_q.weight") != nullptr;
        const bool gate = tensor(il, "ffn_gate_inp.weight") != nullptr;
        const std::string say = "ssm_in=" + std::string(ssm ? "yes" : "no") + ", attn_q=" +
                                std::string(q ? "yes" : "no") + ", ffn_gate_inp=" + std::string(gate ? "yes" : "no");
        if (recr) {
            if (!ssm || q || gate)
                throw std::runtime_error(blk(il, "") + " is recurrent by its metadata, but its tensors say " + say);
            if (first_recr < 0) first_recr = il;
        } else if (attn) {
            if (!q || ssm || gate)
                throw std::runtime_error(blk(il, "") + " is attention by its metadata, but its tensors say " + say);
        } else {
            if (!gate || ssm || q)
                throw std::runtime_error(blk(il, "") + " is a MoE layer by its metadata, but its tensors say " + say);
        }
        // Host note 4: on a MoE layer the reference takes the expert width from expert_feed_forward_length and
        // falls back to feed_forward_length / expert_used_count when that key is absent (ref-nemotron-h.cpp:117-119).
        // This engine refuses to guess which of the two a file means: both numbers are named.
        if (!recr && !attn && ff_exp != ff)
            throw std::runtime_error(blk(il, "expert_feed_forward_length") + " is " + std::to_string(ff_exp) +
                                     " and " + blk(il, "feed_forward_length") + " is " + std::to_string(ff) +
                                     ": the experts' width is the one the two agree on");
    }
    need(n_kv_ > 0, "no attention layer");

    // The file's own projection and convolution widths, both numbers in the message.
    const TensorInfo* sin = first_recr >= 0 ? tensor(first_recr, "ssm_in.weight") : nullptr;
    need(sin && sin->shape.size() == 2 && (int64_t) sin->shape[0] == n_embd_ &&
             (int64_t) sin->shape[1] == d_in_proj_,
         "ssm_in.weight is " + (sin ? shape_of(*sin) : std::string("missing")) + ", expected [" +
             std::to_string(n_embd_) + ", " + std::to_string(d_in_proj_) + "]");
    const TensorInfo* sconv = first_recr >= 0 ? tensor(first_recr, "ssm_conv1d.weight") : nullptr;
    need(sconv && sconv->shape.size() == 2 && (int64_t) sconv->shape[0] == d_conv_ &&
             (int64_t) sconv->shape[1] == conv_dim_,
         "ssm_conv1d.weight is " + (sconv ? shape_of(*sconv) : std::string("missing")) + ", expected [" +
             std::to_string(d_conv_) + ", " + std::to_string(conv_dim_) + "]");

    // nothing here clamps a SwiGLU
    clamp_exp_.assign((size_t) n_layer_ + 1, INFINITY);
    clamp_shexp_ = clamp_exp_;
}

void Nemotron::load_layer(int il) {
    // The draft block's index is one past the model's layers: it is a NextN/MTP head, which this engine refuses
    // (check_draft_block()).  The base engine calls load_layer() for it before load_drafter(), and kind_ has only
    // the model's layers, so the refusal has to come first.
    if (il >= n_layer_) {
        check_draft_block();
        throw std::runtime_error("nemotron_h_moe: a NextN/MTP head is not one this engine runs");
    }
    Layer& L = layers_[(size_t) il];
    Block& B = blocks_[(size_t) il];
    auto f32 = [&](const char* n, int64_t a, int64_t b = 1) { return upload(tensor(il, n), blk(il, n), a, b, kF32); };
    auto mat = [&](const char* n, int64_t a, int64_t b) { return upload(tensor(il, n), blk(il, n), a, b, kAnyMat); };
    // An NVFP4 matrix keeps its type, so the file's [1]-shaped multiplier has to be read beside it and fed to
    // Dev::scale, which lin() multiplies by.
    auto scaled = [&](const char* n, const char* s, int64_t a, int64_t b) {
        Dev d = mat(n, a, b);
        const TensorInfo* sc = tensor(il, s);
        if (!sc || sc->type != kTypeF32 || sc->elements() != 1)
            throw std::runtime_error(blk(il, s) + " is " + (sc ? shape_of(*sc) : std::string("missing")) +
                                     ", expected F32 [1]");
        home(*sc).payload(*sc);
        std::memcpy(&d.scale, home(*sc).data(*sc), sizeof(float));
        return d;
    };

    L.attn_norm = f32("attn_norm.weight", n_embd_);   // the only norm; ffn_norm stays null (no draft path)
    L.fixed = true;   // every block's launches read the position on the card when they are recorded

    const int k = kind_[(size_t) il];
    if (k == 0) {
        B.recr = true;
        B.ssm_in = scaled("ssm_in.weight", "ssm_in.scale", n_embd_, d_in_proj_);
        B.ssm_out = scaled("ssm_out.weight", "ssm_out.scale", d_inner_, n_embd_);
        B.ssm_conv = f32("ssm_conv1d.weight", d_conv_, conv_dim_);
        B.ssm_conv_b = f32("ssm_conv1d.bias", conv_dim_);
        B.ssm_dt_b = f32("ssm_dt.bias", n_ssm_head_);
        B.ssm_a = f32("ssm_a", 1, n_ssm_head_);
        B.ssm_d = f32("ssm_d", 1, n_ssm_head_);
        B.ssm_norm = f32("ssm_norm.weight", d_inner_ / n_group_, n_group_);
        ++n_ssm_;
        return;
    }
    if (k == 1) {
        B.attn = true;
        B.wq = mat("attn_q.weight", n_embd_, (int64_t) n_head_ * head_dim_);
        B.wk = mat("attn_k.weight", n_embd_, (int64_t) n_kv_ * head_dim_);
        B.wv = mat("attn_v.weight", n_embd_, (int64_t) n_kv_ * head_dim_);
        B.wo = mat("attn_output.weight", (int64_t) n_head_ * head_dim_, n_embd_);
        ++n_attn_;
        return;
    }

    // A latent MoE layer.  The per-layer truth is the tensor shapes (the reference's mechanism); the metadata's
    // arrays are the cross-check, and a disagreement is refused by name.
    B.moe = true;
    const TensorInfo* router = tensor(il, "ffn_gate_inp.weight");
    const TensorInfo* down = tensor(il, "ffn_down_exps.weight");
    if (!router || router->shape.size() != 2 || !down || down->shape.size() != 3)
        throw std::runtime_error(blk(il, "ffn_gate_inp.weight") + " or " + blk(il, "ffn_down_exps.weight") +
                                 " is missing or not the shape a MoE layer has");
    const int n_expert = (int) router->shape[1];
    const int n_ff_exp = (int) down->shape[0];
    if (n_ff_exp != ff_exp_[(size_t) il])
        throw std::runtime_error(blk(il, "ffn_down_exps.weight") + " is " + shape_of(*down) +
                                 ", but " + blk(il, "expert_feed_forward_length") + " says " +
                                 std::to_string(ff_exp_[(size_t) il]));
    if (used_[(size_t) il] <= 0 || n_expert < used_[(size_t) il] || n_expert > n_expert_)
        throw std::runtime_error(blk(il, "ffn_gate_inp.weight") + " has " + std::to_string(n_expert) +
                                 " experts; the metadata's expert_used_count is " +
                                 std::to_string(used_[(size_t) il]) + " and expert_count is " +
                                 std::to_string(n_expert_));
    if (tensor(il, "ffn_gate_exps.weight"))
        throw std::runtime_error(blk(il, "ffn_gate_exps.weight") +
                                 " is present: this engine runs these experts ungated (ReLU²) only");
    L.n_expert = n_expert;
    L.n_used = used_[(size_t) il];
    L.n_ff_exp = n_ff_exp;
    // The model-wide stride of the slot, profile and routing arrays is the largest layer; keep it so before any
    // of them is sized (alloc() runs after the two load passes).
    n_expert_ = std::max(n_expert_, L.n_expert);
    L.gu_planes = 1;   // ffn_up_exps alone, no gate: ReLU²
    L.act = 2;
    load_moe(il, /*shared=*/false);
    B.lat_down = mat("ffn_latent_down.weight", n_embd_, moe_latent_);
    B.lat_up = mat("ffn_latent_up.weight", moe_latent_, n_embd_);
    L.sh_up = mat("ffn_up_shexp.weight", n_embd_, n_ff_shexp_);
    L.sh_down = mat("ffn_down_shexp.weight", n_ff_shexp_, n_embd_);
    B.probs_b.assign((size_t) L.n_expert, 0.f);
    const TensorInfo* b = tensor(il, "exp_probs_b.bias");
    if (!b || b->type != kTypeF32 || (int64_t) b->elements() != L.n_expert)
        throw std::runtime_error(blk(il, "exp_probs_b.bias") + " is " + (b ? shape_of(*b) : std::string("missing")) +
                                 ", expected F32 [" + std::to_string(L.n_expert) + "]");
    home(*b).payload(*b);
    std::memcpy(B.probs_b.data(), home(*b).data(*b), (size_t) L.n_expert * sizeof(float));
    B.probs_b_dev = upload(b, blk(il, "exp_probs_b.bias"), L.n_expert, 1, kF32);
}

// Every recurrent layer's scan state and convolution window together, in layer order: alloc_caches() hands each
// Mamba-2 block its B.past and then its B.s out of state_.
size_t Nemotron::recurrent_floats() const {
    return size_t(n_ssm_) * (size_t(d_inner_) * d_state_ + size_t(3) * conv_dim_);
}

int Nemotron::widest() const { return std::max(d_in_proj_, n_embd_); }

void Nemotron::alloc_caches() {
    float* st = state_;
    for (int il = 0; il < (int) blocks_.size(); ++il) {
        Block& B = blocks_[(size_t) il];
        if (!runs(il)) continue;
        if (B.recr) {
            B.past = st;
            st += size_t(3) * conv_dim_;
            B.s = st;
            st += size_t(d_inner_) * d_state_;
        } else if (B.attn) {
            const size_t row = (size_t) n_kv_ * head_dim_ * sizeof(uint16_t);
            B.keys = (uint16_t*) kv_alloc(row, 1);
            B.vals = (uint16_t*) kv_alloc(row, 1);
        }
    }
    chunk_ = (int) std::clamp<size_t>(kScoreFloats / ((size_t) n_head_ * opt_.max_context), 1, (size_t) rows_);
}

// Per-token buffers hold rows_ tokens, row after row.  The Mamba-2 block reads its x|B|C slice and its gate out
// of proj_ (mamba()), so there is no separate buffer for either; the MoE's latent-up projection lands in tmp_.
void Nemotron::assign_scratch() {
    const size_t n = (size_t) rows_;
    const size_t heads = n * n_head_ * head_dim_, kv = n * n_kv_ * head_dim_;
    proj_ = fbuf(n * d_in_proj_);
    conv_ = fbuf(n * conv_dim_);
    y_ = fbuf(n * d_inner_);
    dt_ = fbuf(n * n_ssm_head_);
    lat_ = fbuf(n * moe_latent_);
    lat_acc_ = fbuf(n * moe_latent_);
    tmp_ = fbuf(n * n_embd_);
    latq_ = fbuf((kernels::native_q8_1_bytes(moe_latent_) * n + sizeof(float) - 1) / sizeof(float));
    q_ = fbuf(heads);
    k_ = fbuf(kv);
    v_ = fbuf(kv);
    att_ = fbuf(heads);
    score_ = fbuf((size_t) chunk_ * n_head_ * opt_.max_context);
    if (rows_ >= kStageRows) awork_ = fbuf(heads + heads / 2 + 1 + 3 * n * n_head_);   // a prompt batch's attention
}

// The latent MoE's expert I/O lives in the arch's own scratch (moe_in_ = lat_, moe_acc_ = lat_acc_) instead of the
// base engine's shifted xn_/fo_, so those buffers have to move with the row the base shift_rows() moves.  A window
// with experts on the CPU goes through the layers in two halves (run(), engine.cpp:2607); without this, half 1's
// layer_head() would write lat_/latq_ at the unshifted base and clobber half 0's before half 0's moe() consumed it.
// moe_in_/moe_acc_ re-point in layer_head(), which always runs after the shift.  The rest of the scratch (q_, k_, v_,
// att_, score_, awork_, u_, dt_) is produced and consumed inside one head call, so it needs nothing here.
void Nemotron::shift_rows(ptrdiff_t by) {
    Engine::shift_rows(by);
    lat_ += by * moe_latent_;
    latq_ = (uint8_t*) latq_ + by * (ptrdiff_t) kernels::native_q8_1_bytes(moe_latent_);
    lat_acc_ += by * moe_latent_;
    tmp_ += by * n_embd_;
}

// A Mamba-2 layer's scan state and convolution window lie one behind the other in state_ (alloc_caches()).
void Nemotron::save_layer(int il, int64_t, uint8_t* dst) {
    const Block& B = blocks_[(size_t) il];
    if (!B.past) return;
    const size_t n = size_t(3) * conv_dim_ + size_t(d_inner_) * d_state_;
    ck(cudaMemcpy(dst + (size_t) (B.past - state_) * sizeof(float), B.past, n * sizeof(float),
                  cudaMemcpyDeviceToHost),
       "save state");
}

size_t Nemotron::card_row_bytes() const { return (size_t) n_attn_ * 2 * n_kv_ * head_dim_ * sizeof(uint16_t); }

void Nemotron::save_card_rows(int64_t first, int64_t n, uint8_t* dst) {
    if (first < 0 || n < 0 || first + n > opt_.max_context)
        throw std::logic_error("save_rows: the rows are outside the context");
    ck(cudaStreamSynchronize(cs(st_)), "save rows");
    const size_t row = (size_t) n_kv_ * head_dim_ * sizeof(uint16_t), at = (size_t) first * row, bytes = (size_t) n * row;
    for (const Block& B : blocks_) {
        if (!B.attn || bytes == 0) continue;
        for (const uint16_t* cache : {B.keys, B.vals}) {
            ck(cudaMemcpy(dst, (const uint8_t*) cache + at, bytes, cudaMemcpyDeviceToHost), "save rows");
            dst += bytes;
        }
    }
}

void Nemotron::load_card_rows(const uint8_t* src, int64_t from, int64_t count, int64_t first, int64_t n) {
    if (from < 0 || first < from || n < 0 || first + n > from + count || first + n > opt_.max_context)
        throw std::logic_error("load_rows: the rows are outside what was saved");
    cover(first + n);
    ck(cudaStreamSynchronize(cs(st_)), "load rows");
    const size_t row = (size_t) n_kv_ * head_dim_ * sizeof(uint16_t), at = (size_t) first * row, bytes = (size_t) n * row;
    const size_t skip = (size_t) (first - from) * row, all = (size_t) count * row;
    for (const Block& B : blocks_) {
        if (!B.attn || bytes == 0) continue;
        for (uint16_t* cache : {B.keys, B.vals}) {
            ck(cudaMemcpy((uint8_t*) cache + at, src + skip, bytes, cudaMemcpyHostToDevice), "load rows");
            src += all;
        }
    }
}

// ---- the forward pass: the Mamba-2 block (M4), the attention and the latent MoE with its shared expert (M5) ----

// The Mamba-2 block.  The projection splits into z, x|B|C and dt in that order (ref/src/models/mamba-base.cpp:
// 185-208); the convolution takes four taps over x|B|C and its SiLU; the scan carries the state; and the
// post-scan is silu(z)*y, the grouped RMS norm and ssm_out (with its own scale).
void Nemotron::mamba(Block& B, int il, int n) {
    const int ssm_dim = d_inner_ / n_ssm_head_;                  // the per-head width, derived (64)
    const int dt_off = 2 * d_inner_ + 2 * n_group_ * d_state_;   // the dt slice of the projection
    lin(B.ssm_in, xn_, proj_, n, xq_);                            // the file's scale multiplies (Dev::scale)
    // dt is n_head a token in the reference's strided view; the scan takes it compact, with ssm_dt.bias added
    // (ref/src/models/mamba-base.cpp:250).
    ck(cudaMemcpy2DAsync(dt_, (size_t) n_ssm_head_ * sizeof(float), proj_ + dt_off,
                         (size_t) d_in_proj_ * sizeof(float), (size_t) n_ssm_head_ * sizeof(float), (size_t) n,
                         cudaMemcpyDeviceToDevice, cs(st_)),
       "the Mamba-2 dt slice");
    add_bias(fp(B.ssm_dt_b), dt_, n_ssm_head_, n, st_);
    mamba_conv(proj_ + d_inner_, d_in_proj_, B.past, fp(B.ssm_conv), fp(B.ssm_conv_b), conv_, conv_dim_, n, st_,
               snap_bits_ ? snap_at_ + (B.past - state_) : nullptr, rec_n_, snap_bits_);
    mamba_scan(conv_, conv_ + d_inner_, conv_ + d_inner_ + (size_t) n_group_ * d_state_, dt_, fp(B.ssm_a),
               fp(B.ssm_d), B.s, y_, ssm_dim, n_ssm_head_, d_state_, n_group_, n, st_,
               snap_bits_ ? snap_at_ + (B.s - state_) : nullptr, rec_n_, snap_bits_);
    dbg("mamba2_y_add_d", il, y_, d_inner_);   // the reference's cb after y += x*ssm_d (folded into the scan)
    dump_rows(pos_, "mamba2_y_add_d", il, y_, d_inner_, n, cs(st_));   // every row, for the pos = 1 dA check
    mamba_norm(y_, proj_, d_in_proj_, fp(B.ssm_norm), d_inner_ / n_group_, n_group_, n, rms_eps_, st_);
    lin(B.ssm_out, y_, ao_, n);
}

// The attention block.  Queries, keys and values are one projection each; the keys and values take their rows in
// the caches at the position on the card (at_), so a recorded launch replays for every position; the attention's
// scale is 1/sqrt(head_dim) (ref-nemotron-h.cpp:285-289).  No rope and no head norm: this architecture maps to
// LLAMA_ROPE_TYPE_NONE (ref/src/llama-model.cpp:3467-3472).
void Nemotron::attn(Block& B, int64_t pos, int n) {
    const long long* at = recording_ ? at_ : nullptr;   // a recorded launch: the tokens from what at_ holds
    lin(B.wq, xn_, q_, n, xq_);
    lin(B.wk, xn_, k_, n, xq_);
    lin(B.wv, xn_, v_, n, xq_);
    if (at) {
        store_f16_at(k_, B.keys, n_kv_ * head_dim_, at, n, 0, st_);
        store_f16_at(v_, B.vals, n_kv_ * head_dim_, at, n, 0, st_);
    } else {
        store_f16(k_, B.keys, n_kv_ * head_dim_, pos, n, st_);
        store_f16(v_, B.vals, n_kv_ * head_dim_, pos, n, st_);
    }
    const float scale = 1.f / std::sqrt((float) head_dim_);
    // a prompt batch through cuBLAS; a few tokens with the kernels that give a token the same bits in any company
    if (n >= kStageRows)
        attend_batch(blas_, q_, B.keys, B.vals, pos, n, score_, (size_t) chunk_ * n_head_ * opt_.max_context, awork_,
                     att_, head_dim_, n_head_, n_kv_, scale, st_);
    else
        attend(q_, B.keys, B.vals, pos, n, score_, chunk_, att_, head_dim_, n_head_, n_kv_, scale, st_, 0, 0, false,
               at, at ? room(n) : 0);
    lin(B.wo, att_, ao_, n);
}

// A layer from its input to its residual.  Every block has one attn_norm and the sublayer reads that normed input.
// A Mamba-2 or attention block adds its sublayer output to the residual here; a latent MoE block does not -- its
// routed experts run on the latent, so the sum is completed in layer_tail().
void Nemotron::layer_head(Layer& L, int il, int n) {
    Block& B = blocks_[(size_t) il];
    rms_norm(r_, fp(L.attn_norm), xn_, n_embd_, n, rms_eps_, st_);
    kernels::native_quantize_q8_1(xn_, xq_, n_embd_ * n, 1, st_);
    dbg("attn_in", il, xn_, n_embd_);
    dump_rows(pos_, "attn_in", il, xn_, n_embd_, n, cs(st_));
    if (B.recr) {
        mamba(B, il, n);
    } else if (B.attn) {
        attn(B, pos_, n);
    } else {
        // The latent MoE: the experts' input is the narrow projection of the normed input, and the routed sum
        // lands in lat_acc_.  The plan's note 4: this loader refuses a file whose expert width key is absent, where
        // the reference falls back to n_ff/n_expert_used (ref-nemotron-h.cpp:117-119); this file has the array.
        lin(B.lat_down, xn_, lat_, n, xq_);
        kernels::native_quantize_q8_1(lat_, latq_, moe_latent_ * n, 1, st_);
        moe_in_ = lat_;
        moe_in_xq_ = latq_;
        moe_in_w_ = moe_out_w_ = moe_latent_;
        moe_acc_ = lat_acc_;
        // The routed experts' sum accumulates into moe_acc_ (add_scaled/moe_sum/moe_add), so it starts from
        // nothing here: lat_acc_ is one buffer for every MoE layer and every token, and without this the sum of a
        // layer and token would hold the ones before it.
        ck(cudaMemsetAsync(lat_acc_, 0, (size_t) n * moe_latent_ * sizeof(float), cs(st_)), "experts");
        if (!batched(L, n)) moe_router(L, n);   // a prompt batch routes in moe_rows()
        return;
    }
    dbg("attn", il, ao_, n_embd_);
    dump_rows(pos_, "attn", il, ao_, n_embd_, n, cs(st_));
    axpy(ao_, 1.f, r_, n * n_embd_, true, st_);
}

// An attention block's launches are sized by the rows they read; a Mamba-2 block's are the same for any size.
int Nemotron::head_key(int il, int n) const { return blocks_[(size_t) il].attn ? room_key(n) : 0; }

// A latent MoE layer's output: the routed sum projected back up to the width of the stream and added to the shared
// expert's output (the reference's `cur = moe_out + ffn_shexp`, ref-nemotron-h.cpp:336-343), then the residual.
// The shared expert ran in moe_shared() and left its output in fo_; the routed sum is in moe_acc_ (lat_acc_).
void Nemotron::layer_tail(Layer& L, int il, int n) {
    Block& B = blocks_[(size_t) il];
    dbg("ffn_moe_out", il, moe_acc(), moe_latent_);
    dump_rows(pos_, "ffn_moe_out", il, moe_acc(), moe_latent_, n, cs(st_));
    lin(B.lat_up, moe_acc(), tmp_, n);
    axpy(tmp_, 1.f, fo_, n * n_embd_, true, st_);
    dbg("ffn_out", il, fo_, n_embd_);
    dump_rows(pos_, "ffn_out", il, fo_, n_embd_, n, cs(st_));
    axpy(fo_, 1.f, r_, n * n_embd_, true, st_);
}

void Nemotron::output_rows(int first, int rows, float* h) {
    rms_norm(r_ + (size_t) first * n_embd_, fp(out_norm_), h, n_embd_, rows, rms_eps_, st_);
}

// The shared expert: an ungated up, ReLU², the down; its output goes to fo_, where the routed sum is added to it
// (ref-nemotron-h.cpp:338-343).  It runs on the GPU while the host routes.
void Nemotron::moe_shared(const Layer& L, int il, int n) {
    lin(L.sh_up, xn_, u_, n, xq_);
    relu2_rows(u_, n_ff_shexp_, n, st_);
    lin(L.sh_down, u_, fo_, n);
    dbg("ffn_shexp", il, fo_, n_embd_);
    dump_rows(pos_, "ffn_shexp", il, fo_, n_embd_, n, cs(st_));
}

// What ggml's MoE graph does for this model: sigmoid gates, the n_used largest of gate + bias, weights from the
// bare gates, made to sum to one, floored as ggml does, times the file's expert_weights_scale.  The per-layer
// counts are the layer's own (L.n_expert/L.n_used), not the model-wide maxima.  The reference's ffn_moe_topk is
// this selection (ref-nemotron-h.cpp:318-330).
void Nemotron::route(int il, const float* logits, int32_t token, int* ids, float* w, float* probs) const {
    const Layer& L = layers_[(size_t) il];
    const std::vector<float>& bias = blocks_[(size_t) il].probs_b;
    float* pick = probs + L.n_expert;
    for (int e = 0; e < L.n_expert; ++e) {
        probs[e] = 1.f / (1.f + std::exp(-logits[e]));
        pick[e] = probs[e] + bias[(size_t) e];
    }
    float sum = 0.f;
    for (int k = 0; k < L.n_used; ++k) {
        int best = 0;
        for (int e = 1; e < L.n_expert; ++e)
            if (pick[e] > pick[best]) best = e;
        pick[best] = -INFINITY;
        ids[k] = best;
        w[k] = probs[best];
        sum += w[k];
    }
    const float denom = w_norm_ ? std::max(sum, kMinWeightSum) : 1.f;
    for (int k = 0; k < L.n_used; ++k) {
        w[k] /= denom;
        if (w_scale_ != 0.f) w[k] *= w_scale_;
    }
    // Opt-in: the layer's chosen ids and normalised weights, on the host, beside the reference's ffn_moe_topk.
    if (const char* e = std::getenv("STRATA_NEMOTRON_ROUTE"); e && *e && *e != '0') {
        std::fprintf(stderr, "route blk=%d token=%d host", il, token);
        for (int k = 0; k < L.n_used; ++k) std::fprintf(stderr, " %d:%.6f", ids[k], w[k]);
        std::fprintf(stderr, "\n");
    }
}

// A prompt batch's routing on the card, and -- with STRATA_NEMOTRON_ROUTE set -- the card's picks and weights
// beside the host route()'s for the same scores.  A top-1 id mismatch is a failure whatever the weights.
bool Nemotron::route_card(int il, int n, int32_t* ids, float* w) {
    const Layer& L = layers_[(size_t) il];
    const Block& B = blocks_[(size_t) il];
    route_rows(rl_, L.n_expert, L.n_used, n, 1, 1.f, fp(B.probs_b_dev), nullptr, w_norm_, w_scale_, ids, w, st_);
    const char* e = std::getenv("STRATA_NEMOTRON_ROUTE");
    if (e && *e && *e != '0') {
        // The card's picks and the scores they came from, back to the host, then the same route() over them.
        std::vector<float> lr((size_t) n * L.n_expert);
        std::vector<float> cw((size_t) n * L.n_used);
        std::vector<int32_t> cid((size_t) n * L.n_used);
        ck(cudaStreamSynchronize(cs(st_)), "route dump");
        ck(cudaMemcpy(lr.data(), rl_, lr.size() * sizeof(float), cudaMemcpyDeviceToHost), "route dump");
        ck(cudaMemcpy(cw.data(), w, cw.size() * sizeof(float), cudaMemcpyDeviceToHost), "route dump");
        ck(cudaMemcpy(cid.data(), ids, cid.size() * sizeof(int32_t), cudaMemcpyDeviceToHost), "route dump");
        for (int t = 0; t < n; ++t) {
            int hid[kMaxUsed];
            float hw[kMaxUsed];
            std::vector<float> tmp((size_t) 2 * L.n_expert);
            route(il, lr.data() + (size_t) t * L.n_expert, t, hid, hw, tmp.data());
            std::fprintf(stderr, "route blk=%d token=%d card", il, t);
            for (int k = 0; k < L.n_used; ++k)
                std::fprintf(stderr, " %d:%.6f%s", cid[(size_t) t * L.n_used + k],
                             cw[(size_t) t * L.n_used + k], cid[(size_t) t * L.n_used + k] != hid[k] ? "(DIFF)" : "");
            std::fprintf(stderr, " | host");
            for (int k = 0; k < L.n_used; ++k) std::fprintf(stderr, " %d:%.6f", hid[k], hw[k]);
            std::fprintf(stderr, "\n");
        }
    }
    return true;
}

}  // namespace strata::poly
