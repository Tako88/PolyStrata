// src/poly/nemotron.cpp - the layers of NVIDIA-Nemotron-3.5-Bolt-74B-A9B (llama.cpp architecture
// "nemotron_h_moe").  The reference for the geometry and the load is llama.cpp's nemotron-h loader and its
// mamba-base graph (ref/src/models/mamba-base.cpp, ref-nemotron-h.cpp); the Mamba-2/attention/MoE forward is
// M4/M5 of the plan and is not here yet.
#include "nemotron.hpp"

#include "runtime.hpp"
#include "strata/poly/kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace strata::poly {
namespace {

constexpr size_t kScoreFloats = size_t(16) << 20;  // the attention's scores of the tokens it takes at a time
// The forward pass is the next milestone's: this class loads and allocates, and anything that multiplies a
// token refuses by name rather than answering with something wrong.
const char* const kForward = "nemotron_h_moe: the forward pass is not built yet (milestones M4/M5)";

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

// Per-token buffers hold rows_ tokens, row after row.  The values stay unwritten until the forward is built
// (M4/M5); the sizes here are the ones the plan's M3 loads and allocates.
void Nemotron::assign_scratch() {
    const size_t n = (size_t) rows_;
    const size_t heads = n * n_head_ * head_dim_, kv = n * n_kv_ * head_dim_;
    proj_ = fbuf(n * d_in_proj_);
    xbc_ = fbuf(n * conv_dim_);
    conv_ = fbuf(n * conv_dim_);
    z_ = fbuf(n * d_inner_);
    y_ = fbuf(n * d_inner_);
    dt_ = fbuf(n * n_ssm_head_);
    lat_ = fbuf(n * moe_latent_);
    lat_acc_ = fbuf(n * moe_latent_);
    latq_ = fbuf((kernels::native_q8_1_bytes(moe_latent_) * n + sizeof(float) - 1) / sizeof(float));
    q_ = fbuf(heads);
    kraw_ = fbuf(kv);
    k_ = fbuf(kv);
    v_ = fbuf(kv);
    att_ = fbuf(heads);
    score_ = fbuf((size_t) chunk_ * n_head_ * opt_.max_context);
    if (rows_ >= kStageRows) awork_ = fbuf(heads + heads / 2 + 1 + 3 * n * n_head_);   // a prompt batch's attention
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

// ---- the forward pass, which is M4/M5 of the plan ----
void Nemotron::layer_head(Layer&, int, int) { throw std::runtime_error(kForward); }
void Nemotron::layer_tail(Layer&, int, int) { throw std::runtime_error(kForward); }
void Nemotron::output_rows(int, int, float*) { throw std::runtime_error(kForward); }
void Nemotron::route(int, const float*, int32_t, int*, float*, float*) const { throw std::runtime_error(kForward); }
void Nemotron::moe_shared(const Layer&, int, int) { throw std::runtime_error(kForward); }

}  // namespace strata::poly
