// src/poly/lfm2moe.hpp - LiquidAI LFM2.5 with routed experts (llama.cpp architecture "lfm2moe"), as
// LFM2.5-24B-A2B is.
//
// One residual stream.  The attention layers (attention.head_count_kv != 0) have attention over the keys and
// values of every token so far, with an RMSNorm over a head and a rotary embedding; the rest have LFM2's short
// convolution: a depthwise causal convolution of shortconv.l_cache taps over b*x, multiplied by c, with the
// taps - 1 past b*x values a channel kept as the layer's state.  Every layer norms the residual before its
// operator and again before its FFN; the first leading_dense_block_count layers have a dense SwiGLU FFN and the
// rest routed experts with no shared one (their sum starts from nothing).  The output matrix is the token table
// and the output norm is token_embd_norm.weight.
#pragma once

#include "engine.hpp"

namespace strata::poly {

class Lfm2Moe final : public Engine {
public:
    size_t card_row_bytes() const override;
    void save_card_rows(int64_t first, int64_t n, uint8_t* dst) override;
    void load_card_rows(const uint8_t* src, int64_t from, int64_t count, int64_t first, int64_t n) override;

protected:
    void load_geometry() override;
    void load_layer(int il) override;
    void begin_pass() override;
    size_t recurrent_floats() const override;
    bool snaps_spread() const override { return true; }
    void save_layer(int il, int64_t pos, uint8_t* dst) override;
    void alloc_caches() override;
    void assign_scratch() override;
    int widest() const override;
    int streams() const override { return 1; }
    void layer_head(Layer& L, int il, int n) override;
    void layer_tail(Layer& L, int il, int n) override;
    void output_rows(int first, int rows, float* h) override;
    void route(int il, const float* logits, int* ids, float* w, float* tmp) const override;
    void moe_shared(const Layer& L, int il, int n) override;
    int head_key(int il, int n) const override;
    const Names* out_norm_alias() const override;

private:
    /// A layer's weights beside the ones every architecture has (Layer), and its places in the state and caches.
    struct Block {
        bool conv = false;   ///< the short convolution; else attention over the cache
        Dev in_proj, conv_w, out_proj;
        Dev wq, wk, wv, wo, q_norm, k_norm;
        Dev ffn_gate, ffn_up, ffn_down;   ///< the dense FFN
        float* past = nullptr;            ///< the convolution's: (taps - 1) * n_embd
        uint16_t* keys = nullptr;         ///< max_context * n_kv * head_dim, fp16
        uint16_t* vals = nullptr;
    };

    std::vector<Block> blocks_;   ///< one a layer of layers_
    std::vector<int> kv_;         ///< per layer: the heads with keys of their own, 0 for a short convolution
    std::vector<float> bias_;     ///< per (layer, expert): exp_probs_b, added to the selection score alone
    int n_head_ = 0, n_kv_ = 0, head_dim_ = 0, n_rot_ = 0;
    float theta_scale_ = 0;   ///< a rotary pair turns by position * theta_scale^pair
    int n_conv_ = 0, n_attn_ = 0;
    int taps_ = 0;            ///< shortconv.l_cache: the convolution's taps, and one more than the state's values
    int leading_dense_ = 0;   ///< the layers before this one have the dense FFN
    bool dump_route_ = false; ///< STRATA_DUMP_ROUTE: print the router's ids, weights and biased scores
    int chunk_ = 1;           ///< tokens the attention takes at a time

    // device scratch
    float *proj_ = nullptr, *conv_ = nullptr, *qraw_ = nullptr, *q_ = nullptr, *kraw_ = nullptr, *k_ = nullptr,
          *v_ = nullptr, *att_ = nullptr, *score_ = nullptr;
    float* awork_ = nullptr;   ///< a prompt batch's attention (attend_batch)
};

}  // namespace strata::poly
