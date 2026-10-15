// src/poly/nemotron.hpp - NVIDIA's Nemotron-H with routed experts (llama.cpp architecture "nemotron_h_moe"):
// NVIDIA-Nemotron-3.5-Bolt-74B-A9B.
//
// One residual stream.  A block is one of three kinds, told apart by its metadata: a Mamba-2 block (a projection
// that splits into z, x|B|C and dt, a depthwise convolution of four taps, a selective scan and a grouped RMS
// norm), an attention block (grouped queries and keys, no rotary embedding -- the architecture maps to
// LLAMA_ROPE_TYPE_NONE), or a latent MoE block.  Every block has one attn_norm ahead of its sublayer and adds
// the residual after it.
//
// The MoE block projects the normed input down to a narrow latent width, runs the routed experts on that latent
// with ReLU² (they have no gate), projects the result back up, and adds a shared expert computed on the full
// width.  The routed experts' counts and widths are per layer, read from the tensors and cross-checked against
// the metadata arrays.
#pragma once

#include "engine.hpp"

namespace strata::poly {

class Nemotron final : public Engine {
public:
    size_t card_row_bytes() const override;
    void save_card_rows(int64_t first, int64_t n, uint8_t* dst) override;
    void load_card_rows(const uint8_t* src, int64_t from, int64_t count, int64_t first, int64_t n) override;

protected:
    void load_geometry() override;
    void load_layer(int il) override;
    void begin_pass() override;
    void check_draft_block() override;
    size_t recurrent_floats() const override;
    void alloc_caches() override;
    void assign_scratch() override;
    int widest() const override;
    int streams() const override { return 1; }
    bool snaps_spread() const override { return true; }
    int head_key(int il, int n) const override;
    void layer_head(Layer& L, int il, int n) override;
    void layer_tail(Layer& L, int il, int n) override;
    void output_rows(int first, int rows, float* h) override;
    void route(int il, const float* logits, int32_t token, int* ids, float* w, float* tmp) const override;
    void moe_shared(const Layer& L, int il, int n) override;
    void save_layer(int il, int64_t pos, uint8_t* dst) override;

private:
    /// A layer's weights beside the ones every architecture has (Layer), and its place in the state and caches.
    struct Block {
        bool recr = false;   ///< a Mamba-2 block
        bool attn = false;   ///< an attention block
        bool moe = false;    ///< a latent MoE block
        // Mamba-2
        Dev ssm_in, ssm_conv, ssm_conv_b, ssm_dt_b, ssm_a, ssm_d, ssm_norm, ssm_out;
        float* past = nullptr;   ///< the conv's d_conv - 1 = 3 past inputs a channel (3 * conv_dim_)
        float* s = nullptr;      ///< the scan's state: {d_state, head_dim, n_ssm_head_} (d_inner_ * d_state_)
        // attention
        Dev wq, wk, wv, wo;
        uint16_t* keys = nullptr;   ///< max_context rows of n_kv_ * head_dim_ fp16
        uint16_t* vals = nullptr;
        // latent MoE: the down/up projections around the experts, the shared expert, and its selection bias
        Dev lat_down, lat_up, sh_up, sh_down;
        std::vector<float> probs_b;
        Dev probs_b_dev;
    };

    /// The Mamba-2 block's forward: the projection, the convolution, the scan and the grouped norm.
    void mamba(Block& B, int il, int n);

    std::vector<Block> blocks_;      ///< one a layer of layers_, plus the draft block (there is none here)
    std::vector<uint8_t> kind_;      ///< a layer's kind: 0 Mamba-2, 1 attention, 2 MoE
    std::vector<int> kv_heads_;      ///< a layer's key heads; 0 for a Mamba-2 block
    std::vector<int> used_;          ///< a MoE layer's n_used
    std::vector<int> ff_exp_;        ///< a MoE layer's n_ff_exp
    int n_head_ = 0;                 ///< attention query heads
    int n_kv_ = 0;                   ///< attention key heads (the same on every attention block)
    int head_dim_ = 0;               ///< attention head width
    int n_ssm_head_ = 0;             ///< the scan's heads (ssm.time_step_rank)
    int n_group_ = 0;                ///< the scan's groups (ssm.group_count)
    int d_conv_ = 0;                 ///< the convolution's taps
    int d_inner_ = 0;                ///< the scan's width (ssm.inner_size)
    int d_state_ = 0;                ///< the scan's state width per group (ssm.state_size)
    int d_in_proj_ = 0;              ///< 2 * d_inner_ + 2 * n_group_ * d_state_ + n_ssm_head_
    int conv_dim_ = 0;               ///< d_inner_ + 2 * n_group_ * d_state_
    int moe_latent_ = 0;             ///< the MoE's narrow width (moe_latent_size, 0 -> n_embd_)
    int n_ssm_ = 0, n_attn_ = 0;     ///< the layers counted by begin_pass(), in layer order
    bool w_norm_ = true;             ///< the routed experts' weights are made to sum to one
    float w_scale_ = 0.f;            ///< and then scaled by this, when it is not 0
    int chunk_ = 1;                  ///< tokens the attention takes at a time

    // device scratch, row after row for rows_ tokens
    float* proj_ = nullptr;      ///< the ssm_in projection (d_in_proj_ a token)
    float* xbc_ = nullptr;       ///< the x|B|C slice before the convolution (conv_dim_ a token)
    float* conv_ = nullptr;      ///< the convolution's output (conv_dim_ a token)
    float* z_ = nullptr;         ///< the gate part of the projection (d_inner_ a token)
    float* y_ = nullptr;         ///< the scan's output (d_inner_ a token)
    float* dt_ = nullptr;        ///< the time steps (n_ssm_head_ a token)
    float* lat_ = nullptr;       ///< the MoE's latent (moe_latent_ a token)
    void* latq_ = nullptr;       ///< lat_ in Q8_1
    float* lat_acc_ = nullptr;   ///< the routed experts' latent sum
    float* q_ = nullptr, *kraw_ = nullptr, *k_ = nullptr, *v_ = nullptr, *att_ = nullptr, *score_ = nullptr;
    float* awork_ = nullptr;     ///< a prompt batch's attention (attend_batch)
};

}  // namespace strata::poly
