// src/artifact/gguf_reader_test.cpp - the reader refuses what it cannot arbitrate (synthetic GGUF, no model,
// no GPU).
//
//   1. two tensors with distinct names: the file opens and find() sees both;
//   2. the same name twice: refused at open, and the error names the tensor and the file - GGUF has no
//      index to say which of the two a name means, and find() is first-match, so the alternative is a
//      lookup that silently picks by position.
//   3. no tensors, and the file ends at its header, before the 32-byte-aligned data start: the file opens,
//      with data_start() clamped to the file size.  A split model's first shard can hold metadata only and
//      end unpadded (unsloth's UD-Q5_K_XL shard 1 ends 6 bytes before its aligned data start; llama.cpp's
//      gguf-py reads it); nothing is ever read from its data section.
//   4. one tensor, and the file ends at its header: still refused at open ("data section starts past EOF").
//
// The fixture is a minimal GGUF v3 written here (header, no metadata, two F32[8] tensors, 32-byte
// alignment), so the test needs neither gguf-py nor a shard.  The file is closed before it is removed:
// on Windows an open mapping keeps it.
#include "strata/artifact/gguf_reader.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

namespace {
int g_fail = 0;
void check(bool ok, const char* what) {
    std::printf("  %-66s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}

template <class T> void put(std::vector<uint8_t>& b, T v) {
    const auto n = b.size();
    b.resize(n + sizeof(T));
    std::memcpy(b.data() + n, &v, sizeof(T));
}
void put_str(std::vector<uint8_t>& b, const std::string& s) {
    put<uint64_t>(b, s.size());
    b.insert(b.end(), s.begin(), s.end());
}

// A GGUF v3 file of F32[8] tensors named `names`, laid out 32 bytes apart from a 32-byte-aligned data start.
// `header_only`: the file ends right after the tensor directory - no alignment padding, no data.
std::filesystem::path write_gguf(const std::vector<std::string>& names, bool header_only = false) {
    std::vector<uint8_t> b;
    put<uint32_t>(b, 0x46554747u);   // "GGUF"
    put<uint32_t>(b, 3);
    put<uint64_t>(b, names.size());  // n_tensors
    put<uint64_t>(b, 0);             // n_kv
    for (size_t i = 0; i < names.size(); ++i) {
        put_str(b, names[i]);
        put<uint32_t>(b, 1);         // n_dims
        put<uint64_t>(b, 8);
        put<uint32_t>(b, 0);         // F32
        put<uint64_t>(b, 32 * i);    // offset from data_start
    }
    if (!header_only) b.resize((b.size() + 31) / 32 * 32 + 32 * names.size(), 0);
    const auto path = std::filesystem::temp_directory_path() / "strata_gguf_reader_test.gguf";
    std::ofstream(path, std::ios::binary).write(reinterpret_cast<const char*>(b.data()), (std::streamsize)b.size());
    return path;
}

// The metadata values this test writes: an INT32, an INT32 array, a STRING array (the GGUF MetaType ids 5, 9, 8).
std::vector<uint8_t> i32_value(int32_t x) {
    std::vector<uint8_t> v;
    put<uint32_t>(v, 5);   // I32
    put<int32_t>(v, x);
    return v;
}
std::vector<uint8_t> i32_array_value(const std::vector<int32_t>& xs) {
    std::vector<uint8_t> v;
    put<uint32_t>(v, 9);   // ARRAY
    put<uint32_t>(v, 5);   // of I32
    put<uint64_t>(v, xs.size());
    for (int32_t x : xs) put<int32_t>(v, x);
    return v;
}
std::vector<uint8_t> str_array_value(const std::vector<std::string>& xs) {
    std::vector<uint8_t> v;
    put<uint32_t>(v, 9);   // ARRAY
    put<uint32_t>(v, 8);   // of STRING
    put<uint64_t>(v, xs.size());
    for (const std::string& s : xs) put_str(v, s);
    return v;
}

// A metadata-only GGUF v3 (no tensors) with the KV entries given.  Padded, so the reader opens it and get()
// sees every key.
std::filesystem::path write_gguf_kv(const std::vector<std::pair<std::string, std::vector<uint8_t>>>& kvs) {
    std::vector<uint8_t> b;
    put<uint32_t>(b, 0x46554747u);   // "GGUF"
    put<uint32_t>(b, 3);
    put<uint64_t>(b, 0);             // n_tensors
    put<uint64_t>(b, kvs.size());    // n_kv
    for (const auto& kv : kvs) {
        put_str(b, kv.first);
        b.insert(b.end(), kv.second.begin(), kv.second.end());
    }
    b.resize((b.size() + 31) / 32 * 32);
    const auto path = std::filesystem::temp_directory_path() / "strata_gguf_reader_meta_test.gguf";
    std::ofstream(path, std::ios::binary).write(reinterpret_cast<const char*>(b.data()), (std::streamsize)b.size());
    return path;
}
}  // namespace

int main() {
    std::printf("gguf_reader_test\n");
    {
        const auto path = write_gguf({"blk.0.attn_q.weight", "blk.0.attn_k.weight"});
        std::string err;
        size_t n = 0;
        bool both = false;
        try {
            strata::GgufFile g(path.string());
            n = g.tensors().size();
            both = g.find("blk.0.attn_q.weight") && g.find("blk.0.attn_k.weight");
        } catch (const std::exception& e) {
            err = e.what();
        }
        std::filesystem::remove(path);
        check(err.empty() && n == 2 && both, "distinct names: the file opens and find() sees both tensors");
    }
    {
        const auto path = write_gguf({"blk.0.attn_q.weight", "blk.0.attn_q.weight"});
        std::string err;
        try {
            strata::GgufFile g(path.string());
        } catch (const std::exception& e) {
            err = e.what();
        }
        std::filesystem::remove(path);
        check(!err.empty(), "the same name twice: refused at open");
        check(err.find("blk.0.attn_q.weight") != std::string::npos, "  the error names the tensor");
        check(err.find(path.filename().string()) != std::string::npos, "  the error names the file");
        if (!err.empty()) std::printf("  (%s)\n", err.c_str());
    }
    {
        const auto path = write_gguf({}, true);   // 24 bytes: the aligned data start (32) is past EOF
        std::string err;
        uint64_t size = 0, start = 1;
        size_t n = 1;
        try {
            strata::GgufFile g(path.string());
            size = g.file_size();
            start = g.data_start();
            n = g.tensors().size();
        } catch (const std::exception& e) {
            err = e.what();
        }
        std::filesystem::remove(path);
        check(err.empty() && n == 0, "no tensors, ends before the aligned data start: the file opens");
        check(err.empty() && start == size, "  data_start() is the file size (nothing to read past it)");
        if (!err.empty()) std::printf("  (%s)\n", err.c_str());
    }
    {
        const auto path = write_gguf({"blk.0.attn_q.weight"}, true);
        std::string err;
        try {
            strata::GgufFile g(path.string());
        } catch (const std::exception& e) {
            err = e.what();
        }
        std::filesystem::remove(path);
        check(err.find("data section starts past EOF") != std::string::npos,
              "one tensor, ends before the aligned data start: refused at open");
    }
    // A per-layer array is read layer by layer: a numeric array of a model's layer count is kept whole, a scalar
    // is every layer's value, a short array is refused by name, and a string array stays a sample.
    {
        std::vector<int32_t> xs(88);
        for (int i = 0; i < 88; ++i) xs[(size_t) i] = 1000 + i;
        const auto path = write_gguf_kv({{"nemotron_h_moe.head_count_kv", i32_array_value(xs)}});
        bool ok = false;
        try {
            strata::GgufFile g(path.string());
            const strata::MetaValue* v = g.get("nemotron_h_moe.head_count_kv");
            ok = v && v->type == strata::MetaType::ARRAY && v->items.size() == 88 &&
                 strata::meta_layer_value(*v, 0, 88) == 1000 && strata::meta_layer_value(*v, 87, 88) == 1087;
        } catch (const std::exception& e) {
            std::printf("  (%s)\n", e.what());
        }
        std::filesystem::remove(path);
        check(ok, "a numeric array of 88: all 88 kept, meta_layer_value() reads the last");
    }
    {
        const auto path = write_gguf_kv({{"nemotron_h_moe.expert_used_count", i32_value(9)}});
        bool ok = false;
        try {
            strata::GgufFile g(path.string());
            const strata::MetaValue* v = g.get("nemotron_h_moe.expert_used_count");
            ok = v && strata::meta_layer_value(*v, 0, 88) == 9 && strata::meta_layer_value(*v, 87, 88) == 9;
        } catch (const std::exception& e) {
            std::printf("  (%s)\n", e.what());
        }
        std::filesystem::remove(path);
        check(ok, "a scalar: the same value for every layer");
    }
    {
        const auto path = write_gguf_kv({{"nemotron_h_moe.feed_forward_length", i32_array_value({1, 2, 3})}});
        bool threw = false;
        std::string msg;
        try {
            strata::GgufFile g(path.string());
            const strata::MetaValue* v = g.get("nemotron_h_moe.feed_forward_length");
            (void) strata::meta_layer_value(*v, 87, 88);
        } catch (const std::exception& e) {
            threw = true;
            msg = e.what();
        }
        std::filesystem::remove(path);
        check(threw, "a 3-value array for 88 layers: meta_layer_value() throws");
        check(msg.find("88") != std::string::npos, "  the error names the layer count");
    }
    {
        std::vector<std::string> toks(131072);
        for (size_t i = 0; i < toks.size(); ++i) toks[i] = "t" + std::to_string(i);
        const auto path = write_gguf_kv({{"tokenizer.ggml.tokens", str_array_value(toks)}});
        bool ok = false;
        try {
            strata::GgufFile g(path.string());
            const strata::MetaValue* v = g.get("tokenizer.ggml.tokens");
            ok = v && v->type == strata::MetaType::ARRAY && v->count == 131072 && v->items.size() == 64;
        } catch (const std::exception& e) {
            std::printf("  (%s)\n", e.what());
        }
        std::filesystem::remove(path);
        check(ok, "a 131,072-entry string array: still sampled to 64, count kept");
    }
    std::printf(g_fail ? "gguf_reader_test: %d FAILED\n" : "gguf_reader_test: all passed\n", g_fail);
    return g_fail ? 1 : 0;
}
