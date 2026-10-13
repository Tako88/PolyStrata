// What does releasing VMM pieces do to this stack's accounting? (plan M6.1)
//
// On ROCm 7.2.4 / gfx1200 hipMemUnmap + hipMemRelease returns nothing to hipMemGetInfo: after releasing
// 2 GiB in 32 pieces, "free" is byte-identical, while the process can re-create and re-map the same 2 GiB
// (the runtime keeps the physical memory in its own pool).  src/poly/engine.cpp decides whether it may grow
// the expert cache from exactly that number -- cover()'s free_b and mind_card_vram() -- so a cache that gave
// VRAM up can never take it back, which is the shape the live session of 2026-10-13 walked to 0 experts
// (0.0% cache hits, ~40 tok/s, ~14.8 GiB held for a 0-byte cache).  M6's fix is that the arena keeps the
// pieces it releases and re-maps them instead of asking the driver again.
//
// The file records the stack's behaviour, and the M6 gate is phase B: if unmapping a piece and re-mapping its
// own handle (no hipMemCreate) steps the card's kernel-used figure up like the release-then-recreate record
// does (2.93 -> 4.93 GiB), the free list is not a win and M6.2-M6.5 do not ship.  Both phases mirror
// Engine::vmm_map's order: one hipMemAddressReserve, then per piece hipMemCreate -> hipMemMap ->
// hipMemSetAccess; the shrink order is vmm_unmap's, from the back.
//
// Reporting, not gating, in the style of tests/hip/vmm_probe.cpp: every call prints "holds" / "DOES NOT HOLD"
// (with hipGetErrorString() when it failed) and the process exits 0 whenever it ran at all.  It exits 77
// (ctest "skipped") when there is no HIP device.  Deliberately not wired into CMake: a new ctest entry would
// move the hip_ test set off its 20-test baseline (16 Passed, 4 Skipped).
//
//   Phase A  the release-then-recreate record (the original probe, kept as the comparison): map 32 x 64 MiB,
//            unmap the last 16 and hipMemRelease them, unmap and release the rest, then hipMemCreate 32 fresh
//            handles and map them into the same addresses.
//   Phase B  the M6 shape: map 32 x 64 MiB, unmap the last 16 WITHOUT hipMemRelease, re-map those same 16
//            handles (no hipMemCreate) and hipMemSetAccess them, write a byte through each re-mapped piece
//            and read it back, then tear down.
//
//   hipcc -O0 --rocm-path=/opt/rocm --rocm-device-lib-path=/opt/rocm/amdgcn/bitcode \
//     tests/hip/vmm_release_probe.cpp -o /tmp/vmm_release_probe && /tmp/vmm_release_probe
//
//   /tmp/vmm_release_probe       both phases, in this order (the plan's M6.1 command)
//   /tmp/vmm_release_probe A|B   one phase in a process of its own: the pool phase A leaves behind cannot
//                                colour phase B's readings
//
// Run it with the engine stopped: with an engine resident neither free VRAM nor the card's used figure says
// anything about this process.

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace {

int not_held = 0;

// A check whose answer is a boolean we already know.
bool report(const char* what, bool holds) {
    std::printf("  %-58s %s\n", what, holds ? "holds" : "DOES NOT HOLD");
    if (!holds) ++not_held;
    return holds;
}

// A check that is a HIP call: hipGetErrorString() on the failure.
bool report_call(const char* what, hipError_t err) {
    const bool holds = err == hipSuccess;
    std::printf("  %-58s %s", what, holds ? "holds" : "DOES NOT HOLD");
    if (!holds) std::printf("  (%s)", hipGetErrorString(err));
    std::printf("\n");
    if (!holds) ++not_held;
    return holds;
}

double free_mib() {
    size_t f = 0, t = 0;
    hipMemGetInfo(&f, &t);
    return (double) f / 1048576.0;
}

// The card's own figure, from the kernel: hipMemGetInfo does not count what the runtime's pool holds.
double kernel_used_mib() {
    long v = 0, best = 0;
    for (int card = 0; card < 8; ++card) {
        const std::string path = "/sys/class/drm/card" + std::to_string(card) + "/device/mem_info_vram_used";
        std::ifstream f(path);
        if (f >> v && v > best) best = v;
    }
    return (double) best / 1048576.0;
}

void show(const char* what) {
    std::printf("%-34s hipMemGetInfo free %8.2f MiB | kernel used %8.2f MiB\n", what, free_mib(), kernel_used_mib());
}

__global__ void fill_bytes(unsigned char* p, size_t n, unsigned char v) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] = v;
}

unsigned int blocks(size_t n, unsigned int block) { return (unsigned int) ((n + block - 1) / block); }

// Phase A: what the original probe measured.  Release everything, then create it again.
void phase_a(const hipMemAllocationProp& prop, const hipMemAccessDesc& access, const size_t piece,
             const size_t n, const size_t total) {
    std::printf("\nPhase A - release then recreate (the original probe, the comparison)\n");
    show("start");

    void* base = nullptr;
    if (!report_call("hipMemAddressReserve(2 GiB)", hipMemAddressReserve(&base, total, 0, nullptr, 0)))
        return;
    std::vector<hipMemGenericAllocationHandle_t> h(n, nullptr);
    for (size_t i = 0; i < n; ++i) {
        char label[80];
        std::snprintf(label, sizeof label, "hipMemCreate piece %zu", i + 1);
        if (!report_call(label, hipMemCreate(&h[i], piece, &prop, 0))) return;
        std::snprintf(label, sizeof label, "hipMemMap piece %zu", i + 1);
        if (!report_call(label, hipMemMap((char*) base + i * piece, piece, 0, h[i], 0))) return;
        std::snprintf(label, sizeof label, "hipMemSetAccess piece %zu", i + 1);
        if (!report_call(label, hipMemSetAccess((char*) base + i * piece, piece, &access, 1))) return;
    }
    show("mapped 2 GiB in 32 pieces");

    for (size_t i = n / 2; i < n; ++i) {   // half, from the end (vmm_unmap's order)
        if (hipMemUnmap((char*) base + i * piece, piece) != hipSuccess) { std::printf("unmap %zu failed\n", i); return; }
        if (hipMemRelease(h[i]) != hipSuccess) { std::printf("release %zu failed\n", i); return; }
    }
    hipDeviceSynchronize();
    show("unmapped+released 16 pieces");

    for (size_t i = 0; i < n / 2; ++i) {   // and the rest
        if (hipMemUnmap((char*) base + i * piece, piece) != hipSuccess) { std::printf("unmap %zu failed\n", i); return; }
        if (hipMemRelease(h[i]) != hipSuccess) { std::printf("release %zu failed\n", i); return; }
    }
    hipDeviceSynchronize();
    show("unmapped+released all 32");

    bool regrew = true;                    // can the process take the memory back?
    for (size_t i = 0; i < n; ++i) {
        if (hipMemCreate(&h[i], piece, &prop, 0) != hipSuccess) { regrew = false; std::printf("re-create %zu FAILED\n", i); break; }
        if (hipMemMap((char*) base + i * piece, piece, 0, h[i], 0) != hipSuccess) { regrew = false; std::printf("re-map %zu FAILED\n", i); break; }
        if (hipMemSetAccess((char*) base + i * piece, piece, &access, 1) != hipSuccess) { regrew = false; std::printf("re-setaccess %zu FAILED\n", i); break; }
    }
    show(regrew ? "re-mapped all 32 (regrow ok)" : "regrow FAILED (see above)");

    for (size_t i = 0; i < n; ++i) {
        hipMemUnmap((char*) base + i * piece, piece);
        hipMemRelease(h[i]);
    }
    hipMemAddressFree(base, total);
    hipDeviceSynchronize();
    show("freed and done");
}

// Phase B: the M6 shape.  Keep the handles of the pieces that are given up, and map them again.
void phase_b(const hipMemAllocationProp& prop, const hipMemAccessDesc& access, const size_t piece,
             const size_t n, const size_t total) {
    std::printf("\nPhase B - unmap without release, re-map the same handles (M6's free list)\n");
    show("start");

    void* base = nullptr;
    if (!report_call("hipMemAddressReserve(2 GiB)", hipMemAddressReserve(&base, total, 0, nullptr, 0)))
        return;
    std::vector<hipMemGenericAllocationHandle_t> h(n, nullptr);
    for (size_t i = 0; i < n; ++i) {
        char label[80];
        std::snprintf(label, sizeof label, "hipMemCreate piece %zu", i + 1);
        if (!report_call(label, hipMemCreate(&h[i], piece, &prop, 0))) return;
        std::snprintf(label, sizeof label, "hipMemMap piece %zu", i + 1);
        if (!report_call(label, hipMemMap((char*) base + i * piece, piece, 0, h[i], 0))) return;
        std::snprintf(label, sizeof label, "hipMemSetAccess piece %zu", i + 1);
        if (!report_call(label, hipMemSetAccess((char*) base + i * piece, piece, &access, 1))) return;
    }
    show("mapped 2 GiB in 32 pieces");

    for (size_t i = n / 2; i < n; ++i)   // unmap the last 16 and keep every handle
        if (!report_call("hipMemUnmap(the last 16, no release)",
                         hipMemUnmap((char*) base + i * piece, piece)))
            return;
    hipDeviceSynchronize();
    show("unmapped 16, handles kept");

    bool holds = true;
    for (size_t i = n / 2; i < n; ++i) {   // and map those same handles back, piece by piece, as vmm_map does
        char label[80];
        std::snprintf(label, sizeof label, "hipMemMap(handle %zu back at its own offset)", i + 1);
        if (!report_call(label, hipMemMap((char*) base + i * piece, piece, 0, h[i], 0))) { holds = false; break; }
        std::snprintf(label, sizeof label, "hipMemSetAccess(piece %zu again)", i + 1);
        if (!report_call(label, hipMemSetAccess((char*) base + i * piece, piece, &access, 1))) { holds = false; break; }
    }
    hipDeviceSynchronize();
    show(holds ? "re-mapped the same 16 handles" : "re-map FAILED (see above)");

    // The pieces must be usable, not merely mapped: a byte through each one, written by a kernel and read
    // back -- the engine writes experts into the pieces it re-maps.
    bool pattern = holds;
    hipStream_t stream = nullptr;
    if (holds && report_call("hipStreamCreate", hipStreamCreate(&stream))) {
        const unsigned int block = 256;
        unsigned char* whole = (unsigned char*) base;
        for (size_t i = 0; i < n; ++i) {   // one distinct byte per piece, over the mapped prefix of it
            const unsigned char v = (unsigned char) (0x40 + i);
            fill_bytes<<<blocks(piece, block), block, 0, stream>>>(whole + i * piece, piece, v);
            if (!report_call("hipStreamSynchronize (a kernel wrote the piece)", hipStreamSynchronize(stream)))
                { pattern = false; break; }
            std::vector<unsigned char> back(piece, 0);
            if (!report_call("hipMemcpy reads the piece back", hipMemcpy(back.data(), whole + i * piece, piece, hipMemcpyDeviceToHost)))
                { pattern = false; break; }
            for (size_t j = 0; j < piece; ++j)
                if (back[j] != v) { pattern = false; break; }
            if (!pattern) break;
        }
        hipStreamDestroy(stream);
    }
    report("every piece (re-mapped ones included) holds what the kernel wrote", pattern);
    show("wrote and read through all 32 pieces");

    for (size_t i = 0; i < n; ++i) {
        hipMemUnmap((char*) base + i * piece, piece);
        hipMemRelease(h[i]);
    }
    hipMemAddressFree(base, total);
    hipDeviceSynchronize();
    show("freed and done");
}

}  // namespace

int main(int argc, char** argv) {
    int runtime = 0;
    if (hipRuntimeGetVersion(&runtime) != hipSuccess) return 1;
    int count = 0;
    if (hipGetDeviceCount(&count) != hipSuccess || count < 1) {
        std::printf("no HIP device\n");
        return 77;   // ctest "skipped"
    }
    int device = 0;
    if (hipGetDevice(&device) != hipSuccess) device = 0;
    hipDeviceProp_t props = {};
    if (hipGetDeviceProperties(&props, device) == hipSuccess)
        std::printf("HIP runtime %d, device %d: %s (%s)\n", runtime, device, props.name, props.gcnArchName);
    else
        std::printf("HIP runtime %d, device %d\n", runtime, device);

    hipMemAllocationProp prop = {};
    prop.type = hipMemAllocationTypePinned;
    prop.location.type = hipMemLocationTypeDevice;
    prop.location.id = device;
    hipMemAccessDesc access = {};
    access.location = prop.location;
    access.flags = hipMemAccessFlagsProtReadWrite;

    const size_t piece = size_t(64) << 20, n = 32;   // 32 x 64 MiB = 2 GiB, like the arena's pieces
    const char* which = argc > 1 ? argv[1] : nullptr;
    if (!which || which[0] == 'A' || which[0] == 'a') phase_a(prop, access, piece, n, piece * n);
    if (!which || which[0] == 'B' || which[0] == 'b') phase_b(prop, access, piece, n, piece * n);

    std::printf("\n%d check(s) did not hold\n", not_held);
    (void) hipGetLastError();
    return 0;
}
