// Does the engine's multi-piece VMM path work on this stack? (plan M1: branch A vs branch B)
//
// src/poly/engine.cpp grows the attention caches and the expert cache by reserving one address
// range and mapping VRAM into it a piece at a time (vmm_reserve/vmm_map/vmm_unmap/vmm_free,
// :447-500): reserve up_to(bytes, granularity) addresses, then for each piece hipMemCreate,
// hipMemMap at the running offset, and hipMemSetAccess read-write; later it unmaps and releases
// the trailing pieces to hand VRAM back to the other cache. Whether ROCm 7.2 on gfx1200 supports
// that sequence decides M2's branch: branch A maps VRAM this way (the VMM adapter), branch B
// compiles the path out (one dmalloc per cache at load, the --kv-whole shape).
//
// Reporting, not gating, in the style of tests/hip/mapped_alias.cpp: every call prints
// "holds" / "DOES NOT HOLD" (with hipGetErrorString() when it failed), and the process exits 0
// whenever it ran at all. It exits 77 (ctest "skipped") when there is no HIP device.
//
//   Phase 1  the primitives, one piece: attribute -> granularity -> reserve -> create -> map at
//            offset 0 -> SetAccess -> a kernel writes a byte and hipMemcpy reads it back ->
//            unmap -> release -> free the addresses.
//   Phase 2  the engine's actual sequence with N = 4 pieces: reserve N*g, create N handles, map
//            each at a piece offset, SetAccess per piece, run a stream kernel across the whole
//            range, unmap the last piece only, map a different handle into the freed slot,
//            SetAccess it, read it back, then tear the whole thing down.

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdio>
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

hipMemAllocationProp pinned_prop(int device) {
    hipMemAllocationProp prop = {};
    prop.type = hipMemAllocationTypePinned;
    prop.location.type = hipMemLocationTypeDevice;
    prop.location.id = device;
    return prop;
}

hipMemAccessDesc readwrite(const hipMemAllocationProp& prop) {
    hipMemAccessDesc access = {};
    access.location = prop.location;
    access.flags = hipMemAccessFlagsProtReadWrite;
    return access;
}

__global__ void fill_bytes(unsigned char* p, size_t n, unsigned char v) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] = v;
}

// A distinct byte per piece, so each piece can be checked on its own.
__global__ void fill_pieces(unsigned char* p, size_t n, size_t piece) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] = (unsigned char) (0x10 + i / piece);
}

unsigned int blocks(size_t n, unsigned int block) {
    return (unsigned int) ((n + block - 1) / block);
}

void phase1(int device) {
    std::printf("\nPhase 1 - the primitives\n");
    const hipMemAllocationProp prop = pinned_prop(device);
    const hipMemAccessDesc access = readwrite(prop);

    int supported = 0;
    report_call("hipDeviceGetAttribute(hipDeviceAttributeVirtualMemoryManagementSupported)",
                hipDeviceGetAttribute(&supported, hipDeviceAttributeVirtualMemoryManagementSupported, device));
    report("the device reports VMM support", supported != 0);

    size_t g = 0;
    if (!report_call("hipMemGetAllocationGranularity(hipMemAllocationGranularityMinimum)",
                     hipMemGetAllocationGranularity(&g, &prop, hipMemAllocationGranularityMinimum)))
        return;
    report("the granularity is a positive number of bytes", g > 0);
    if (g == 0) return;
    std::printf("  granularity %zu bytes\n", g);

    void* base = nullptr;
    if (!report_call("hipMemAddressReserve(g)", hipMemAddressReserve(&base, g, 0, nullptr, 0))) return;

    hipMemGenericAllocationHandle_t h = nullptr;
    if (!report_call("hipMemCreate(g)", hipMemCreate(&h, g, &prop, 0))) {
        (void) hipMemAddressFree(base, g);
        return;
    }
    if (!report_call("hipMemMap(at offset 0)", hipMemMap(base, g, 0, h, 0))) {
        (void) hipMemRelease(h);
        (void) hipMemAddressFree(base, g);
        return;
    }
    if (!report_call("hipMemSetAccess(readwrite)", hipMemSetAccess(base, g, &access, 1))) {
        (void) hipMemUnmap(base, g);
        (void) hipMemRelease(h);
        (void) hipMemAddressFree(base, g);
        return;
    }

    const unsigned char want = 0xA5;
    unsigned char got = 0;
    fill_bytes<<<1, 1>>>(static_cast<unsigned char*>(base), 1, want);
    report_call("a kernel writes a byte", hipDeviceSynchronize());
    report_call("hipMemcpy reads it back", hipMemcpy(&got, base, 1, hipMemcpyDeviceToHost));
    report("the byte round-trips", got == want);

    report_call("hipMemUnmap", hipMemUnmap(base, g));
    report_call("hipMemRelease", hipMemRelease(h));
    report_call("hipMemAddressFree", hipMemAddressFree(base, g));
}

void phase2(int device) {
    std::printf("\nPhase 2 - the engine's multi-piece sequence\n");
    const hipMemAllocationProp prop = pinned_prop(device);
    const hipMemAccessDesc access = readwrite(prop);

    size_t g = 0;
    if (!report_call("hipMemGetAllocationGranularity(hipMemAllocationGranularityMinimum)",
                     hipMemGetAllocationGranularity(&g, &prop, hipMemAllocationGranularityMinimum)))
        return;
    if (g == 0) {
        report("the granularity is a positive number of bytes", false);
        return;
    }

    const int N = 4;
    const size_t reserve = size_t(N) * g;
    void* base = nullptr;
    if (!report_call("hipMemAddressReserve(N*g)", hipMemAddressReserve(&base, reserve, 0, nullptr, 0)))
        return;

    std::vector<hipMemGenericAllocationHandle_t> handle(N, nullptr);
    std::vector<bool> created(N, false), mapped(N, false);
    for (int i = 0; i < N; ++i) {
        char label[80];
        std::snprintf(label, sizeof label, "hipMemCreate piece %d of %d", i + 1, N);
        if (report_call(label, hipMemCreate(&handle[i], g, &prop, 0))) created[i] = true;
    }

    // Map each handle at its own piece offset and give it access, as the engine does per piece.
    for (int i = 0; i < N; ++i) {
        if (!created[i]) continue;
        void* at = static_cast<unsigned char*>(base) + size_t(i) * g;
        char label[80];
        std::snprintf(label, sizeof label, "hipMemMap piece %d at offset %d*g", i + 1, i);
        if (!report_call(label, hipMemMap(at, g, 0, handle[i], 0))) continue;
        mapped[i] = true;
        std::snprintf(label, sizeof label, "hipMemSetAccess piece %d", i + 1);
        report_call(label, hipMemSetAccess(at, g, &access, 1));
    }

    hipStream_t stream = nullptr;
    if (!report_call("hipStreamCreate", hipStreamCreate(&stream))) stream = nullptr;

    const unsigned int block = 256;
    fill_pieces<<<blocks(reserve, block), block, 0, stream>>>(static_cast<unsigned char*>(base), reserve, g);
    report_call("a stream kernel runs across the whole range", hipStreamSynchronize(stream));

    std::vector<unsigned char> whole(reserve, 0);
    report_call("read the whole range back", hipMemcpy(whole.data(), base, reserve, hipMemcpyDeviceToHost));
    bool pattern = true;
    for (size_t i = 0; i < reserve; ++i)
        if (whole[i] != (unsigned char) (0x10 + i / g)) pattern = false;
    report("every piece holds what the kernel wrote", pattern);

    // Unmap the last piece only, then map a different handle into the freed slot.
    const int last = N - 1;
    void* last_at = static_cast<unsigned char*>(base) + size_t(last) * g;
    if (mapped[last]) {
        report_call("hipMemUnmap the last piece only", hipMemUnmap(last_at, g));
        mapped[last] = false;
    }
    if (created[last]) {
        report_call("hipMemRelease the last piece", hipMemRelease(handle[last]));
        created[last] = false;
        handle[last] = nullptr;
    }
    hipMemGenericAllocationHandle_t fresh = nullptr;
    if (report_call("hipMemCreate a different handle for the freed slot", hipMemCreate(&fresh, g, &prop, 0))) {
        handle[last] = fresh;
        created[last] = true;
        if (report_call("hipMemMap it into the freed slot", hipMemMap(last_at, g, 0, fresh, 0))) {
            mapped[last] = true;
            report_call("hipMemSetAccess the freed slot", hipMemSetAccess(last_at, g, &access, 1));
            const unsigned char mark = 0x7E;
            fill_bytes<<<blocks(g, block), block, 0, stream>>>(static_cast<unsigned char*>(last_at), g, mark);
            report_call("a kernel writes the fresh piece", hipStreamSynchronize(stream));
            std::vector<unsigned char> back(g, 0);
            report_call("read the fresh piece back", hipMemcpy(back.data(), last_at, g, hipMemcpyDeviceToHost));
            bool fresh_ok = true;
            for (size_t i = 0; i < g; ++i)
                if (back[i] != mark) fresh_ok = false;
            report("the fresh piece reads back what the kernel wrote", fresh_ok);
            std::vector<unsigned char> rest(size_t(N - 1) * g, 0);
            report_call("read the surviving pieces back",
                        hipMemcpy(rest.data(), base, rest.size(), hipMemcpyDeviceToHost));
            bool rest_ok = true;
            for (size_t i = 0; i < rest.size(); ++i)
                if (rest[i] != (unsigned char) (0x10 + i / g)) rest_ok = false;
            report("the surviving pieces are unchanged", rest_ok);
        }
    }

    // Tear the whole thing down.
    std::printf("  tear down\n");
    if (stream) {
        (void) hipStreamSynchronize(stream);
        (void) hipStreamDestroy(stream);
    }
    for (int i = 0; i < N; ++i) {
        void* at = static_cast<unsigned char*>(base) + size_t(i) * g;
        char label[80];
        if (mapped[i]) {
            std::snprintf(label, sizeof label, "hipMemUnmap piece %d", i + 1);
            report_call(label, hipMemUnmap(at, g));
        }
        if (created[i]) {
            std::snprintf(label, sizeof label, "hipMemRelease piece %d", i + 1);
            report_call(label, hipMemRelease(handle[i]));
        }
    }
    report_call("hipMemAddressFree(N*g)", hipMemAddressFree(base, reserve));
}

}  // namespace

int main() {
    int runtime = 0;
    if (hipRuntimeGetVersion(&runtime) != hipSuccess) return 1;
    int count = 0;
    if (hipGetDeviceCount(&count) != hipSuccess || count < 1) {
        std::printf("no HIP device\n");
        return 77;  // ctest "skipped"
    }
    int device = 0;
    if (hipGetDevice(&device) != hipSuccess) device = 0;
    hipDeviceProp_t props = {};
    if (hipGetDeviceProperties(&props, device) == hipSuccess)
        std::printf("HIP runtime %d, device %d: %s (%s)\n", runtime, device, props.name, props.gcnArchName);
    else
        std::printf("HIP runtime %d, device %d\n", runtime, device);

    phase1(device);
    phase2(device);

    std::printf("\n%d check(s) did not hold\n", not_held);
    (void) hipGetLastError();
    return 0;
}
