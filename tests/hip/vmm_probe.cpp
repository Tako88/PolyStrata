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
//
//   Phase 3  the driver's mixed-size rule (M4b.1), one reservation per shape, mapped piece by
//            piece.  Measured on this stack (ROCm 7.2.4, gfx1200): a reservation that holds a
//            whole kArenaPiece (64 MiB) together with a smaller piece is refused at
//            hipMemSetAccess -- 15 x 64 MiB + 4 KiB fails 3/3, 15 x 64 MiB + 17829888 B fails,
//            4 KiB first then 15 x 64 MiB fails 5/5 -- while uniform pieces (16 x 64 MiB), a
//            single piece (16 x 64 MiB + 4 KiB), and a smaller piece that is only reserved (not
//            mapped) all hold.  The order matters, and the plan's "size order was tested and
//            refuted" is not what phase 3 measures: a larger piece after a smaller one fails
//            (64 MiB then 1024 MiB fails, 4 KiB then 64 MiB fails), while a smaller piece after a
//            larger one holds (1024 MiB then 64 MiB holds).  Alignment was tested and does not
//            matter.  The engine works around it by mapping whole pieces only
//            (src/poly/engine.cpp, M4b.2).
//            A driver that fixes the rule shows up here as the mixed shapes turning to "holds".
//            No shape allocates more than 1.1 GiB and each tears down before the next, so the
//            probe stays runnable beside a desktop; the diagnosis's 240-piece / 15 GiB case is
//            not reproduced (it needs the whole card).
//            Reporting, not gating: a DOES NOT HOLD here is the driver's rule, not a test
//            failure, and phase 3 does not change the exit code.

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

// M4b.1: the driver's mixed-size rule, recorded rather than asserted.  One reservation per shape;
// for each piece hipMemCreate -> hipMemMap at the running offset -> hipMemSetAccess; the shape stops
// at the first failure and the failing call is named.  Every shape tears down before the next.
size_t up_to(size_t n, size_t g) { return (n + g - 1) / g * g; }

struct Shape {
    const char* label;
    std::vector<size_t> pieces;   // mapped in this order, at the running offset
    size_t reserve_extra;         // reserved but never mapped (shape "the 4 KiB reserved but not mapped")
    bool mixed;                   // pieces of unequal size: the shapes the driver's rule is about
};

Shape make_shape(const char* label, std::vector<size_t> pieces, size_t extra, bool mixed) {
    Shape s;
    s.label = label;
    s.pieces = std::move(pieces);
    s.reserve_extra = extra;
    s.mixed = mixed;
    return s;
}

void phase3(int device) {
    std::printf("\nPhase 3 - the driver's mixed-size rule (reporting, not gating)\n");
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

    const size_t KiB = size_t(1) << 10;
    const size_t MiB = size_t(1) << 20;
    const size_t p64 = 64 * MiB;
    const std::vector<size_t> fifteen(15, p64);
    const std::vector<size_t> sixteen(16, p64);

    std::vector<Shape> shapes;
    {
        std::vector<size_t> p = fifteen;
        p.push_back(4 * KiB);
        shapes.push_back(make_shape("15 x 64 MiB + 4 KiB", std::move(p), 0, true));
    }
    {
        std::vector<size_t> p = fifteen;
        p.push_back(17829888);
        shapes.push_back(make_shape("15 x 64 MiB + 17829888 B", std::move(p), 0, true));
    }
    shapes.push_back(make_shape("16 x 64 MiB", sixteen, 0, false));
    shapes.push_back(make_shape("16 x 64 MiB, the 4 KiB reserved but not mapped", sixteen, 4 * KiB, false));
    shapes.push_back(make_shape("one piece of 16 x 64 MiB + 4 KiB", {16 * p64 + 4 * KiB}, 0, false));
    {
        std::vector<size_t> p;
        p.push_back(4 * KiB);
        p.insert(p.end(), fifteen.begin(), fifteen.end());
        shapes.push_back(make_shape("4 KiB first, then 15 x 64 MiB", std::move(p), 0, true));
    }
    shapes.push_back(make_shape("1024 MiB then 64 MiB", {1024 * MiB, p64}, 0, true));
    shapes.push_back(make_shape("64 MiB then 1024 MiB", {p64, 1024 * MiB}, 0, true));

    bool mixed_holds = true;
    for (const Shape& s : shapes) {
        size_t free_before = 0, total = 0;
        (void) hipMemGetInfo(&free_before, &total);
        size_t want = s.reserve_extra;
        for (size_t n : s.pieces) want += n;
        const size_t reserve = up_to(want, g);
        std::printf("\n  shape %s\n", s.label);
        std::printf("    %zu piece(s) in a %zu-byte reservation, free VRAM %zu MiB before\n", s.pieces.size(),
                    reserve, free_before >> 20);
        for (size_t i = 0; i < s.pieces.size(); ++i)
            std::printf("      piece %zu of %zu: %zu bytes\n", i + 1, s.pieces.size(), s.pieces[i]);
        if (s.reserve_extra) std::printf("      reserved but not mapped: %zu bytes\n", s.reserve_extra);

        void* base = nullptr;
        if (!report_call("hipMemAddressReserve", hipMemAddressReserve(&base, reserve, 0, nullptr, 0))) {
            if (s.mixed) mixed_holds = false;
            continue;
        }

        std::vector<hipMemGenericAllocationHandle_t> handle(s.pieces.size(), nullptr);
        std::vector<bool> created(s.pieces.size(), false), mapped(s.pieces.size(), false);
        bool shape_holds = true;
        size_t at_off = 0;
        for (size_t i = 0; i < s.pieces.size() && shape_holds; ++i) {
            const size_t n = s.pieces[i];
            void* at = static_cast<unsigned char*>(base) + at_off;
            char label[128];
            std::snprintf(label, sizeof label, "hipMemCreate piece %zu of %zu (%zu bytes)", i + 1,
                          s.pieces.size(), n);
            const hipError_t cr = hipMemCreate(&handle[i], n, &prop, 0);
            if (cr != hipSuccess) {
                report_call(label, cr);
                shape_holds = false;
                break;
            }
            created[i] = true;
            std::snprintf(label, sizeof label, "hipMemMap piece %zu of %zu (%zu bytes) at offset %zu", i + 1,
                          s.pieces.size(), n, at_off);
            const hipError_t mr = hipMemMap(at, n, 0, handle[i], 0);
            if (mr != hipSuccess) {
                report_call(label, mr);
                shape_holds = false;
                break;
            }
            mapped[i] = true;
            std::snprintf(label, sizeof label, "hipMemSetAccess piece %zu of %zu (%zu bytes)", i + 1,
                          s.pieces.size(), n);
            const hipError_t ar = hipMemSetAccess(at, n, &access, 1);
            if (ar != hipSuccess) {
                report_call(label, ar);
                shape_holds = false;
                break;
            }
            at_off += n;
        }
        if (shape_holds)
            report(s.label, true);
        else {
            std::printf("  %-58s %s\n", s.label, "DOES NOT HOLD");
            if (s.mixed) mixed_holds = false;
        }

        // Tear the shape down before the next one, whether it held or not.
        size_t off = 0;
        for (size_t i = 0; i < s.pieces.size(); ++i) {
            if (mapped[i]) (void) hipMemUnmap(static_cast<unsigned char*>(base) + off, s.pieces[i]);
            if (created[i]) (void) hipMemRelease(handle[i]);
            off += s.pieces[i];
        }
        (void) hipMemAddressFree(base, reserve);
        size_t free_after = 0;
        (void) hipMemGetInfo(&free_after, &total);
        std::printf("    reserved base %p, free VRAM %zu MiB after\n", base, free_after >> 20);
    }

    report("the driver's mixed-size rule on this stack", mixed_holds);
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
    phase3(device);

    std::printf("\n%d check(s) did not hold\n", not_held);
    (void) hipGetLastError();
    return 0;
}
