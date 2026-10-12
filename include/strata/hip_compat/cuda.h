#pragma once
// cuda.h for HIP builds: the CUDA *driver* API names src/poly/engine.cpp and engine.hpp use, as inline adapters
// over HIP's virtual-memory and device calls.  A HIP build reaches this file because strata_hip_runtime puts
// include/strata/hip_compat/ BEFORE on the include path (cmake/hip_backend.cmake); a CUDA build never sees this
// directory, so it keeps NVIDIA's own <cuda.h>.
//
// HIP's numeric enumerators are not assumed to equal CUDA's: every enumeration below maps a CUDA name to its HIP
// name, so the compiler substitutes HIP's own value.  HIP's hipMemGenericAllocationHandle_t is a pointer while the
// engine stores handles as unsigned long long (Engine::Mapped::pieces), so the handle adapters cast through
// uintptr_t, and addresses are HIP's void*.
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

using CUresult = hipError_t;
#define CUDA_SUCCESS hipSuccess

using CUdevice = int;
using CUdeviceptr = unsigned long long;
using CUmemGenericAllocationHandle = unsigned long long;

using CUmemAllocationProp = hipMemAllocationProp;
using CUmemAccessDesc = hipMemAccessDesc;

#define CU_MEM_ALLOCATION_TYPE_PINNED hipMemAllocationTypePinned
#define CU_MEM_LOCATION_TYPE_DEVICE hipMemLocationTypeDevice
#define CU_MEM_ACCESS_FLAGS_PROT_READWRITE hipMemAccessFlagsProtReadWrite
#define CU_MEM_ALLOC_GRANULARITY_MINIMUM hipMemAllocationGranularityMinimum
#define CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED hipDeviceAttributeVirtualMemoryManagementSupported

// Writes HIP's message through the out-parameter the CUDA signature has; the caller only reads the string.
inline CUresult cuGetErrorString(CUresult error, const char** pStr) {
    *pStr = hipGetErrorString(error);
    return CUDA_SUCCESS;
}

inline CUresult cuDeviceGet(CUdevice* device, int ordinal) { return hipDeviceGet(device, ordinal); }

inline CUresult cuDeviceGetAttribute(int* pi, int attrib, CUdevice device) {
    return hipDeviceGetAttribute(pi, (hipDeviceAttribute_t) attrib, device);
}

// Return values pass through unchanged, so every "== CUDA_SUCCESS" test keeps its meaning.
inline CUresult cuMemAddressReserve(CUdeviceptr* ptr, size_t size, size_t alignment, CUdeviceptr addr,
                                    unsigned long long flags) {
    void* p = nullptr;
    const hipError_t e = hipMemAddressReserve(&p, size, alignment, (void*) (uintptr_t) addr, flags);
    if (e == hipSuccess) *ptr = (CUdeviceptr) (uintptr_t) p;
    return e;
}

inline CUresult cuMemAddressFree(CUdeviceptr ptr, size_t size) {
    return hipMemAddressFree((void*) (uintptr_t) ptr, size);
}

inline CUresult cuMemCreate(CUmemGenericAllocationHandle* handle, size_t size, const CUmemAllocationProp* prop,
                            unsigned long long flags) {
    hipMemGenericAllocationHandle_t h = nullptr;
    const hipError_t e = hipMemCreate(&h, size, prop, flags);
    if (e == hipSuccess) *handle = (CUmemGenericAllocationHandle) (uintptr_t) h;
    return e;
}

inline CUresult cuMemRelease(CUmemGenericAllocationHandle handle) {
    return hipMemRelease((hipMemGenericAllocationHandle_t) (uintptr_t) handle);
}

inline CUresult cuMemMap(CUdeviceptr ptr, size_t size, size_t offset, CUmemGenericAllocationHandle handle,
                         unsigned long long flags) {
    return hipMemMap((void*) (uintptr_t) ptr, size, offset, (hipMemGenericAllocationHandle_t) (uintptr_t) handle,
                     flags);
}

inline CUresult cuMemUnmap(CUdeviceptr ptr, size_t size) { return hipMemUnmap((void*) (uintptr_t) ptr, size); }

inline CUresult cuMemSetAccess(CUdeviceptr ptr, size_t size, const CUmemAccessDesc* desc, size_t count) {
    return hipMemSetAccess((void*) (uintptr_t) ptr, size, desc, count);
}

inline CUresult cuMemGetAllocationGranularity(size_t* granularity, const CUmemAllocationProp* prop,
                                              hipMemAllocationGranularity_flags option) {
    return hipMemGetAllocationGranularity(granularity, prop, option);
}
