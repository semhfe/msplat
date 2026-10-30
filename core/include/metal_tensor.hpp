#ifndef METAL_TENSOR_H
#define METAL_TENSOR_H

#include <vector>
#include <cstdint>
#include <cassert>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>

#include <CoreFoundation/CoreFoundation.h>

// Forward-declare the Metal buffer type for C++ compatibility.
// Full Metal/Metal.h is only needed in .mm files.
#ifdef __OBJC__
#import <Metal/Metal.h>
#else
typedef void* MTLBufferRef;  // opaque handle in pure C++
#endif

enum class DType : uint8_t {
    Float32,
    Int32,
    Int64,
    UInt8,
    Float64,
};

inline size_t dtypeSize(DType dt) {
    switch (dt) {
        case DType::Float32: return 4;
        case DType::Int32:   return 4;
        case DType::Int64:   return 8;
        case DType::UInt8:   return 1;
        case DType::Float64: return 8;
    }
}

// Installed by the engine once its Metal context exists (msplat_metal.mm): releases an owned
// MTLBuffer only after GPU work that may still use it has completed. nullptr = release now.
inline void (*g_mtensor_release_hook)(void *buffer) noexcept = nullptr;  // must not throw

// Lightweight GPU tensor — wraps an MTLBuffer with shape metadata.
class MTensor {
public:
    MTensor() = default;

#ifdef __OBJC__
    // GPU allocation (Objective-C++ only)
    MTensor(id<MTLDevice> device, std::vector<int64_t> shape, DType dtype)
        : _shape(std::move(shape)), _dtype(dtype) {
        _numel = 1;
        for (auto s : _shape) _numel *= s;
        size_t bytes = _numel * dtypeSize(_dtype);
        if (bytes == 0) bytes = 4;
        id<MTLBuffer> buf = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        if (!buf) {
            // newBufferWithLength returns nil instead of raising; a nil buffer reaches a
            // compute encoder and faults the kernel. Fail here with a readable message;
            // "out of memory" is one of the app's OOM markers (RunSafety.swift oomMarkers).
            throw std::runtime_error(
                "msplat: out of memory: GPU buffer allocation failed for " + std::to_string(bytes) +
                " bytes (device already holds " +
                std::to_string((unsigned long long)[device currentAllocatedSize]) +
                " bytes); lower --cap-max or the image resolution");
        }
        _buffer = (__bridge_retained void*)buf;
        _ownsBuffer = true;
        _data = [buf contents];  // cache CPU-accessible pointer for C++ access
    }

    id<MTLBuffer> buffer() const { return (__bridge id<MTLBuffer>)_buffer; }
#endif

    // CPU allocation (no Metal buffer)
    MTensor(std::vector<int64_t> shape, DType dtype)
        : _shape(std::move(shape)), _dtype(dtype) {
        _numel = 1;
        for (auto s : _shape) _numel *= s;
        size_t bytes = _numel * dtypeSize(_dtype);
        _cpu_data.resize(bytes);
    }

    // A4 (upstream v1.1.4): RAII. An owning MTensor holds one retain on its MTLBuffer;
    // copies retain, destruction/reassignment releases, moves transfer. Views stay
    // non-owning through copies. Before this, copies were shallow and nothing released,
    // and reset() released only in Objective-C++ TUs (an ODR violation).
    ~MTensor() { releaseBuffer(); }

    MTensor(const MTensor &o)
        : _buffer(o._buffer), _data(o._data), _cpu_data(o._cpu_data),
          _shape(o._shape), _dtype(o._dtype), _numel(o._numel),
          _ownsBuffer(o._ownsBuffer) {
        if (_buffer && _ownsBuffer) CFRetain(_buffer);
    }

    MTensor(MTensor &&o) noexcept
        : _buffer(o._buffer), _data(o._data), _cpu_data(std::move(o._cpu_data)),
          _shape(std::move(o._shape)), _dtype(o._dtype), _numel(o._numel),
          _ownsBuffer(o._ownsBuffer) {
        o._buffer = nullptr; o._data = nullptr; o._numel = 0; o._ownsBuffer = false;
    }

    MTensor& operator=(const MTensor &o) {
        if (this == &o) return *this;
        void *incoming = o._buffer;
        const bool incomingOwns = o._ownsBuffer;
        if (incoming && incomingOwns) CFRetain(incoming);  // retain before release: o may share our buffer
        releaseBuffer();
        _buffer = incoming; _ownsBuffer = incomingOwns;
        _data = o._data; _cpu_data = o._cpu_data;
        _shape = o._shape; _dtype = o._dtype; _numel = o._numel;
        return *this;
    }

    MTensor& operator=(MTensor &&o) noexcept {
        if (this == &o) return *this;
        releaseBuffer();
        _buffer = o._buffer; _ownsBuffer = o._ownsBuffer;
        _data = o._data; _cpu_data = std::move(o._cpu_data);
        _shape = std::move(o._shape); _dtype = o._dtype; _numel = o._numel;
        o._buffer = nullptr; o._data = nullptr; o._numel = 0; o._ownsBuffer = false;
        return *this;
    }

    bool defined() const { return _buffer != nullptr || !_cpu_data.empty(); }
    bool isGpu() const { return _buffer != nullptr; }

    int64_t numel() const { return _numel; }
    int64_t size(int dim) const {
        if (dim < 0) dim += _shape.size();
        return _shape[dim];
    }
    int ndim() const { return (int)_shape.size(); }
    const std::vector<int64_t>& shape() const { return _shape; }
    DType dtype() const { return _dtype; }
    size_t elementSize() const { return dtypeSize(_dtype); }
    size_t nbytes() const { return _numel * dtypeSize(_dtype); }

    void* data_ptr() {
        if (_data) return _data;
        return _cpu_data.data();
    }
    const void* data_ptr() const {
        if (_data) return _data;
        return _cpu_data.data();
    }

    template<typename T> T* data() { return static_cast<T*>(data_ptr()); }
    template<typename T> const T* data() const { return static_cast<const T*>(data_ptr()); }

    void zero() {
        memset(data_ptr(), 0, _numel * dtypeSize(_dtype));
    }

    // Create a CPU copy of the data
    MTensor cpu() const {
        MTensor out(_shape, _dtype);
        memcpy(out.data_ptr(), data_ptr(), nbytes());
        return out;
    }

    void reset() {
        releaseBuffer();
        _data = nullptr;
        _cpu_data.clear();
        _shape.clear();
        _numel = 0;
    }

    // Stride for dim 0 (elements per row)
    int64_t stride0() const {
        if (_shape.size() <= 1) return 1;
        int64_t s = 1;
        for (size_t i = 1; i < _shape.size(); i++) s *= _shape[i];
        return s;
    }

    // Create a view of the first `n` elements along dim 0.
    // WARNING: Non-owning — shares the underlying MTLBuffer without retaining it
    // (and stays non-owning through copies).
    // The caller MUST ensure the parent MTensor outlives all views.
    // Use-after-free if the parent is destroyed while a view exists.
    MTensor view(int64_t n) const {
        MTensor v;
        v._buffer = _buffer;  // shares the buffer (non-owning)
        v._data = _data;      // shares the CPU-accessible pointer
        v._shape = _shape;
        v._shape[0] = n;
        v._dtype = _dtype;
        v._numel = n * stride0();
        v._ownsBuffer = false;
        return v;
    }

private:
    void releaseBuffer() {
        if (_buffer && _ownsBuffer) {
            if (g_mtensor_release_hook) g_mtensor_release_hook(_buffer);
            else CFRelease(_buffer);
        }
        _buffer = nullptr;
        _ownsBuffer = false;
    }

    void* _buffer = nullptr;  // id<MTLBuffer> as void*, retained when _ownsBuffer
    void* _data = nullptr;    // cached CPU-accessible pointer (shared memory on Apple Silicon)
    std::vector<uint8_t> _cpu_data;
    std::vector<int64_t> _shape;
    DType _dtype = DType::Float32;
    int64_t _numel = 0;
    bool _ownsBuffer = false;
};

// Factory helpers (Objective-C++ only)
#ifdef __OBJC__
inline MTensor mtensor_empty(id<MTLDevice> dev, std::vector<int64_t> shape, DType dt) {
    return MTensor(dev, std::move(shape), dt);
}

inline MTensor mtensor_zeros(id<MTLDevice> dev, std::vector<int64_t> shape, DType dt) {
    MTensor t(dev, std::move(shape), dt);
    t.zero();
    return t;
}
#endif

#endif
