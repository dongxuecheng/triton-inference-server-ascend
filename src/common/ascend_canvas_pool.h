// Shared device-resident canvas pool for the split YOLO11 backends.
//
// Triton's memory manager cannot hand out NPU memory on this platform (its
// device allocator is CUDA-based), so the ensemble would otherwise copy the
// NV12 canvas to the host and straight back to the device. Both backends link
// this single shared library, which owns a small pool of device buffers that
// the preprocess backend fills and the inference backend runs from.
//
// Ownership is a simple busy flag: the producer acquires a canvas and hands its
// device address to the consumer through a Triton tensor, and the consumer
// releases it once the model has finished reading it. Because the shared
// library carries a SONAME and is therefore loaded once per process even when a
// copy sits inside each model's version directory, both backends see one pool.

#ifndef TRITON_ASCEND_CANVAS_POOL_H
#define TRITON_ASCEND_CANVAS_POOL_H

#include <cstddef>
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

// Acquire a device-resident canvas of at least `bytes` on `device_id`.
//
// Returns the device address, or nullptr when every slot is busy or the device
// allocation failed. The caller must have selected the device already.
void* AscendCanvasAcquire(int32_t device_id, size_t bytes);

// Release a canvas obtained from AscendCanvasAcquire so it can be reused.
// Passing nullptr is a no-op.
void AscendCanvasRelease(void* canvas);

// Number of slots currently held by the pool, including busy ones.
size_t AscendCanvasPoolSize(void);

#ifdef __cplusplus
}
#endif

#endif  // TRITON_ASCEND_CANVAS_POOL_H
