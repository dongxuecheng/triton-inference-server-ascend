#include "ascend_canvas_pool.h"

#include <acl/acl.h>

#include <cstdlib>
#include <mutex>
#include <vector>

namespace {

struct Slot {
    void* pointer;
    size_t bytes;
    int32_t device_id;
    bool busy;
};

std::mutex& PoolMutex()
{
    static std::mutex mutex;
    return mutex;
}

std::vector<Slot>& PoolSlots()
{
    static std::vector<Slot> slots;
    return slots;
}

// Slot count is bounded so a leaked canvas (consumer never ran because a
// request failed between the two ensemble steps) degrades the pipeline instead
// of growing without limit.
size_t PoolCapacity()
{
    static const size_t capacity = [] {
        const char* value = std::getenv("ASCEND_CANVAS_SLOTS");
        if (value == nullptr || value[0] == '\0') {
            return size_t{8};
        }
        char* end = nullptr;
        const long parsed = std::strtol(value, &end, 10);
        if (end == value || *end != '\0' || parsed <= 0) {
            return size_t{8};
        }
        return static_cast<size_t>(parsed);
    }();
    return capacity;
}

}  // namespace

extern "C" void* AscendCanvasAcquire(int32_t device_id, size_t bytes)
{
    std::lock_guard<std::mutex> lock(PoolMutex());
    std::vector<Slot>& slots = PoolSlots();

    for (Slot& slot : slots) {
        if (slot.device_id == device_id && slot.bytes >= bytes && !slot.busy) {
            slot.busy = true;
            return slot.pointer;
        }
    }
    if (slots.size() >= PoolCapacity()) {
        return nullptr;
    }

    void* pointer = nullptr;
    if (aclrtMalloc(&pointer, bytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
        return nullptr;
    }
    slots.push_back(Slot{pointer, bytes, device_id, true});
    return pointer;
}

extern "C" void AscendCanvasRelease(void* canvas)
{
    if (canvas == nullptr) {
        return;
    }
    std::lock_guard<std::mutex> lock(PoolMutex());
    for (Slot& slot : PoolSlots()) {
        if (slot.pointer == canvas) {
            slot.busy = false;
            return;
        }
    }
}

extern "C" size_t AscendCanvasPoolSize()
{
    std::lock_guard<std::mutex> lock(PoolMutex());
    return PoolSlots().size();
}
