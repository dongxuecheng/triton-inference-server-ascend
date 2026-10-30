"""检查 ACL 是否能初始化所选设备；不加载模型，不执行推理。"""

import ctypes
import os


def main():
    acl = ctypes.CDLL("libascendcl.so")
    acl.aclInit.argtypes = [ctypes.c_char_p]
    acl.aclInit.restype = ctypes.c_int
    acl.aclrtGetDeviceCount.argtypes = [ctypes.POINTER(ctypes.c_uint32)]
    acl.aclrtGetDeviceCount.restype = ctypes.c_int
    acl.aclrtSetDevice.argtypes = [ctypes.c_int]
    acl.aclrtSetDevice.restype = ctypes.c_int
    acl.aclrtResetDevice.argtypes = [ctypes.c_int]
    acl.aclrtResetDevice.restype = ctypes.c_int
    acl.aclFinalize.argtypes = []
    acl.aclFinalize.restype = ctypes.c_int

    def check(name, status):
        if status != 0:
            raise RuntimeError(f"{name} failed: ACL status={status}")

    device = int(os.environ.get("ASCEND_DEVICE_ID", "2"))
    check("aclInit", acl.aclInit(None))
    selected = False
    try:
        count = ctypes.c_uint32()
        check("aclrtGetDeviceCount", acl.aclrtGetDeviceCount(ctypes.byref(count)))
        if not 0 <= device < count.value:
            raise RuntimeError(f"device {device} outside visible device count {count.value}")
        check("aclrtSetDevice", acl.aclrtSetDevice(device))
        selected = True
        print(f"ACL OK: visible_count={count.value}, logical_device={device}")
        print("This verifies initialization only; verify OM loading and inference separately.")
    finally:
        if selected:
            acl.aclrtResetDevice(device)
        acl.aclFinalize()


if __name__ == "__main__":
    main()
