"""内置 CANN 的部署回归测试，不依赖 NPU、镜像或 Docker daemon。"""

import contextlib
import importlib.util
import io
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch

import yaml

ROOT = Path(__file__).resolve().parents[1]
BASH = os.environ.get("TEST_BASH") or shutil.which("bash")
CMAKE = os.environ.get("TEST_CMAKE") or shutil.which("cmake")


class DeploymentConfigTests(unittest.TestCase):
    def test_compose_uses_legacy_schema_and_no_host_toolkit(self):
        config = yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
        self.assertEqual(config["version"], "2.4")
        service = config["services"]["triton-ascend"]
        self.assertEqual(service["environment"]["ASCEND_DEVICE_ID"], "${ASCEND_DEVICE_ID:-2}")
        self.assertEqual(service["build"]["dockerfile"], "Dockerfile")
        for volume in service["volumes"]:
            self.assertIsInstance(volume, str)
            self.assertNotIn("TOOLKIT", volume)
            self.assertNotIn(":/usr/local/Ascend:", volume)
            self.assertNotIn(":/models:", volume)

    def test_selected_chain_is_complete_and_no_model_pins_device_zero(self):
        models = [line.strip() for line in (ROOT / "workspace/models_to_load.txt").read_text(encoding="utf-8").splitlines()
                  if line.strip() and not line.startswith("#")]
        self.assertEqual(set(models), {"CUSTOM_LABELS", "YUV_640_LETTERBOX_PREPROCESS", "YOLO26_DET_PRE",
                                      "YOLO26_DET_PRE_POSTPROCESS", "YOLO26_DET_PRE_YUV_ENSEMBLE"})
        for model in models:
            self.assertTrue((ROOT / "workspace/models" / model / "config.pbtxt").is_file())
        for path in (ROOT / "workspace/models").glob("*/config.pbtxt"):
            self.assertNotIn('key: "device_id"', path.read_text(encoding="utf-8"), str(path))


@unittest.skipUnless(BASH, "Bash required")
class EntrypointTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cann = self.root / "cann"
        self.cann.mkdir()
        (self.cann / "set_env.sh").write_text('export FROM_CANN_ENV=yes\n', encoding="utf-8")
        self.env = dict(os.environ, ASCEND_HOME_PATH=self.cann.as_posix(), ASCEND_DEVICE_ID="2")
        self.env.pop("BASH_ENV", None)

    def run_entry(self, *args, **env):
        return subprocess.run([BASH, str(ROOT / "docker/entrypoint.sh"), *args],
                              env=dict(self.env, **env), capture_output=True, text=True)

    def test_scripts_have_valid_bash_syntax(self):
        for path in [ROOT / "build.sh", *ROOT.glob("scripts/*.sh"), *ROOT.glob("docker/*.sh")]:
            result = subprocess.run([BASH, "-n", str(path)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_debug_command_loads_cann_env_and_preserves_device(self):
        result = self.run_entry(BASH, "-c", 'printf "%s:%s" "$FROM_CANN_ENV" "$ASCEND_DEVICE_ID"')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "yes:2")

    def test_missing_toolkit_fails_before_executing_command(self):
        result = self.run_entry("echo", "must-not-execute", ASCEND_HOME_PATH=(self.root / "absent").as_posix())
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("CANN directory not found", result.stderr)
        self.assertNotIn("must-not-execute", result.stdout)

    def test_missing_env_script_fails_clearly(self):
        (self.cann / "set_env.sh").unlink()
        result = self.run_entry("echo", "must-not-execute")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("set_env.sh not found", result.stderr)

    def test_missing_empty_or_directory_model_is_rejected(self):
        empty = self.root / "empty.om"
        empty.touch()
        for path in [self.root / "missing.om", empty, self.cann]:
            with self.subTest(path=path):
                result = self.run_entry(YOLO26_MODEL_FILE=path.as_posix())
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("YOLO26 .om is missing or empty", result.stderr)

    def test_cann_env_keeps_existing_library_path(self):
        result = self.run_entry(BASH, "-c", 'printf "%s" "$LD_LIBRARY_PATH"', LD_LIBRARY_PATH="/existing/lib")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("/existing/lib", result.stdout)

    def test_legacy_fwkacllib_is_available_at_runtime(self):
        (self.cann / "fwkacllib/lib64").mkdir(parents=True)
        result = self.run_entry(BASH, "-c", 'printf "%s" "$LD_LIBRARY_PATH"')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("/fwkacllib/lib64", result.stdout)

    def fake_driver_environment(self, injected_path=""):
        host = self.root / "Ascend"
        for directory in ["driver/lib64/driver", "driver/lib64/common", "driver/lib64", "develop/lib64"]:
            (host / directory).mkdir(parents=True, exist_ok=True)
        for directory in ["lib64", "runtime/lib64", "fwkacllib/lib64"]:
            (self.cann / directory).mkdir(parents=True, exist_ok=True)
        # 用临时目录模拟宿主机，不需要在测试机写 /usr/local/Ascend。
        # Bash 的 pwd 也将 Windows 路径转成无盘符冒号的 POSIX 路径，方便解析库列表。
        host_posix = subprocess.run([BASH, "-c", "pwd -P"], cwd=host,
                                    capture_output=True, text=True, check=True).stdout.strip()
        scripts = self.root / "scripts"
        scripts.mkdir()
        shutil.copyfile(ROOT / "docker/entrypoint.sh", scripts / "entrypoint.sh")
        source = (ROOT / "docker/cann-env.sh").read_text(encoding="utf-8")
        (scripts / "cann-env.sh").write_text(source.replace("/usr/local/Ascend", host_posix), encoding="utf-8")
        if injected_path:
            # 模拟 set_env.sh 或镜像原有环境将 develop 放到最前面的情况。
            (self.cann / "set_env.sh").write_text(
                f'export LD_LIBRARY_PATH="{host_posix}/{injected_path}:/inherited/lib"\n', encoding="utf-8")
        result = subprocess.run(
            [BASH, str(scripts / "entrypoint.sh"), BASH, "-c", 'printf "%s" "$LD_LIBRARY_PATH"'],
            env=dict(self.env, ASCEND_HOME_PATH=f"{host_posix}/../cann", LD_LIBRARY_PATH="/inherited/lib"),
            capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        return host_posix, result.stdout.split(":")

    def test_real_driver_paths_are_first_in_exact_priority_order(self):
        host, paths = self.fake_driver_environment()
        expected = [f"{host}/driver/lib64/driver", f"{host}/driver/lib64/common", f"{host}/driver/lib64"]
        self.assertEqual(paths[:3], expected)
        self.assertTrue(paths[3].endswith("/cann/lib64"))
        self.assertTrue(paths[4].endswith("/cann/runtime/lib64"))
        self.assertTrue(paths[5].endswith("/cann/fwkacllib/lib64"))
        self.assertEqual(paths[-1], "/inherited/lib")

    def test_existing_develop_path_cannot_override_real_driver(self):
        host, paths = self.fake_driver_environment("develop/lib64")
        self.assertEqual(paths[0], f"{host}/driver/lib64/driver")
        self.assertGreater(paths.index(f"{host}/develop/lib64"), paths.index(f"{host}/driver/lib64"))
        self.assertEqual(paths[-1], "/inherited/lib")

    def test_develop_and_stub_directories_are_not_automatically_added(self):
        host, paths = self.fake_driver_environment()
        self.assertNotIn(f"{host}/develop/lib64", paths)
        self.assertFalse(any("/devlib" in path or "/stub" in path for path in paths))


@unittest.skipUnless(BASH, "Bash required")
class BuildPreflightTests(unittest.TestCase):
    def probe(self, header=None):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            if header is not None:
                path = root / "cann" / header
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            env = dict(os.environ, ASCEND_HOME_PATH=(root / "cann").as_posix(),
                       TRITON_DIR=(root / "missing-triton").as_posix())
            # 使用与脚本相同的 uname，避免在 Windows Git Bash 上假定架构。
            return subprocess.run([BASH, str(ROOT / "build.sh")], env=env,
                                  capture_output=True, text=True)

    def test_both_acl_header_layouts_pass_to_next_validation(self):
        arch = subprocess.run([BASH, "-c", "uname -m"], capture_output=True,
                              text=True, check=True).stdout.strip()
        for header in ["include/acl/acl.h", f"{arch}-linux/include/acl/acl.h"]:
            with self.subTest(header=header):
                result = self.probe(header)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("ACL include:", result.stdout)
                self.assertIn("tritonbackend.h not found", result.stderr)
                self.assertNotIn("acl/acl.h not found", result.stderr)

    def test_missing_header_lists_checked_paths(self):
        result = self.probe()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("acl/acl.h not found", result.stderr)
        self.assertIn("/include/acl/acl.h", result.stderr)
        self.assertIn("-linux/include/acl/acl.h", result.stderr)


@unittest.skipUnless(CMAKE, "CMake required")
class ToolkitDiscoveryTests(unittest.TestCase):
    def probe(self, files, mode="acl", override=None):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for file in files:
                path = root / file
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("", encoding="utf-8")
            script = root / "probe.cmake"
            script.write_text(
                'set(CMAKE_SYSTEM_PROCESSOR aarch64)\n'
                'set(CMAKE_FIND_LIBRARY_PREFIXES lib)\nset(CMAKE_FIND_LIBRARY_SUFFIXES .so)\n'
                f'include("{(ROOT / "cmake/AscendToolkit.cmake").as_posix()}")\n'
                + (f'set(ASCENDC_CMAKE_DIR "{(root / override).as_posix()}")\n' if override else "")
                + f'triton_find_{mode}("{root.as_posix()}")\n'
                + 'message(STATUS "FOUND=${ASCEND_RUNTIME_LIBRARY_DIR};${ASCENDC_CMAKE_DIR}")\n',
                encoding="utf-8",
            )
            return subprocess.run([CMAKE, "-P", str(script)], capture_output=True, text=True)

    def test_acl_old_and_new_layouts(self):
        for directory in ["runtime/lib64", "lib64", "aarch64-linux/lib64", "fwkacllib/lib64"]:
            with self.subTest(directory=directory):
                result = self.probe(["include/acl/acl.h", f"{directory}/libascendcl.so"])
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn(directory, result.stdout)

    def test_arch_specific_acl_headers(self):
        result = self.probe(["aarch64-linux/include/acl/acl.h", "aarch64-linux/lib64/libascendcl.so"])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("aarch64-linux/include", result.stdout)

    def test_stub_only_install_is_rejected(self):
        result = self.probe(["include/acl/acl.h", "devlib/libascendcl.so"])
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("ASCENDCL_LIBRARY", result.stderr)

    def test_ascendc_standard_and_relocated_layouts(self):
        for directory in ["tools/tikcpp/ascendc_kernel_cmake", "compiler/tikcpp/ascendc_kernel_cmake",
                          "aarch64-linux/tikcpp/ascendc_kernel_cmake", "share/new-package/cmake"]:
            with self.subTest(directory=directory):
                result = self.probe([f"{directory}/ascendc.cmake"], "ascendc")
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn(directory, result.stdout)

    def test_arch_specific_ascendc_is_preferred_over_unrelated_fallback(self):
        directory = "aarch64-linux/tikcpp/ascendc_kernel_cmake"
        result = self.probe([f"{directory}/ascendc.cmake", "share/other/ascendc.cmake"], "ascendc")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(directory, result.stdout)

    def test_missing_or_ambiguous_ascendc_fails(self):
        for files in [[], ["share/one/ascendc.cmake", "share/two/ascendc.cmake"]]:
            result = self.probe(files, "ascendc")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Cannot uniquely locate", result.stderr)

    def test_explicit_ascendc_override(self):
        files = ["share/one/ascendc.cmake", "share/two/ascendc.cmake"]
        result = self.probe(files, "ascendc", "share/two")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("share/two", result.stdout)
        self.assertNotEqual(self.probe(files, "ascendc", "absent").returncode, 0)


class DevicePreflightTests(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location("device_check", ROOT / "docker/check-device.py")
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)
        self.acl = Mock()
        for name in ["aclInit", "aclrtGetDeviceCount", "aclrtSetDevice", "aclrtResetDevice", "aclFinalize"]:
            getattr(self.acl, name).return_value = 0

        def count(pointer):
            pointer._obj.value = 4
            return 0

        self.acl.aclrtGetDeviceCount.side_effect = count

    def test_selected_device_and_cleanup(self):
        with patch.object(self.module.ctypes, "CDLL", return_value=self.acl), \
                patch.dict(os.environ, {"ASCEND_DEVICE_ID": "2"}), contextlib.redirect_stdout(io.StringIO()):
            self.module.main()
        self.acl.aclrtSetDevice.assert_called_once_with(2)
        self.acl.aclrtResetDevice.assert_called_once_with(2)
        self.acl.aclFinalize.assert_called_once()

    def test_out_of_range_device_fails_without_select_or_reset(self):
        with patch.object(self.module.ctypes, "CDLL", return_value=self.acl), \
                patch.dict(os.environ, {"ASCEND_DEVICE_ID": "5"}), self.assertRaisesRegex(RuntimeError, "outside visible"):
            self.module.main()
        self.acl.aclrtSetDevice.assert_not_called()
        self.acl.aclrtResetDevice.assert_not_called()
        self.acl.aclFinalize.assert_called_once()

    def test_acl_selection_failure_is_not_reported_as_success(self):
        self.acl.aclrtSetDevice.return_value = 507033
        with patch.object(self.module.ctypes, "CDLL", return_value=self.acl), \
                patch.dict(os.environ, {"ASCEND_DEVICE_ID": "2"}), self.assertRaisesRegex(RuntimeError, "507033"):
            self.module.main()
        self.acl.aclrtResetDevice.assert_not_called()
        self.acl.aclFinalize.assert_called_once()


if __name__ == "__main__":
    unittest.main()
