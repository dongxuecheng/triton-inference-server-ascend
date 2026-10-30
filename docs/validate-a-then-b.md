# 从 A 迁移到 B（麒麟）

统一入口为 `docker-compose.yml`、`Dockerfile`、`.env.example`。CANN 编译工具和运行库都在镜像内；宿主机仍提供 Ascend 驱动、固件和设备。

A 已在 CANN 9.0.0 镜像内完成 YOLO26 NV12 真实推理，逻辑设备为 1，1920×1080 图片返回 9 个 person、客户端单次推理计时 16.4 ms。此次整理部署入口没有修改 C++、Ascend C、CANN 环境或启动脚本，可以直接迁移当前运行且已验证的镜像。

## 1. A：更新部署文件并导出已验证镜像

在 A 的 Triton 仓库、同一个 Bash 终端执行：

```bash
cd ~/triton-inference-server-ascend
git pull --ff-only

set -a
source .env.cann.a
set +a

# 更新当前终端旧的 dc 函数；项目名、服务名、容器名保持原值。
dc() { docker compose -p triton-cann-a-test -f docker-compose.yml "$@"; }
dc config

test -f "$YOLO26_MODEL_HOST_PATH" && test -s "$YOLO26_MODEL_HOST_PATH"
docker inspect --format '{{.State.Status}} {{.Image}}' "$TRITON_CANN_CONTAINER_NAME"
```

已有 A 环境文件继续使用，不必复制新示例覆盖它；确认 `ASCEND_DEVICE_ID=1`、容器名及端口与当前成功服务一致。B 应使用自己的配置。

从运行容器获取镜像 ID 再打迁移标签，确保导出的是刚才实际推理成功的镜像：

```bash
verified_image=$(docker inspect --format '{{.Image}}' "$TRITON_CANN_CONTAINER_NAME")
docker image inspect "$verified_image" --format '{{.Id}} {{.Os}}/{{.Architecture}}'
docker tag "$verified_image" triton-ascend-cann:24.10-cann9.0.0-a-verified

transfer_dir="$PWD/../triton-ascend-transfer"
mkdir -p "$transfer_dir"
docker save -o "$transfer_dir/triton-ascend-cann9.docker.tar" \
  triton-ascend-cann:24.10-cann9.0.0-a-verified
git archive --format=tar.gz -o "$transfer_dir/source.tar.gz" HEAD
cp "$YOLO26_MODEL_HOST_PATH" "$transfer_dir/yolo26s.om"
docker image inspect "$verified_image" --format '{{.Id}} {{.Os}}/{{.Architecture}}' \
  > "$transfer_dir/image-id.txt"

cd "$transfer_dir"
sha256sum triton-ascend-cann9.docker.tar source.tar.gz yolo26s.om image-id.txt > SHA256SUMS
```

`image-id.txt` 应为 `linux/arm64`。包内包含最终运行镜像、源码、OM 和校验文件；无需另带 CANN 基础镜像、宿主机 Toolkit 或 A 的虚拟环境。源码包不含 `.git` 和本地 `.env`，可用于离线部署；后续也可从自己的 fork 克隆源码来管理更新。

将整个 `triton-ascend-transfer` 目录传到 B，例如 `/data/triton-ascend-transfer`。示例 SSH 地址需替换：

```bash
scp -r "$transfer_dir" root@B机器IP:/data/
```

导出操作不需要停止 A 上的 Triton、SAM3 或拉流服务。以后若修改了后端或镜像内容，先在 A 用 `dc build triton-ascend`、`dc up -d --no-deps --force-recreate triton-ascend` 更新并完成实际推理，再导出新的运行镜像。

## 2. B：校验并导入

B 使用旧命令 `docker-compose`。以下在 B 的 Bash 终端执行：

```bash
cd /data/triton-ascend-transfer
sha256sum -c SHA256SUMS
docker load -i triton-ascend-cann9.docker.tar
cat image-id.txt
docker image inspect triton-ascend-cann:24.10-cann9.0.0-a-verified \
  --format '{{.Id}} {{.Os}}/{{.Architecture}}'

mkdir -p /data/triton-inference-server-ascend
tar -xzf source.tar.gz -C /data/triton-inference-server-ascend
cd /data/triton-inference-server-ascend
mkdir -p workspace/models/YOLO26_DET_PRE/1
cp /data/triton-ascend-transfer/yolo26s.om workspace/models/YOLO26_DET_PRE/1/yolo26s.om
cp -n .env.example .env
```

校验应全部通过，导入后的镜像 ID 和架构应与 A 的 `image-id.txt` 一致。源码解压目录用于本次部署，若已有其他 Triton 项目，请换一个新目录。

编辑 B 的 `.env`，至少设置以下值（其余保持示例）：

```dotenv
TRITON_CANN_IMAGE=triton-ascend-cann:24.10-cann9.0.0-a-verified
TRITON_CANN_CONTAINER_NAME=triton-ascend-cann9-b
YOLO26_MODEL_HOST_PATH=/data/triton-inference-server-ascend/workspace/models/YOLO26_DET_PRE/1/yolo26s.om
ASCEND_DEVICE_ID=2
TRITON_HTTP_PORT=54245
TRITON_GRPC_PORT=54246
TRITON_METRICS_PORT=54247
```

B 此前映射为 NPU 5 / Chip 0 → ACL 逻辑设备 2；启动前核对映射和占用。若这些宿主机端口已占用，更换端口。此配置不会挂载宿主机 Toolkit。

## 3. B：预检、启动和就绪检查

```bash
set -a
source .env
set +a

test -f "$YOLO26_MODEL_HOST_PATH" && test -s "$YOLO26_MODEL_HOST_PATH"
test -d /usr/local/Ascend/driver
test -d /usr/local/Ascend/develop
test -f /etc/ascend_install.info
npu-smi info -m
npu-smi info

dc() { docker-compose -p triton-cann-b -f docker-compose.yml "$@"; }
dc config
dc run --rm --no-deps triton-ascend \
  bash -c 'echo "$ASCEND_HOME_PATH"; ldd /models/YOLO26_DET_PRE/1/libtriton_yolo26_infer.so'
dc run --rm --no-deps triton-ascend python3 /opt/triton-ascend/check-device.py
```

`ldd` 应无 `not found`，设备检查应返回 `ACL OK` 且 `logical_device=2`。预检通过后启动：

```bash
dc up -d --no-build --no-deps triton-ascend
dc logs --tail=200 triton-ascend

curl -sS -f -o /dev/null -w 'server HTTP %{http_code}\n' \
  "http://127.0.0.1:${TRITON_HTTP_PORT}/v2/health/ready"
curl -sS -f -o /dev/null -w 'model HTTP %{http_code}\n' \
  "http://127.0.0.1:${TRITON_HTTP_PORT}/v2/models/YOLO26_DET_PRE_YUV_ENSEMBLE/ready"
```

B 使用已导入的最终镜像，不执行构建；两个接口应为 200，五个模型应全部 READY。宿主机无需安装 CANN、ATC 或编译工具。

## 4. 验证 B 的实际推理后接入业务

可以直接在 A 已安装依赖的 Python 环境中请求 B，避免在 B 再装客户端。将下例地址替换为 B 的实际 IP 和 HTTP 端口，图片使用 A 上同一张测试图：

```bash
cd ~/triton-inference-server-ascend
source .venv-triton-client/bin/activate
python workspace/triton_client/visualize.py \
  --protocol http --url B机器IP:54245 \
  --model YOLO26_DET_PRE_YUV_ENSEMBLE \
  --image /home/dxc/ascend-sam3/test-images/02fa03f0-3e49-4e21-85a1-8b432d74a922.jpg \
  --output /tmp/yolo26_result_b.jpg
```

若在 B 本地测试，按 README 的客户端安装步骤建立环境，并准备同一张图片。比对检测数量、类别、检测框，并在 B 用 `npu-smi info` 检查 tritonserver 出现在预期芯片。A 的成功不能替代 B 的驱动、固件、内核和旧 Docker 上的实际验证。

验证后将 AIDetection 的 `TRITON_YOLO_URL` / `TRITON_YOLO_GRPC_URL` 分别配置为 B 的 HTTP / gRPC 地址；模型名保持 `YOLO26_DET_PRE_YUV_ENSEMBLE`，原 NV12 请求协议可继续使用。
