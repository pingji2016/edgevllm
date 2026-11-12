# 3D 重建（3D Gaussian Splatting）使用说明

> 主机 `pingji-MS-7B89` · RTX 5060 Ti 16 GB · Ubuntu 26.04
> 写于 2026-11-12 · 配套：`~/桌面/环境说明.md`（总环境）、本目录的示例产物

---

## 0. 这个目录里是什么

| 文件 | 说明 |
| --- | --- |
| `转台视频.mp4` | 训练好的 3DGS 模型绕场景旋转 96 帧（800×800，24fps） |
| `预览_01~04.png` | 从视频抽的 4 帧静图 |
| `高斯泼溅.ply` | **高斯泼溅模型本体**（18 MB），可拖进网页查看器实时旋转 |

> 打开方式（任选）：
> - 浏览器打开 <https://superspl.at/editor> 或 <https://antimatter15.com/splat/>
>   把 `高斯泼溅.ply` 拖进去 → 鼠标拖动旋转、滚轮缩放
> - 本机 `ffplay 转台视频.mp4` 看渲染效果

---

## 1. 这套管线是什么

**视频/多图 → 相机位姿 → 3D 高斯泼溅 → 可实时浏览的 .ply**

```
视频.mp4 ──ns-process-data──▶ 多视角图 + COLMAP 位姿
                                   │
                                   ▼
                        ns-train splatfacto （训练 3D 高斯）
                                   │
                    ┌──────────────┴──────────────┐
                    ▼                             ▼
          ns-export gaussian-splat        自定义渲染循环
             （.ply 模型）                （转台视频）
```

---

## 2. 已装好的环境

| 项 | 位置 / 值 |
| --- | --- |
| Python 环境 | `~/apps/3dgs/venv`（Python 3.12） |
| torch | **2.15.0+cu132**（CUDA 13.2，sm_120 已验证 `cap (12,0)`） |
| nerfstudio | 最新版（`ns-train` / `ns-export` / `ns-render`） |
| gsplat | **1.4.0**，CUDA 核**已为 sm_120 编译完成** |
| 编译缓存 | `~/.cache/torch_extensions/py312_cu132/gsplat_cuda/gsplat_cuda.so` |
| CUDA toolkit | `/usr/local/cuda-13.0`（nvcc **13.0.88**），`/usr/local/bin/nvcc` 软链 |
| 编译器 | `gcc-14` / `g++-14`（CUDA 13 不支持 gcc 15） |
| colmap / ffmpeg | 系统装（`/usr/bin/colmap`、`/usr/bin/ffmpeg`） |
| 脚本 | `~/apps/3dgs/bin/`（见下） |
| 示例数据 | `~/data/hf/nerf_synthetic/lego`（136 帧 800×800） |

### 脚本清单

| 脚本 | 用途 |
| --- | --- |
| `~/apps/3dgs/bin/env.sh` | 公共环境变量（PATH / CUDA_HOME / gcc-14 / 架构） |
| `bin/train.sh <数据目录> [迭代=15000] [实验名]` | 训练 splatfacto |
| `bin/export.sh [config.yml] [输出目录]` | 导出 `.ply` |
| `bin/render.sh [config.yml] [out.mp4] [帧数]` | 渲染转台视频 |
| `bin/prepare_transforms.py <场景目录>` | NeRF 原版数据 → nerfstudio 格式 |
| `bin/render_turntable.py` | 上面 render.sh 调用的渲染器 |

> **每个脚本都会自己 `source env.sh`**，不需要先激活 venv。

---

## 3. 跑自己的场景（三步）

### 3.1 拍视频的要点

| 要 | 不要 |
| --- | --- |
| 绕物体走一圈（30–60 秒） | 原地转身（没有视差） |
| 匀速、平稳 | 快速甩动（运动模糊） |
| 光照稳定 | 中途开关灯/变焦 |
| 物体静止 | 有人或宠物闯入 |
| 多角度俯视/平视 | 只在一个高度拍 |

### 3.2 视频 → 数据集（COLMAP）

```bash
source ~/apps/3dgs/bin/env.sh
ns-process-data video --data ~/我的视频.mp4 --output-dir ~/data/myscene
```

耗时参考：60 秒 1080p 视频（约 150 帧）→ COLMAP 约 **5–15 分钟**（6 核 CPU）。

> 如果已经有「图片 + COLMAP 结果」，可以跳过这步，直接用 `--data` 指向该目录。
> 如果用的是 NeRF 合成格式数据（带 `transforms_train.json`），先跑：
> `python ~/apps/3dgs/bin/prepare_transforms.py <场景目录>`

### 3.3 训练 + 导出 + 渲染

```bash
~/apps/3dgs/bin/train.sh  ~/data/myscene 15000 myscene   # 会打印 CONFIG=...
~/apps/3dgs/bin/export.sh <CONFIG>                       # 出 .ply
~/apps/3dgs/bin/render.sh <CONFIG> ~/桌面/我的转台.mp4 96
```

---

## 4. 实测数据（NeRF 合成 lego 场景）

| 项 | 值 |
| --- | --- |
| 数据 | 136 帧 / 800×800 |
| 迭代 | 10,000（splatfacto） |
| **训练耗时** | **约 2 分钟** |
| 显存峰值 | **13.3 GB / 16.3 GB** |
| 高斯点数 | 约 100 万级（.ply 18 MB） |
| 渲染 | 96 帧约 20 秒（≈5 帧/秒） |
| 渲染输出 | 556 KB / 24fps |

> 真实拍摄数据（带噪声、运动模糊）通常需要 15k–30k 迭代，训练 5–15 分钟。

---

## 5. 踩过的坑（都已解决，重装时会遇到）

| 现象 | 原因 | 解决 |
| --- | --- | --- |
| `gsplat: No CUDA toolkit found` | 预编译核不含 sm_120，必须现编；Ubuntu 自带 CUDA 只有 **12.4**，编不了 sm_120 | 装 **CUDA 13.0 toolkit** + **gcc-14**，`TORCH_CUDA_ARCH_LIST=12.0` |
| `Ninja is required` | torch 用 `which ninja` 找 ninja | `pip install ninja` 且 **venv/bin 必须在 PATH 里** |
| 训练时又说没 CUDA toolkit | 启动脚本漏了 `PATH=$CUDA_HOME/bin` | 补 PATH（`env.sh` 已包含） |
| `transforms.json` 不存在 | nerfstudio 要这份，数据集只有 `transforms_train.json` | `prepare_transforms.py` |
| `FileNotFoundError: .../r_0` | NeRF 原版 `file_path` 没有扩展名 | 同上，补 `.png` |
| `AssertionError: fx not specified` | 缺 `fl_x/cx/cy` | 同上，由 `camera_angle_x` 计算 |
| `weights_only` UnpicklingError | torch ≥2.6 默认 `weights_only=True` | `export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` |
| `ns-render spiral` 断言失败 | splatfacto 用 `FullImageDatamanager`，官方 spiral 只支持 Vanilla* | 用 `bin/render_turntable.py` 自己渲染 |
| 渲染出来全黑 | `rgb[0]` 误切掉高度维（返回是 H×W×3，无 batch 维） | 先判断 `ndim == 4` |
| 数据集下不动 | nerfstudio 的数据全在 Google Drive | 用 HF 镜像 `hf-mirror.com` 或自己拍 |

---

## 6. 能力边界（16 GB 显存）

| 能做 | 做不到 |
| --- | --- |
| 室内小物体/房间级场景（100–500 帧） | 城市级、大场景（显存和内存都不够） |
| 15k–30k 迭代，几分钟到十几分钟 | 4D 动态场景（Animate3D/4DGS 需 24 GB+） |
| 导出 .ply 到网页/Unity/UE 查看 | 高精度 mesh（3DGS 是点云式，不是网格；要 mesh 用 SuGaR/2DGS 再提） |
| 与 COLMAP/NeRF 数据互通 | 无位姿输入时的纯前馈重建（需 DUSt3R/VGGT 那类模型） |

> 显存不是最紧的：**COLMAP 和训练中间文件很吃内存/磁盘**，32 GB 内存 + 170 GB 盘刚好够用。
