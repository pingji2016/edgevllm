# K3 流式语音转文字（ASR）服务

把 **sherpa-onnx 的流式 zipformer（中英双语）** 封装成一个 **WebSocket 流式 ASR 服务**，供手机/网页/后端调用；服务端支持**多工作线程 + 批处理**（多人并发）。

> 模型：`sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20`（int8，**仅中/英**）。
> 要更快/多语 → 换 SenseVoice（离线，见 `k3-asr.md`）。

---

## 1. 架构

```
手机 / 网页 / 后端
      │  WebSocket  ws://<k3-ip>:6006
      │  ② 二进制帧：float32 PCM(16k,单声道,归一化)   ①. 接收增量 JSON 结果
      ▼                                              ▲
┌─────────────────────────────────────────────────────┐
│ sherpa-onnx-online-websocket-server                   │
│  · 每连接一个流式识别会话                             │
│  · num-work-threads 个工作线程 + max-batch-size 批处理 │
│  · streaming zipformer (int8) @ CPU(RVV/IME)          │
└─────────────────────────────────────────────────────┘
```

## 2. 启动服务

```bash
cd ~/mage-vl/sherpa
./start-asr-server.sh                 # 默认 :6006
PORT=6006 THREADS=8 BATCH=8 ./start-asr-server.sh   # 调并发
```
开机自启（可选）：
```bash
sudo cp asr-server.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now asr-server
```

## 3. 协议（sherpa-onnx online websocket）

- **连接**：`ws://<k3-ip>:6006`
- **客户端 → 服务端**
  - **二进制帧**：音频样本，**float32、16kHz、单声道，范围 [-1,1]**（推荐每帧 0.1–0.5s，如 8000 样本=0.5s）
  - **文本帧**：`Done` 表示本句结束（服务端 flush 并给最终结果）
- **服务端 → 客户端**（文本 JSON）
  ```json
  {"text":"昨天是 MONDAY","tokens":[...],"timestamps":[...],
   "segment":0,"is_final":false,"is_eof":false}
  ```
  说一句出一串增量（partial），最后一条 `is_final:true`。
- 模型在服务端启动时固定，客户端无需传参。

## 4. 客户端示例

**Python（`client/asr_stream_client.py`）**：
```bash
python3 client/asr_stream_client.py foo.wav 127.0.0.1 6006 0.5
# 输出（边说边出）:
# [part ] 昨天
# [part ] 昨天是 MON
# [part ] 昨天是 MONDAY
# [FINAL] 昨天是 MONDAY
```

**手机接入**：用任意 WebSocket 库，采集麦克风 PCM（16k/mono/float32），按帧 `send(bytes)`，说完 `send("Done")`，解析回包 JSON 的 `text`。
> 手机上传带宽：16k×16bit×1ch = **32 KB/s（256 kbps）**，4G/WiFi 都毫无压力；瓶颈在服务端算力，不在网络。

## 5. 并发与性能（K3，8 核上可用）

| 并发 | 实测 | 单路 RTF | 说明 |
| --- | --- | --- | --- |
| 1 路（client 按实时发） | 正常出增量 | **~0.5**（≈2× 实时） | 服务端比实时快约 2 倍 |
| 4 路并发 | 4 路**全部成功** | 聚合 **~3× 实时** | worker/批处理 |
| 离线 SenseVoice（非流式） | 15.15s 音频 2.19s | **0.145（≈7× 实时）** | 更准/多语 |

**"手机调用 RTF 能到多少"**：
- RTF 是**服务端**指标（手机只负责传音频）。本板子单路 ≈0.5；约 **3–4 路并发时接近实时（RTF≈1）**；再多人就会积压。
- 手机端**感知延迟** = 网络 RTT + 分片时长(0.1–0.5s) + 处理(≈0.5×分片)；**局域网大约 <0.5s**，流式输出。
- 想扛更多人：换 SenseVoice/更小模型、或加多机/进程池做水平扩展（服务无状态，每连接独立）。

## 6. 文件

```
~/mage-vl/sherpa/
├── start-asr-server.sh          # 启动脚本
├── asr-server.service           # systemd 单元
├── client/asr_stream_client.py  # Python 流式客户端
├── tokens.txt / encoder|decoder|joiner-*.int8.onnx   # 模型
└── sherpa-onnx-...-spacemit-shared/bin/               # 预编译 sherpa-onnx
```
