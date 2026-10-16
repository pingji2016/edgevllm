# K3 门户 / 板子访问说明

记录「K3 门户」是怎么访问的，以及网页终端（ttyd）的协议细节。**终端一敲就断的坑在 §4。**

> 板子的硬件能力 / 短板 → [`k3-hardware.md`](../k3-hardware/k3-hardware.md)；
> 「装不了 arm64 App」的解法 → [`arm64-on-riscv64.md`](../k3-arm64-on-riscv/arm64-on-riscv64.md)；
> 端侧 AI 推理实测 → [`k3-ondevice-ai.md`](../k3-ondevice-ai/k3-ondevice-ai.md)。

---

## 1. 它是什么

- 门户地址：`http://111.228.40.211:3000/`，一个自建的 **K3 门户**，输入口令登录。
- 门户背后是一台 **SpacemiT K3（RISC-V）开发板**，登录用户 `pingji`，系统 **Bianbu 4.0.1**（内核 `6.18.3-generic` riscv64）。
- 门户提供两样东西：
  - **状态页** `/`（+ `/api/status`）：反向隧道、心跳、内网 IP、出口网卡等。
  - **网页终端** `/term/`：一个被反代的 **ttyd**，登录板子的 shell。

### 板子配置（实测）

| 项 | 值 |
| --- | --- |
| 主机名 | `pingji-k3` |
| 系统 | Bianbu 4.0.1 (Resolute Raccoon)，`ID=bianbu` |
| 内核 | `Linux 6.18.3-generic #1.0.1.4 SMP PREEMPT_DYNAMIC riscv64` |
| SoC | SpacemiT K3，双簇 16 核：**X100 ×8 @2.4GHz** + **A100 ×8 @2.0GHz** |
| 内存 | 15 GiB（无 swap） |
| 存储 | `/dev/sda` 119.3G：`sda3`→`/`、`sda2`→`/boot`；另有 MTD（SPI flash） |
| 网络 | WiFi `wlP4p1s0` = `192.168.110.194/24`，网关 `192.168.110.1`；有线 `end0`/`enP2p1s0` DOWN |
| 温度 | top ~64°C，GPU ~65°C |

---

## 2. 链路

```
浏览器 / 脚本
   │  http://127.0.0.1:3000        ← 本机 SSH 本地转发（-L）
   ▼
本机 127.0.0.1:3000 ──ssh -L──▶ 公网机 111.228.40.211 上的「K3 门户」(:3000)
                                        │
                                        │ 门户再经 SSH 反向隧道
                                        │ （server 127.0.0.1:2222 ──▶ 板子 :22）
                                        ▼
                                   K3 板  pingji@127.0.0.1 (Bianbu)
```

- **状态页**（`/`、`/api/status`）公网可直接访问（需登录）。
- **终端**（`/term/`）**只允许 loopback 来源**（门户判断请求是不是来自 `127.0.0.1`），所以必须走 `ssh -L`。门户原话：终端是明文，走公网传 shell 太危险。

---

## 3. 访问步骤

1. 在本机开本地转发，窗口保持开着：

   ```bash
   ssh -N -L 3000:127.0.0.1:3000 root@111.228.40.211
   ```

2. 浏览器打开 `http://127.0.0.1:3000/`，输入**门户口令**登录（口令见密码管理器 / 向管理员获取）。

3. 打开 `http://127.0.0.1:3000/term/`，就是板子的 shell（用户 `pingji`）。

> 直接开 `http://111.228.40.211:3000/term/` 时页面上不会出现「打开终端」按钮，因为来源不是 loopback。

登录接口：`POST /login`，字段 `password=<口令>`，成功后下发 cookie：

```
Set-Cookie: k3=v1.<ts>.<hmac>; HttpOnly; SameSite=Strict; Path=/; Max-Age=604800
```

cookie 是 HMAC 签名、有效期 7 天；后续 `/api/status`、`/term/ws` 都靠它鉴权。

---

## 4. 终端（/term/）协议细节 —— 关键坑

`/term/` 是 **`ttyd/1.7.7 (libwebsockets/4.3.5)`**，门户是它的反代。ttyd 的启动命令（从终端窗口标题读到）是：

```
ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o BatchMode=yes \
    -o IdentitiesOnly=yes -o ServerAliveInterval=30 -i /app/ssh/id_ed25519 \
    -p 2222 -l pingji 127.0.0.1
```

即门户在服务器上用 `/app/ssh/id_ed25519`，经反隧道（`127.0.0.1:2222` → 板子 `22`）登录板子。

**WebSocket 端点**：`ws://127.0.0.1:3000/term/ws`，子协议 `tty`，需带 cookie `k3=...`。

### 协议（用真实浏览器抓包 + 实测得到）

服务端 → 客户端（**二进制帧**，首字节是命令）：

| 首字节 | 含义 |
| --- | --- |
| `'0'` + data | 终端输出 |
| `'1'` + text | 窗口标题 |
| `'2'` + json | 偏好设置 |
| ping（opcode 9） | 心跳，必须回 pong（opcode 10） |

客户端 → 服务端：

1. **第一帧必须是认证 JSON**（二进制帧，**没有前缀**）：
   ```json
   {"AuthToken":"","columns":C,"rows":R}
   ```
2. 之后每次按键用 `'0'` + data（二进制帧）。

> ⚠️ **这就是「连上了但一敲就断」的根因**：这个 ttyd 被定制成**先发 JSON 认证帧**。如果按标准 ttyd 协议第一帧就发 `'1'+resize` 或 `'0'+输入`，服务端会直接断开（实测：任意以 `'0'` 开头、或在认证前发的帧，0.03s 内被断开；不发的帧则一直被忽略）。

---

## 5. 可复用的执行脚本

`k3exec.py`：读 `curl -c` 存的 cookie 文件，先发认证 JSON，再发一条命令，收若干秒输出。

```python
#!/usr/bin/env python3
# 用法: python3 k3exec.py '<shell 命令>' [等待秒数]
import socket, base64, os, struct, sys, time, json

cookie = None
for line in open('/tmp/k3_cookies.txt'):
    if line.startswith('#') and not line.startswith('#HttpOnly_'):
        continue
    p = line.split()
    if len(p) >= 7 and p[5] == 'k3':
        cookie = 'k3=' + p[6]
assert cookie, 'no cookie'

cmd = sys.argv[1]
wait = float(sys.argv[2]) if len(sys.argv) > 2 else 25.0
HOST, PORT = '127.0.0.1', 3000
key = base64.b64encode(os.urandom(16)).decode()
req = ("\r\n".join([
    "GET /term/ws HTTP/1.1", f"Host: 127.0.0.1:{PORT}", "Connection: Upgrade",
    "Upgrade: websocket", f"Origin: http://127.0.0.1:{PORT}", "Sec-WebSocket-Version: 13",
    f"Sec-WebSocket-Key: {key}", "Sec-WebSocket-Protocol: tty", f"Cookie: {cookie}",
])) + "\r\n\r\n"
s = socket.create_connection((HOST, PORT), timeout=20)
s.sendall(req.encode())
rx = b''
while b'\r\n\r\n' not in rx:
    d = s.recv(4096)
    if not d: sys.exit('HS-CLOSED')
    rx += d
assert b' 101 ' in rx.split(b'\r\n')[0], rx.split(b'\r\n')[0]
rx = rx.split(b'\r\n\r\n', 1)[1]

def ws(data, op=2):
    n = len(data); hdr = bytes([0x80 | op])
    if n < 126: hdr += bytes([0x80 | n])
    elif n < 65536: hdr += bytes([0x80 | 126]) + struct.pack('>H', n)
    else: hdr += bytes([0x80 | 127]) + struct.pack('>Q', n)
    m = os.urandom(4)
    s.sendall(hdr + m + bytes(data[i] ^ m[i % 4] for i in range(n)))

def need(n, dl):
    global rx
    while len(rx) < n:
        s.settimeout(max(0.05, dl - time.time()))
        try: d = s.recv(65536)
        except socket.timeout: return None
        if not d: return b''
        rx += d
    v, rx = rx[:n], rx[n:]
    return v

buf = b''
def pump(sec):
    global buf
    dl = time.time() + sec
    while time.time() < dl:
        h = need(2, dl)
        if h is None: continue
        if h == b'': return
        b0, b1 = h[0], h[1]; op = b0 & 0x0f; ln = b1 & 0x7f
        if ln == 126: ln = struct.unpack('>H', need(2, dl))[0]
        elif ln == 127: ln = struct.unpack('>Q', need(8, dl))[0]
        if b1 & 0x80: need(4, dl)
        pl = need(ln, dl) if ln else b''
        if pl is None: pl = b''
        if op == 9: ws(pl, op=0xA)          # ping -> pong
        elif op == 8: return
        elif op in (1, 2) and pl[:1] == b'0': buf += pl[1:]

ws(json.dumps({"AuthToken": "", "columns": 200, "rows": 50}).encode())   # ① 认证帧
pump(2.5)
ws(b'0' + (cmd + '\r').encode())                                        # ② 输入
pump(wait)
ws(b'0' + b'exit\r')
s.close()
print(buf.decode('utf-8', 'replace'))
```

先拿 cookie：

```bash
curl -s -c /tmp/k3_cookies.txt -X POST http://127.0.0.1:3000/login -d 'password=<口令>'
python3 k3exec.py 'uname -a; free -h' 10
```

---

## 6. 实测时延 / 带宽

在板子上测得（`ping` / `curl`）。

### 时延（RTT）

| 目标 | RTT | 说明 |
| --- | --- | --- |
| 网关 `192.168.110.1` | **~1.1 ms** | WiFi 局域网 |
| 门户服务器 `111.228.40.211` | TCP `:22` **~101 ms**、`:3000` **~43 ms** | ICMP 被服务器丢，只能用 TCP 测 |
| 阿里 DNS `223.5.5.5` | **~83 ms**（min 49 / max 128） | 国内 |
| Cloudflare `speed.cloudflare.com` | TCP 连接 **~191 ms**、TLS **~401 ms** | 出海 |
| Google `8.8.8.8` | **~312 ms** | 出海 |

### 带宽（Cloudflare 测点，板子 ↔ 公网）

| 方向 | 数据量 / 耗时 | 速率 |
| --- | --- | --- |
| 下行 | 50 MB / 4.60 s | **≈ 10.9 MB/s ≈ 87 Mbps** |
| 上行 | 20 MB / 4.04 s | **≈ 4.95 MB/s ≈ 40 Mbps** |

> 这是板子 WiFi 链路的裸带宽。**走 SSH 反向隧道是单条 TCP**（且 SSH 加密 + TCP-over-TCP），经隧道实际吞吐会明显低于这个，通常只有几 MB/s 量级。

---

## 7. 排错

- **WS 返回 `401 Unauthorized` / 握手后立刻断**：
  - cookie 不对。用 `curl -c` 当场拿的 cookie；经代理 / 换了 host 的 cookie 可能无效（实测：经 `:3099` 代理那次浏览器会话的 cookie 在 `:3000` 上是 401）。
  - 门户**像是有并发终端限制**：浏览器终端标签开着时，脚本连会被拒或即断。**先关掉浏览器里的 `/term/` 再跑脚本。**
- **握手就 `CLOSED`**：确认走的是 `127.0.0.1:3000`（loopback）且带 `k3` cookie。
- **一敲就断**：见 §4 —— 第一帧没发认证 JSON。
- **板子 `ping` 服务器不通**：服务器丢 ICMP，正常；用 TCP（`:22` / `:3000`）判断连通性。

---

## 8. 安全

- 门户口令 = 板子 shell 权限；终端明文过公网，**务必只用 `ssh -L` 访问**。
- 别把口令 / cookie 写进仓库；本文档用占位符。
- 板子用户 `pingji` 在 `sudo` 组，等于 root。

---

## 9. 在 K3 上运行 8B 模型（已跑通）

K3 是 RISC-V + IME/NPU 的芯片，官方 apt 源里就有现成运行时，不用自己编译。下面全部**用户态、无需 root**。

### 9.1 最省事：SpacemiT 定制版 ollama

```bash
# 官方 ollama 工具包，解包到 home（就是单个二进制）
apt-get download spacemit-ollama-toolkit
dpkg-deb -x spacemit-ollama-toolkit_*_riscv64.deb ~/ollama-dist
mkdir -p ~/ollama/bin && cp ~/ollama-dist/usr/bin/ollama ~/ollama/bin/

# 启动服务（后台常驻）
setsid nohup ~/ollama/bin/ollama serve >~/ollama/serve.log 2>&1 </dev/null &

# 拉 8B 模型（Qwen3-8B Q4_K_M ≈ 5.2GB，来自 ollama.com，实测 ~21MB/s）
~/ollama/bin/ollama pull qwen3:8b

# 对话（Qwen3 默认带思考，交互里 /set nothink，或 API 传 think:false）
~/ollama/bin/ollama run qwen3:8b
```

一键脚本 `~/ollama/run.sh`（自动拉起服务并进入对话）：

```bash
~/ollama/run.sh            # 默认 qwen3:8b
~/ollama/run.sh qwen3:4b   # 换模型
```

HTTP API：

```bash
curl http://127.0.0.1:11434/api/chat -d '{"model":"qwen3:8b","stream":false,"think":false,
  "messages":[{"role":"user","content":"你好"}]}'
```

### 9.2 更快 prompt：SpacemiT 版 llama.cpp（走 X100 的 IME2 矩阵扩展）

```bash
for p in llama.cpp-tools-spacemit spacemit-runtime spacemit-onnxruntime; do apt-get download $p; done
mkdir -p /tmp/sp && for f in llama.cpp-tools-spacemit*.deb spacemit-runtime*.deb spacemit-onnxruntime*.deb; do dpkg-deb -x "$f" /tmp/sp; done
export LD_LIBRARY_PATH=/tmp/sp/usr/lib
M=$(ls -S ~/.ollama/models/blobs | head -1)   # 复用 ollama 拉下来的 GGUF
/tmp/sp/usr/bin/llama-bench -m ~/.ollama/models/blobs/$M -p 128 -n 64 -t 8
```

启动兼容 OpenAI 的服务：`/tmp/sp/usr/bin/llama-server -m <gguf> -t 8 --host 127.0.0.1 --port 8080`。

### 9.3 实测（Qwen3-8B Q4_K_M，8 个大核）

| 运行时 | prompt 处理 pp128 | 生成 tg |
| --- | --- | --- |
| ollama (`+spacemit`) | ~3.6 t/s | **2.45 t/s** |
| SpacemiT llama.cpp（IME2，`-t 8`） | **27.5 t/s** | **2.86 t/s** |

结论：**K3 确实能跑 8B**，但生成受 DDR 带宽限制，约 **2.5–3 tok/s**（一秒一个词左右）；prompt 处理走 IME 快很多。要更快只能上更小的模型（`qwen3:4b`/`1.7b`）。

### 9.4 备注

- 板子已带 NPU 设备节点（`/dev/npuctl`、`/dev/npu0`）和 `libspert.so`、`libspacemit_ep.so`（onnxruntime 的 SpacemiT 执行器）；上面 llama.cpp 用的是 **CPU+IME2** 后端，不是 NPU 全量 offload。
- 想装系统级 `ollama.service` 或官方图形版 `spacemit-ailab`（模型体验平台）需要 **sudo**（`pingji` 有 sudo 权限但要密码）。
- 二进制 + 模型都在 `~/ollama`，约占 5.5GB。
