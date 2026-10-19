# K3 反向隧道失联的恢复手册（2026-10-19 事故复盘）

## 一句话结论

**板子活着、出网正常，但 `板子 → 111:22` 的 SSH 会话被中间设备掐断在握手阶段。**
远程没有任何入口，只能从**板子所在局域网**或**物理接触**恢复。

---

## 证据（全部实测）

| 观象 | 数据 | 说明 |
|---|---|---|
| 门户心跳 | 每 30s 更新，`uptime` 连续 | 板子活着、没重启、出网正常 |
| 心跳源 IP | `112.81.104.243`（从 111:3000 抓的） | 板子的公网出口 |
| `111:3000` 上的 HTTP | 三次握手 + POST 422B + FIN **全通** | 家用网络到 111 是通的 |
| `111:22` 上的 SSH | 板子发 SYN → 111 回 SYN-ACK → **板子再无任何回包** | 握手后立刻死 |
| 已建立的隧道连接 | 111 侧 `Send-Q 1162`（SSH banner+KEXINIT）**始终不被 ACK** | 111→板子方向丢包 |
| 180 秒内新 SYN | **只有 1 个** | 不是"每 10s 重试"，是**卡死** |
| 111 负载 / iptables / fail2ban | 0.21、ufw inactive、该 IP 未被封 | 不是 111 的问题 |
| 我的 Mac → 111:22 | 完全正常 | 不是 111 sshd 的问题 |

**为什么排除 MTU**：被丢的那个包只有 1162 字节，两端 MTU 分别是 1450 / 1440，远未到分片。

**为什么是"按端口/协议掐流"**：同一个 IP、同一时刻，`:3000` 的 HTTP 正常，`:22` 的 SSH 恰好在
`SSH-2.0-` 这个特征出现时死掉 —— 这是家用路由器"禁止 SSH 外连"或运营商 DPI 的典型形态。

**为什么卡住不会自愈**（单元配置的坑）：

- `ConnectTimeout=10` 只管 **TCP connect**，不管 SSH 握手；
- `ServerAliveInterval=30` **只在认证成功后才生效**；
- 于是 ssh 永远挂在 banner 交换里，既不退出也不超时，`Restart=always` 永远不触发。

---

## 恢复路径（按可行性排序）

### A. 有人在板子所在局域网（最快，2 分钟）

板子的 sshd 在局域网里是好的，任何连在同一个 WiFi/交换机上的设备都能进：

```bash
ssh pingji@192.168.110.194          # 或它当前的 LAN IP
# 进去后二选一：
sudo systemctl restart k3-tunnel    # 先试这个（可能只是瞬时故障）
# 若 30 秒后 111 上仍无 127.0.0.1:2222 监听 → 换端口，见下节
```

判定成功：111 上 `ss -ltn | grep 2222` 出现 `127.0.0.1:2222`，且门户心跳正常。

### B. 物理接触

键盘 + HDMI，或串口。登录后同上。

### C. 彻底没人的情况

只能等。板子不会自己恢复（见上面的"卡住不会自愈"）。

---

## 换端口（绕过 22）—— 这是长期解法

### 已完成（111 侧，2026-10-19）

- 111 的 sshd 已额外监听 **8443**（`/etc/ssh/sshd_config.d/10-altport.conf` +
  `ssh.socket.d/10-altport.conf`）。
- 注意：111 是 `ssh.socket` **套接字激活**，实际监听由 `systemd-ssh-generator`
  从 `sshd_config` 的 `Port` 行生成；只改 socket drop-in 不生效，必须两边都改再
  `daemon-reload` + `restart ssh.socket`。
- **不要用 2222**：反向转发自己就要绑 `127.0.0.1:2222`，会和 sshd 的 `*:2222` 冲突，
  配合 `ExitOnForwardFailure=yes` 会导致隧道无限重连。

### 需要你做（京东云控制台，约 30 秒）

安全组入站放行 **TCP 8443**。

当前放行的只有 `22 / 80 / 3000 / 8000 / 8080`（实测 443、2222、8443 都是 filtered）。

### 板子侧（一键）

```bash
./tunnel-altport.sh 8443 111.228.40.211
# 脚本会：备份 unit → 预检 8443 连通性 → 在 ExecStart 前插 -p 8443
#         → daemon-reload → restart → 打印状态和日志
# 回滚： ./tunnel-altport.sh --revert
```

改完 unit 里那一行会从：

```
  root@111.228.40.211
```

变成：

```
  -p 8443 root@111.228.40.211
```

> ⚠️ 如果板子上装过 Android 的隧道 drop-in（`/etc/systemd/system/k3-tunnel.service.d/android.conf`），
> 它的 `ExecStart=`（空值）会**整体清空**主 unit 的命令，再给一份完整的。
> 所以只改主 unit 不生效 —— 脚本会把主 unit 和所有 drop-in 里含 `user@host` 的行一起改。

---

## 加固：让它下次能自愈

装看门狗（`k3-tunnel-watchdog.timer`，每分钟检查一次）：

```bash
sudo install -m755 k3-tunnel-watchdog.sh     /usr/local/bin/
sudo install -m644 k3-tunnel-watchdog.service /etc/systemd/system/
sudo install -m644 k3-tunnel-watchdog.timer   /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now k3-tunnel-watchdog.timer
```

判据是"隧道 socket 的 `Send-Q` 连续两个周期 > 0"（本端有数据一直没被对端 ACK），
健康但空闲的隧道 `Send-Q` 为 0，不会被误杀。

---

## 三条教训

1. **心跳通 ≠ 可管理。** 心跳（HTTP:3000）和隧道（SSH:22）是两条独立链路，
   排障时先分开看：`ss -ltn | grep 2222` 有没有监听，比"能不能 ping"有用得多。
2. **SSH 的 `ConnectTimeout` 和 `ServerAliveInterval` 都盖不住"握手卡死"**，
   必须有外部看门狗，否则一次掐流就是永久失联。
3. **管理端口别和转发端口撞。** 换端口时先看 `PermitListen` / `-R` 里绑的本地端口。
