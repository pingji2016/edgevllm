# ⚠️ 重要提示：这台机器的磁盘红线

> 主机 `pingji-MS-7B89` · Ubuntu 26.04.1 LTS · 写于 2026-11-11
>
> **一句话：两块 NVMe 盘全是 Windows 的东西，只能读、绝对不能写。
> 只能改 `sda`（Linux 系统盘）。**

---

## 1. 磁盘清单与权限

| 设备 | 大小 | 里面是什么 | 当前挂载 | 允许操作 |
| --- | ---: | --- | --- | --- |
| `/dev/sda` | 238 G | **Linux 系统盘** | `/boot/efi` `/` | ✅ **随便读写**，所有安装/构建/数据都放这里 |
| ├ `/dev/sda1` | 1 G | EFI 分区 | `/boot/efi` | ✅ 可读写（但别乱动） |
| └ `/dev/sda2` | 237 G | ext4 根分区 | `/` | ✅ **可读写**，剩 ~203 G |
| `/dev/nvme0n1` | 233 G | **Windows 系统盘** | 未挂载 | ❌ **禁止任何写入** |
| ├ `nvme0n1p1` | 16 M | Windows 保留分区 | — | ❌ 禁止 |
| ├ `nvme0n1p2` | 232 G | Windows C: | — | ❌ 禁止 |
| └ `nvme0n1p3` | 812 M | Windows 恢复分区 | — | ❌ 禁止 |
| `/dev/nvme1n1` | 931 G | **Windows 数据盘** | p2 只读挂载 `/mnt/v2` | ❌ **禁止任何写入** |
| ├ `nvme1n1p1` | 300 G | 数据分区 | 未挂载 | ❌ 禁止 |
| └ `nvme1n1p2` | 631 G | 数据分区（99% 满） | `/mnt/v2`（**ro 只读**） | ❌ 禁止（只读挂载仅供复制文件） |

> 目前**没有 `sdb`**。除 `sda` 之外，任何盘都不许写。

---

## 2. 绝对禁止的操作

### 2.1 分区/格式化类（对 `nvme*` 一律禁止）

```bash
# ❌ 全部禁止，会破坏 Windows 分区表 / 文件系统
mkfs.ext4 /dev/nvme1n1p1        mkfs.ntfs /dev/nvme1n1p1
parted /dev/nvme1n1             fdisk /dev/nvme0n1
gdisk /dev/nvme1n1              sgdisk /dev/nvme1n1
cfdisk /dev/nvme0n1             wipefs /dev/nvme1n1p2
dd if=... of=/dev/nvme1n1       # 任何 dd 写到 nvme*
```

### 2.2 NTFS 修复类（会写入，禁止）

```bash
# ❌
ntfsfix /dev/nvme1n1p2
ntfsresize /dev/nvme1n1p2
chkdsk /f                       # 在 Windows 里也别随便跑
```

### 2.3 挂载类

```bash
# ❌ 默认是 rw，会写盘
sudo mount /dev/nvme1n1p2 /mnt/v2

# ✅ 必须显式只读
sudo mount -o ro /dev/nvme1n1p2 /mnt/v2
```

### 2.4 图形界面里的隐形写入（最容易踩）

| 操作 | 后果 |
| --- | --- |
| 在 `/mnt/v2` 里删除/改名/新建文件 | 写 NTFS，可能损坏 |
| 双击打开文档后保存 | 写回原文件 |
| 拖拽移动文件到/从 `/mnt/v2` | 写入 |
| 文件管理器生成缩略图/回收站（`.Trash-1000`） | 写入 |
| 自动索引（`tracker`、`locate`/`updatedb`） | 写入 |

**正确做法：只 `cp` 出来到 `~/`，在系统盘上处理。**

---

## 3. 正确姿势

```bash
# 只读挂载
sudo mkdir -p /mnt/v2
sudo mount -o ro /dev/nvme1n1p2 /mnt/v2

# 确认是只读（看输出里有 "ro"）
mount | grep nvme

# 把需要的东西复制到系统盘再用
cp "/mnt/v2/do/xxx.deb" ~/Downloads/

# 用完卸载（或者干脆重启，只读挂载不会自动恢复）
sudo umount /mnt/v2

# 卸载前确认没人占用
lsof /mnt/v2
```

---

## 4. 为什么这么严格

1. **Windows 的 NTFS 带"快速启动/休眠"残留状态**：Windows 关机后磁盘可能仍是"脏"的，
   Linux 一旦写入，轻则文件损坏，重则整个分区表报废。
2. **不可撤销**：NTFS 误写没有 `fsck` 级别的后悔药，Windows 那边可能直接起不来，需要重装。
3. **里面装满了东西**：931 G 那块盘已用 99%（只剩 12 G），是长期资料盘。

---

## 5. 如果非要写 NVMe 盘（正常情况下不要）

必须同时满足：

1. 在 Windows 里关闭「快速启动」和「休眠」（`powercfg /h off`）；
2. **完全关机**，不是睡眠/休眠；
3. 明确知道自己要做什么，并已备份；
4. 只动你确定不需要 Windows 的那一个分区。

否则：**不要做**。

---

## 6. 怎么确认没有误写

```bash
# 看挂载是否为 ro
mount | grep -E 'nvme|/mnt/v2'

# 看内核有没有 NTFS 写操作/错误
sudo dmesg | grep -i ntfs | tail -20
```

`mount` 输出里 `/dev/nvme1n1p2 on /mnt/v2 type ntfs3 (ro,...)` —— 出现 `ro` 才是安全的。

---

## 7. 本机已做过的只读操作（留档）

| 时间 | 操作 | 是否写入 |
| --- | --- | --- |
| 2026-11-11 | `mount -o ro /dev/nvme1n1p2 /mnt/v2` | 否（只读挂载） |
| 2026-11-11 | `cp /mnt/v2/do/v2rayN-linux-64.deb /tmp/` | 否（从盘读取） |
| 2026-11-11 | `find /mnt/v2 -iname "*v2ray*"` | 否（只读遍历） |

---

## 8. 给未来的自己 / 给 AI 的指令

> **任何涉及磁盘的操作，先确认设备名。**
> 名字里带 `nvme` 的，一律只读。要写入只允许 `sda*`。
> 不确定就先 `lsblk` 确认，再问人。
