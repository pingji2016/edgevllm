/* k3aix —— K3 的 A100 核启用器
 *
 * 用途：把**本线程**划入 A100（HMP_AI）核组，然后 execve 目标程序。
 *
 * 为什么不能直接 `echo 0 > /proc/set_ai_thread`：
 *   内核要求「必须在执行任何 RVV 向量指令之前」完成切换（X100 是 VLEN=256，
 *   A100 是 VLEN=1024，切换后向量状态语义不同）。shell/glibc 在做 memcpy 等
 *   操作时可能已经用过向量指令，会触发内核告警：
 *     hmp_set_ai_thread: pid:%u(%s), vector has been enabled already!!!
 *   这个极小的 C 程序不做任何向量操作，写完节点立刻 execve，最接近官方语义
 *   （社区工具 brucehoult/k3_ai 的 aix 也是这个思路，只是用纯汇编写）。
 *
 * 编译：cc -O2 -o k3aix k3aix.c
 * 用法：k3aix 命令 [参数...]        例：k3aix taskset -c 8-15 nproc
 *      k3aix llama-cli -m model.gguf -t 8
 *
 * 注意：A100 是「向量/矩阵专用核」（VLEN 1024 + IME，缓存小、主频低）。
 *      通用 LLM 推理（llama.cpp 的 Q4 路径）在 A100 上反而慢约 5.8 倍，
 *      实测见 k3-ondevice-ai-report.md 第 4.2 节。要用它请走 int8/IME 算子。
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>

static void die(const char *what)
{
    fprintf(stderr, "k3aix: %s: %s\n", what, strerror(errno));
    exit(1);
}

int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr,
            "用法: k3aix 命令 [参数...]\n"
            "  例: k3aix taskset -c 8-15 nproc\n"
            "      k3aix llama-cli -m model.gguf -t 8\n");
        return 2;
    }

    /* 写 "0" = 当前线程 */
    int fd = open("/proc/set_ai_thread", O_WRONLY);
    if (fd < 0)
        die("open(/proc/set_ai_thread) 失败（需要内核 CONFIG_SPACEMIT_HMP=y）");
    if (write(fd, "0", 1) != 1)
        die("write(/proc/set_ai_thread) 失败");
    close(fd);

    execvp(argv[1], &argv[1]);   /* 同一线程，保持 HMP_AI */
    die("execvp 失败");
    return 1;
}
