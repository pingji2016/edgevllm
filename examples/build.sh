#!/usr/bin/env bash
#
# 编译 / 运行 examples 下的 CUDA 示例。Windows(Git Bash) 和 Linux / WSL2 通用。
#
#   ./build.sh                                   编译 examples/ 下所有 .cu
#   ./build.sh run                               编译后运行
#   ./build.sh clean                             清理 build/
#   ./build.sh leetgpu/01_vector_add             只处理指定目录
#   ./build.sh run leetgpu/01_vector_add         编译 + 运行指定目录
#
# 环境变量：
#   CUDA_ARCH   nvcc 的 -arch 值，默认 native（自动探测本机 GPU）
#   NVCC        nvcc 路径，默认从 PATH 中查找
#
# 产物统一放在 examples/build/ 下（.gitignore 已忽略），不污染源码目录。
#
# 注意：运行时输出一律用 ASCII。中文只在注释里——Windows 控制台代码页会把
# UTF-8 中文显示成乱码，和 .cu 文件里那个坑是同一个原因。

set -euo pipefail

EXAMPLES_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$EXAMPLES_DIR/build"
CUDA_ARCH="${CUDA_ARCH:-native}"

# Windows(Git Bash / MSYS / Cygwin) 下可执行文件要带 .exe
case "$(uname -s)" in
    MINGW*|MSYS*|CYGWIN*) EXE_SUFFIX=".exe" ; IS_WINDOWS=1 ;;
    *)                    EXE_SUFFIX=""     ; IS_WINDOWS=0 ;;
esac

die() { printf '\033[31merror\033[0m %s\n' "$*" >&2; exit 1; }
info() { printf '\033[36m==>\033[0m %s\n' "$*"; }
ok()   { printf '\033[32m  ok\033[0m %s\n' "$*"; }

# ---------------------------------------------------------------------------
# 找 nvcc
# ---------------------------------------------------------------------------
find_nvcc() {
    if [[ -n "${NVCC:-}" ]]; then
        [[ -x "$NVCC" ]] || die "NVCC=$NVCC 不可执行"
        return
    fi
    if command -v nvcc >/dev/null 2>&1; then
        NVCC="$(command -v nvcc)"
        return
    fi
    # 兜底：常见安装位置（WSL 上非登录 shell 可能没配 PATH）
    for c in /usr/local/cuda/bin/nvcc /usr/local/cuda-*/bin/nvcc; do
        if [[ -x "$c" ]]; then
            NVCC="$c"
            return
        fi
    done
    die "找不到 nvcc。Windows 装 CUDA Toolkit；WSL 见 examples/README.md。
     或者显式指定：NVCC=/usr/local/cuda-12.8/bin/nvcc ./build.sh"
}

# ---------------------------------------------------------------------------
# Windows 专有检查：.cu 必须是带 BOM 的 UTF-8
#
# 不是 BOM 就报错而不是自动加——自动改用户的源文件太粗暴，而且 BOM 在
# Linux 上无害、在 Windows 上必需，让用户知道原因比默默修好更有价值。
# ---------------------------------------------------------------------------
check_bom() {
    local f="$1"
    (( IS_WINDOWS )) || return 0

    # 纯 ASCII 文件不需要 BOM。
    # 用 [[:print:][:space:]] 而不是 grep -P '[\x80-\xff]'——后者在 LC_ALL=C 下
    # 会直接报 "supports only unibyte and UTF-8 locales" 而失败，检查会被静默跳过。
    if LC_ALL=C grep -q '[^[:print:][:space:]]' "$f"; then
        if [[ "$(head -c 3 "$f" | od -An -tx1 | tr -d ' \n')" != "efbbbf" ]]; then
            die "$f 含非 ASCII 字符但没有 UTF-8 BOM。
     nvcc 前端在 Windows 上会按本地代码页(cp936)读源码，中文注释会被拆错字节，
     报出 'identifier is undefined' / 'missing closing quote' 之类的假错误。
     修法：给文件加 BOM ——  python -c \"p='$f';b=open(p,'rb').read();open(p,'wb').write(b'\\xef\\xbb\\xbf'+b)\""
        fi
    fi
}

# ---------------------------------------------------------------------------
# 编译一个 .cu，返回可执行文件路径到 $BIN
# ---------------------------------------------------------------------------
build_one() {
    local src="$1"
    local rel="${src#"$EXAMPLES_DIR"/}"
    # Windows 下用户可能写反斜杠（leetgpu\01_vector_add），统一成正斜杠，
    # 否则 dirname 切不开，产物会落到 build/leetgpu\01_vector_add/ 这种混合路径
    rel="${rel//\\//}"
    local out_dir="$BUILD_DIR/$(dirname "$rel")"
    local name; name="$(basename "${src%.cu}")"
    local bin="$out_dir/$name$EXE_SUFFIX"

    mkdir -p "$out_dir"
    check_bom "$src"

    # -O3 优化；-arch 默认 native 自动匹配本机 GPU
    # Windows 上 nvcc 会自动调起 MSVC cl.exe；Linux/WSL 用 g++
    if ! "$NVCC" -O3 -arch="$CUDA_ARCH" "$src" -o "$bin" 2>"$out_dir/$name.build.log"; then
        echo "--- 编译输出 ---" >&2
        cat "$out_dir/$name.build.log" >&2
        die "$rel 编译失败"
    fi
    ok "$rel  ->  ${bin#"$EXAMPLES_DIR"/}"
    BIN="$bin"
}

# ---------------------------------------------------------------------------
# 收集要处理的 .cu
# ---------------------------------------------------------------------------
collect_sources() {
    local targets=("$@")
    if (( ${#targets[@]} == 0 )); then
        find "$EXAMPLES_DIR" -name '*.cu' -not -path "$BUILD_DIR/*" | sort
    else
        local t
        for t in "${targets[@]}"; do
            t="${t//\\//}"                       # 反斜杠归一化，见 build_one
            local d="$EXAMPLES_DIR/$t"
            [[ -e "$d" ]] || d="$t"          # 也接受绝对路径 / 相对当前目录
            [[ -e "$d" ]] || die "找不到：$t"
            if [[ -d "$d" ]]; then
                find "$d" -name '*.cu' | sort
            else
                echo "$d"
            fi
        done
    fi
}

# ---------------------------------------------------------------------------
main() {
    local mode="build"
    case "${1:-}" in
        run)   mode="run";   shift ;;
        clean) mode="clean"; shift ;;
        # 打印文件头注释当帮助。按内容截断（从第 3 行起、遇到第一行非注释就停），
        # 不要写死行号——改一次头注释就会把 set -euo pipefail 之类的代码漏出来。
        # 必须写成 if/else 而不是两条并列规则：并列规则会按顺序对同一行求值，
        # 第一条里的 sub() 改掉 $0 之后，第二条的 /^#/ 就不再匹配，第一行就 exit 了。
        -h|--help) awk 'NR>2 { if (/^#/) { sub(/^# ?/, ""); print } else exit }' \
                       "${BASH_SOURCE[0]}"; exit 0 ;;
    esac

    if [[ "$mode" == "clean" ]]; then
        rm -rf "$BUILD_DIR"
        info "已清理 $BUILD_DIR"
        exit 0
    fi

    find_nvcc
    info "nvcc      $NVCC"
    info "arch      $CUDA_ARCH"
    info "platform  $(uname -s)$([[ $IS_WINDOWS == 1 ]] && echo '  (MSVC host compiler)')"
    echo

    local srcs; mapfile -t srcs < <(collect_sources "$@")
    (( ${#srcs[@]} )) || die "没有找到任何 .cu 文件"

    local ran=0
    for src in "${srcs[@]}"; do
        build_one "$src"
        if [[ "$mode" == "run" ]]; then
            echo "--- 运行 $(basename "$BIN") ---"
            "$BIN"
            echo
        fi
        # 不能写 ((ran++))：ran 为 0 时它返回旧值 0，((...)) 判定为假，
        # 在 set -e 下会静默终止整个循环（只会处理第一个文件）。
        ran=$((ran + 1))
    done

    echo
    info "完成：$ran 个目标 -> ${BUILD_DIR#"$EXAMPLES_DIR"/}/"
}

main "$@"
