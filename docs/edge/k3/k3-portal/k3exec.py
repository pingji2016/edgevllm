#!/usr/bin/env python3
# 通过「K3 门户」的网页终端(ttyd)在板子上执行一条命令。
# 用法:
#   curl -s -c /tmp/k3_cookies.txt -X POST http://127.0.0.1:3000/login -d 'password=<口令>'
#   python3 k3exec.py 'uname -a; free -h' 10
import socket, base64, os, struct, sys, time, json

cookie = None
for line in open('/tmp/k3_cookies.txt'):
    if line.startswith('#') and not line.startswith('#HttpOnly_'):
        continue
    p = line.split()
    if len(p) >= 7 and p[5] == 'k3':
        cookie = 'k3=' + p[6]
if cookie is None:
    sys.exit('no cookie in /tmp/k3_cookies.txt (先 curl -c 登录)')

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
if b' 101 ' not in rx.split(b'\r\n')[0]:
    sys.exit('HANDSHAKE FAIL: ' + rx.split(b'\r\n')[0].decode())
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

# 关键：第一帧必须是认证 JSON（二进制、无前缀），否则一敲就断
ws(json.dumps({"AuthToken": "", "columns": 200, "rows": 50}).encode())
pump(2.5)
ws(b'0' + (cmd + '\r').encode())
pump(wait)
ws(b'0' + b'exit\r')
s.close()
print(buf.decode('utf-8', 'replace'))
