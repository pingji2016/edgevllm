import sys, wave, numpy as np

def f0s(path):
    with wave.open(path) as w:
        sr = w.getframerate()
        x = np.frombuffer(w.readframes(w.getnframes()), dtype='<i2').astype(np.float64)/32768.0
    N = int(0.04*sr); hop = int(0.02*sr)
    lo, hi = int(sr/400), int(sr/70)   # 70-400Hz
    vals = []
    for i in range(0, len(x)-N, hop):
        f = x[i:i+N]
        if np.sqrt((f**2).mean()) < 0.02: continue
        f = f - f.mean()
        c = np.correlate(f, f, 'full')[N-1:]
        if c[0] <= 0: continue
        c = c / c[0]
        seg = c[lo:hi]
        if len(seg) == 0: continue
        k = int(np.argmax(seg)) + lo
        if c[k] > 0.3:
            vals.append(sr/k)
    return np.median(vals) if vals else float('nan'), len(vals)

for p in sys.argv[1:]:
    m, n = f0s(p)
    print(f"{p:16s} f0={m:6.1f} Hz   voiced_frames={n}")
