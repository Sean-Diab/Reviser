"""Recompute the paper's analytic FLOPs multipliers at the released decoders' settings (Figure 1, Appendix H).

Uses the appendix primitives (1 MAC = 1 unit, n = 128, d = 768, L = 12, |V| = 50,000). The earlier values are printed
first as a check; they reproduce. See notes/analytic-flops-settings-2026-10-13.md.
"""
# Recompute analytic FLOPs multipliers with the paper's primitives (1 MAC = 1 unit), corrected settings.
G = 1e9
n = 128; d = 768; L = 12; dff = 4 * d; V = 50000

def Fmult(a, b, c): return a * b * c
def Ffull(n_): return L * (4 * Fmult(n_, d, d) + 2 * Fmult(n_, n_, d) + 2 * Fmult(n_, d, dff))
def Fcausal(n_): return L * (4 * Fmult(n_, d, d) + n_ * (n_ + 1) * d + 2 * Fmult(n_, d, dff))
def Fvocab(m): return Fmult(m, d, V)
def Fxattn(n_, ns): return L * (Fmult(n_, d, d) + 2 * Fmult(ns, d, d) + Fmult(n_, d, d) + Fmult(n_, d, ns) + Fmult(n_, ns, d))

AR = Fcausal(n) + Fvocab(n)
print('AR(128) G', AR / G)
# --- checks against the printed values
print('Ffull128', Ffull(n) / G, 'Fxattn', Fxattn(n, n) / G, 'Fvocab128', Fvocab(n) / G)
print('MDLM no cache', 1000 * (Ffull(n) + Fvocab(n)) / AR)

# --- MDLM with the caching sampler: the network runs only when some token changed, plus the final noise-removal pass.
T = 1000
calls_cache = T * (1 - (1 - 1 / T) ** n) + 1
calls_fh = n + 1  # first-hitting: one token per call, plus noise removal
per_call = (Ffull(n) + Fvocab(n)) / AR
print('MDLM cache calls', calls_cache, 'ratio', calls_cache * per_call, 'first-hitting', calls_fh * per_call)

# --- Diffusion-LM official default: T=2000 with clamping every step (gamma=1)
Fround = Fmult(n, d, V)
def difflm(T, g): return (T * Ffull(n) + (1 + g * (T - 1)) * Fround) / AR
print('DiffLM T200 g0', difflm(200, 0), 'T2000 g0', difflm(2000, 0), 'T2000 g1', difflm(2000, 1))

# --- Encoder-decoder AR baseline for the MT rows (NAT, CMLM count an encoder and cross-attention)
AR_MT = Ffull(n) + Fcausal(n) + Fxattn(n, n) + Fvocab(n)
print('AR enc-dec G', AR_MT / G)
NAT = Ffull(n) + Fmult(n, d, 50) + Ffull(n) + Fxattn(n, n) + Fvocab(n)
print('NAT G', NAT / G, 'old ratio', NAT / AR, 'new ratio', NAT / AR_MT)
ms = [n] + [int(n * (10 - t) / 10) for t in range(1, 10)]
print('mask schedule', ms, sum(ms))
iter_sum = 10 * (Ffull(n) + Fxattn(n, n)) + sum(Fvocab(m) for m in ms)
for l in (1, 5):
    F = Ffull(n) + l * iter_sum
    print('CMLM l=%d' % l, 'G', F / G, 'old-baseline', F / AR, 'enc-dec baseline', F / AR_MT)

# --- LevT: decoder-side proxy, no early exit, R = mean iterations under the released adaptive stop (3.27) and the cap 10
n0 = 0.8 * n; m = 0.4 * n; Kmax = 16
it = Ffull(n0) + Ffull(n0) + Ffull(n) + Fmult(n0, d, 2) + Fmult(n0 + 1, 2 * d, Kmax + 1) + Fvocab(m)
print('LevT iter G', it / G)
for R in (3.27, 5, 10):
    print('LevT R=%s' % R, R * it / AR)

# --- Large models (gated MLP, GQA)
def full_g(n_, d_, L_, ff, dkv): return L_ * (2 * Fmult(n_, d_, d_) + 2 * Fmult(n_, d_, dkv) + 2 * Fmult(n_, n_, d_) + 3 * Fmult(n_, d_, ff))
def causal_g(n_, d_, L_, ff, dkv): return L_ * (2 * Fmult(n_, d_, d_) + 2 * Fmult(n_, d_, dkv) + n_ * (n_ + 1) * d_ + 3 * Fmult(n_, d_, ff))
# Dream-7B / Qwen2.5-7B
qd, qL, qff, qV, qkv = 3584, 28, 18944, 152064, 512
ARq = causal_g(n, qd, qL, qff, qkv) + Fmult(n, qd, qV)
print('AR Qwen7B G', ARq / G)
for T in (10, 20, 64, 128):
    F = T * full_g(n, qd, qL, qff, qkv) + (T + 1) / 2 * Fmult(n, qd, qV)
    print('Dream T=%d' % T, F / ARq)
# LLaDA-8B vs LLaMA3-8B
ld, lL, lff, lV, lkv = 4096, 32, 12288, 126464, 4096
ARl = causal_g(n, 4096, 32, 14336, 1024) + Fmult(n, 4096, 128256)
print('AR LLaMA3-8B G', ARl / G)
for T in (64, 128, 256):
    F = T * full_g(n, ld, lL, lff, lkv) + (T + 1) / 2 * Fmult(n, ld, lV)
    print('LLaDA full-seq T=%d' % T, F / ARl)

print('--- table values (G MAC, ratio)')
F = calls_cache * (Ffull(n) + Fvocab(n)); print('MDLM cache', round(F / G, 4), round(F / AR, 4))
F = 2000 * Ffull(n) + 2000 * Fround; print('DiffLM T2000 g1', round(F / G, 4), round(F / AR, 4))
for T in (128,):
    F = T * full_g(n, qd, qL, qff, qkv) + (T + 1) / 2 * Fmult(n, qd, qV); print('Dream T128', round(F / G, 4), round(F / ARq, 4))
for T in (64, 128):
    F = T * full_g(n, ld, lL, lff, lkv) + (T + 1) / 2 * Fmult(n, ld, lV); print('LLaDA T%d' % T, round(F / G, 4), round(F / ARl, 4))
F = 3.27 * it; print('LevT R3.27', round(F / G, 4), round(F / AR, 4))
print('AR_MT', round(AR_MT / G, 4), 'NAT', round(NAT / G, 4), round(NAT / AR_MT, 4))
for l in (1, 5):
    F = Ffull(n) + l * iter_sum; print('CMLM l%d' % l, round(F / G, 4), round(F / AR_MT, 4))
