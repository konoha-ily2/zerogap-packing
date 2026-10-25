# 多重 2 次体 Q(√p₁, √p₂, …)（p は素数）の正確な数（2026-10-25）
# 元は Σ c_S √(Π_{p∈S} p)（S は素数の集合、c_S は有理数）。√S·√T = (Π_{p∈S∩T} p)·√(S△T) なので、和・積は閉じる。
# 逆数は共役（√p の符号を反転）を順に掛けて有理数にして求める。符号は、0 でなければ √m の整数の上下限（isqrt）で
# 区間を狭めて決める（基底 √(Π p) は Q 上一次独立なので、係数がすべて 0 のときだけ 0）
from fractions import Fraction as Fr
from math import isqrt

def _sqfree(n):
    """n = m²·f（f は平方因子なし）の (m, f の素因数の集合)"""
    m, ps, p = 1, set(), 2
    while p*p <= n:
        while n % (p*p) == 0: n //= p*p; m *= p
        if n % p == 0: n //= p; ps.add(p)
        p += 1
    if n > 1: ps.add(n)
    return m, frozenset(ps)

class K:
    __slots__ = ("t",)
    def __init__(s, a=0, t=None):
        s.t = {k: v for k, v in t.items() if v != 0} if t is not None else ({frozenset(): Fr(a)} if Fr(a) != 0 else {})
    @staticmethod
    def sqrt(n):
        n = Fr(n)
        if n.denominator != 1:                                  # √(p/q) = √(pq)/q
            return K.sqrt(n.numerator*n.denominator)*K(Fr(1, n.denominator))
        m, ps = _sqfree(int(n)); return K(t={ps: Fr(m)})
    def __add__(s, o):
        o = k_(o); t = dict(s.t)
        for k, v in o.t.items(): t[k] = t.get(k, 0) + v
        return K(t=t)
    __radd__ = __add__
    def __neg__(s): return K(t={k: -v for k, v in s.t.items()})
    def __sub__(s, o): return s + (-k_(o))
    def __rsub__(s, o): return k_(o) - s
    def __mul__(s, o):
        o = k_(o); t = {}
        for k1, v1 in s.t.items():
            for k2, v2 in o.t.items():
                c = v1*v2
                for p in k1 & k2: c *= p
                k = k1 ^ k2; t[k] = t.get(k, 0) + c
        return K(t=t)
    __rmul__ = __mul__
    def conj(s, p): return K(t={k: (-v if p in k else v) for k, v in s.t.items()})
    def primes(s): return set().union(*s.t.keys()) if s.t else set()
    def inv(s):
        if not s.t: raise ZeroDivisionError
        y, acc = s, K(1)
        while y.primes():
            p = min(y.primes()); c = y.conj(p); acc = acc*c; y = y*c
        return acc*K(1/y.t[frozenset()])
    def __truediv__(s, o): return s*k_(o).inv()
    def __rtruediv__(s, o): return k_(o)*s.inv()
    def __eq__(s, o): return (s - k_(o)).t == {}
    def __hash__(s): return hash(frozenset(s.t.items()))
    def sign(s):
        if not s.t: return 0
        bits = 64
        while True:
            D = 1 << bits; lo = hi = Fr(0)
            for k, c in s.t.items():
                m = 1
                for p in k: m *= p
                r = isqrt(m*D*D); a, b = Fr(r, D), Fr(r + (r*r != m*D*D), D)    # a ≤ √m ≤ b
                lo += c*a if c > 0 else c*b; hi += c*b if c > 0 else c*a
            if lo > 0: return 1
            if hi < 0: return -1
            bits *= 2
    def __float__(s):
        x = 0.0
        for k, c in s.t.items():
            m = 1
            for p in k: m *= p
            x += float(c)*m**0.5
        return x
    def __repr__(s):
        if not s.t: return "0"
        out = []
        for k, c in sorted(s.t.items(), key=lambda kv: sorted(kv[0])):
            m = 1
            for p in k: m *= p
            out.append(f"{c}" if not k else f"{c}√{m}")
        return " + ".join(out)
    def to_json(s):
        """[[係数, 根号の中の整数], ...]"""
        out = []
        for k, c in s.t.items():
            m = 1
            for p in k: m *= p
            out.append([str(c), m])
        return out
    @staticmethod
    def from_json(x):
        if isinstance(x, (str, int)): return K(Fr(x))
        if isinstance(x, list) and len(x) == 2 and not isinstance(x[0], list): return K(Fr(x[0])) + K(Fr(x[1]))*K.sqrt(2)  # 旧形式 [a, b] = a + b√2
        return sum((K(Fr(c))*K.sqrt(m) for c, m in x), K(0))

def k_(x): return x if isinstance(x, K) else K(x)

if __name__ == "__main__":
    a = K(1) + K.sqrt(2); b = K.sqrt(3) - K(Fr(1, 2))*K.sqrt(5)
    assert (a*a.inv()) == 1 and (b*b.inv()) == 1 and ((a*b)/b) == a
    phi = (K(1) + K.sqrt(5))/2; assert phi*phi - phi - 1 == 0
    assert (K.sqrt(2)*K.sqrt(3)) == K.sqrt(6) and K.sqrt(8) == 2*K.sqrt(2) and K.sqrt(Fr(1, 2)) == K.sqrt(2)/2
    assert (K.sqrt(2) + K.sqrt(3) - K.sqrt(5) - K(Fr(3, 2))).sign() == -1       # 1.4142 + 1.7321 − 2.2361 − 1.5 = −0.59…
