# 触れ合ってよい配置の厳密な検査（2026-10-25）
# 使い方：python certify_touch.py 配置.json [solids_exact.json]
# 向こうの certify_general.py は隙間が正であることを区間演算で示すので、ちょうど触れ合う配置（隙間 0）は通らない。
# ここでは座標をすべて Q(√2) の正確な数で持ち、誤差なしに次を確かめる（等号＝触れているのは許す）：
#   (1) 中身：各駒のすべての頂点が、容器 s·C のすべての面の内側か面の上にある
#   (2) 重なり：どの 2 個にも、片方の頂点がすべて u·x ≤ t、もう片方がすべて u·x ≥ t となる平面 (u, t) がある
#       （凸体では、これが「内部が重ならない」ことと同じ）。u の候補は、両方の面の法線と、辺どうしの外積
# 配置の形式：{"piece", "container", "n", "s", "pieces": [{"c": [x, y, z], "q": [w, a, b, d]}, ...]}
#   数は "p/q"（有理数）か [a, b]（a + b√2、a と b は有理数の文字列）。回転は R = H(q)/|q|²（q の定数倍によらない）
# 立体は solids_exact.json の最小多項式が x² − 2 のもの（正四面体・正八面体・立方体など）に限る
import json, sys, os, itertools
from fractions import Fraction as Fr

class Q2:
    """a + b√2（a, b は有理数）"""
    __slots__ = ("a", "b")
    def __init__(s, a, b=0): s.a = Fr(a); s.b = Fr(b)
    def __add__(s, o): o = q2(o); return Q2(s.a + o.a, s.b + o.b)
    __radd__ = __add__
    def __sub__(s, o): o = q2(o); return Q2(s.a - o.a, s.b - o.b)
    def __rsub__(s, o): return q2(o) - s
    def __neg__(s): return Q2(-s.a, -s.b)
    def __mul__(s, o): o = q2(o); return Q2(s.a*o.a + 2*s.b*o.b, s.a*o.b + s.b*o.a)
    __rmul__ = __mul__
    def inv(s):
        d = s.a*s.a - 2*s.b*s.b
        if d == 0: raise ZeroDivisionError
        return Q2(s.a/d, -s.b/d)
    def __truediv__(s, o): return s*q2(o).inv()
    def sign(s):
        a, b = s.a, s.b
        if a >= 0 and b >= 0: return 0 if a == 0 and b == 0 else 1
        if a <= 0 and b <= 0: return -1
        t = a*a - 2*b*b                                   # a と b√2 の符号が逆：大きいほうの符号
        return (1 if t > 0 else -1) if a > 0 else (1 if t < 0 else -1)
    def __eq__(s, o): o = q2(o); return s.a == o.a and s.b == o.b
    def __hash__(s): return hash((s.a, s.b))
    def __float__(s): return float(s.a) + float(s.b)*2**0.5
    def __repr__(s): return f"{s.a}" if s.b == 0 else f"{s.a}+{s.b}√2"
def q2(x): return x if isinstance(x, Q2) else Q2(x)
def num(x): return Q2(x[0], x[1]) if isinstance(x, list) else Q2(x)
def dot(u, v): return u[0]*v[0] + u[1]*v[1] + u[2]*v[2]
def sub(u, v): return [u[k] - v[k] for k in range(3)]
def cross(u, v): return [u[1]*v[2] - u[2]*v[1], u[2]*v[0] - u[0]*v[2], u[0]*v[1] - u[1]*v[0]]
def rot(R, v): return [dot(R[k], v) for k in range(3)]
def iszero(u): return all(t == 0 for t in u)
def emin(xs):                                             # 正確な比較での最小・最大
    m = xs[0]
    for t in xs[1:]:
        if (t - m).sign() < 0: m = t
    return m
def emax(xs): return -emin([-t for t in xs])

def solid(name, SOL):
    e = SOL[name]
    if e.get("D") is not None or [Fr(c) for c in e["minpoly"]] not in ([-2, 0, 1], [-1, 1]):   # θ = √2 か θ = 1（立方体）
        raise ValueError(f"{name}：Q(√2) の立体ではない")
    th = Q2(0, 1) if len(e["minpoly"]) == 3 else Q2(1)
    V = [[Q2(Fr(c[0])) + (th*Fr(c[1]) if len(c) > 1 else 0) for c in v] for v in e["V"]]
    N, H = [], []
    for f in e["faces"]:
        nn = cross(sub(V[f[1]], V[f[0]]), sub(V[f[2]], V[f[0]])); h = dot(nn, V[f[0]])
        if h.sign() < 0: nn = [-t for t in nn]; h = -h       # 立体は原点を内部に含むので、h > 0 が外向き
        assert all(dot(nn, v) == h for v in (V[j] for j in f)), "面が平らでない"
        N.append(nn); H.append(h)
    E = sorted({tuple(sorted((f[k], f[(k+1) % len(f)]))) for f in e["faces"] for k in range(len(f))})
    return V, N, H, E

def quat(q):
    w, a, b, d = q; L = w*w + a*a + b*b + d*d
    H = [[w*w+a*a-b*b-d*d, 2*(a*b-w*d), 2*(a*d+w*b)],
         [2*(a*b+w*d), w*w-a*a+b*b-d*d, 2*(b*d-w*a)],
         [2*(a*d-w*b), 2*(b*d+w*a), w*w-a*a-b*b+d*d]]
    return [[t/L for t in r] for r in H]

def check(d, SOL, quiet=False):
    """(0 なら成り立つ, 1 行の結果)。回転は "q"（四元数）か "R"（3 × 3 の行列）で与える"""
    PV, PN, PH, PE = solid(d["piece"], SOL); CV, CN, CH = solid(d["container"], SOL)[:3]
    s = num(d["s"]); P = d["pieces"]; n = len(P); assert n == d["n"], "個数"
    Rs = [quat([num(t) for t in p["q"]]) if "q" in p else [[num(t) for t in r] for r in p["R"]] for p in P]
    cs = [[num(t) for t in p["c"]] for p in P]
    for i, R in enumerate(Rs):                            # H(q)/|q|² は正確に直交する。行列で与えたときはここで確かめる
        if not all(dot(R[a], R[b]) == (1 if a == b else 0) for a in range(3) for b in range(3)): return 1, f"回転が直交でない：駒 {i}"
        det = dot(R[0], cross(R[1], R[2]))
        if det != 1: return 1, f"回転の行列式が 1 でない：駒 {i}"
    r2 = emax([dot(v, v) for v in PV])                    # 外接球の半径の 2 乗。中心が 2r より離れた組は調べなくてよい
    X = [[[c[k] + t for k, t in enumerate(rot(R, v))] for v in PV] for c, R in zip(cs, Rs)]
    # (1) 中身
    wall_touch = 0
    for i in range(n):
        for x in X[i]:
            for nn, h in zip(CN, CH):
                g = (s*h - dot(nn, x)).sign()
                if g < 0: return 1, f"はみ出し：駒 {i}"
                wall_touch += g == 0
    # (2) 重なり
    pair_touch = 0; strict = 0; far = 0
    for i, j in itertools.combinations(range(n), 2):
        dc = sub(cs[i], cs[j])
        if (dot(dc, dc) - 4*r2).sign() > 0: far += 1; continue
        NA = [rot(Rs[i], t) for t in PN]; NB = [rot(Rs[j], t) for t in PN]
        DA = [sub(X[i][b], X[i][a]) for a, b in PE]; DB = [sub(X[j][b], X[j][a]) for a, b in PE]
        found = None
        for u in itertools.chain(NA, NB, (cross(p, q) for p in DA for q in DB)):
            if iszero(u): continue
            pa = [dot(u, x) for x in X[i]]; pb = [dot(u, x) for x in X[j]]
            for lo, hi in ((pa, pb), (pb, pa)):
                sg = (emin(hi) - emax(lo)).sign()
                if sg > 0: found = 1; break
                if sg == 0 and found is None: found = 0
            if found is not None: break                   # 隙間 0 の平面でも足りる（離れている組を「触れ合う」と数えることはある）
        if found is None: return 1, f"重なり：駒 {i} と {j}（分離する平面が見つからない）"
        if found == 1: strict += 1
        else: pair_touch += 1
    if not quiet:
        print(f"{d.get('record_id', '')}：{d['piece']} {n} 個 in s = {s} × {d['container']}")
        print(f"  中身：頂点と面の不等式 {n*len(PV)*len(CN)} 本、うち等号（面に触れている） {wall_touch} 本")
        print(f"  重なり：{n*(n-1)//2} 組のうち、中心が外接球 2 個分より離れた組 {far}、残りはすべて分離する平面あり"
              f"（最初に見つけた平面で隙間 0 の組 {pair_touch}、正の組 {strict}）")
        print("EXACT-VALID（触れ合いを許して、誤差なしで成り立つ）")
    return 0, f"EXACT-VALID 面 {wall_touch}・触れ合う組 {pair_touch}"

def main(fn, solfn):
    code, msg = check(json.load(open(fn, encoding="utf-8")), json.load(open(solfn, encoding="utf-8")))
    if code: print(msg)
    return code

if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    sys.exit(main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else os.path.join(here, "data", "src", "solids_exact.json")))
