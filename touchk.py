# 触れ合ってよい配置の厳密な検査と丸めを、多重 2 次体 Q(√2, √3, √5, …)（numfield.K）に広げたもの（2026-10-25）
#   python touchk.py <ID>    … data/closed_forms/<ID>.json の式と配置で、丸め → 検査。結果を 1 行
#   python touchk.py --all   … 式が多重 2 次体に入り、まだ EXACT-VALID でない記録すべて（CPU−4 並列、窓を出さない）。data/exact_touch/summary_k.tsv に書く
# certify_touch.check と round_touch の手順はそのまま使い、数の型だけ差し替える：
#   立体：solids_exact.json の θ（最小多項式が 1 次か 2 次）を K で表す（正十二面体・正二十面体は θ = (1 + √5)/2）
#   閉じた式：√n（n は整数）を含む式を K に。√(…) の入れ子や 3 乗根は扱えない
#   座標の格子：a + b√d（d は s と立体に現れる根号の積から 1 つ）。回転の成分：基底 {1, √a, √b, √ab} で PSLQ
import json, sys, os, re, glob, math, subprocess, itertools
from fractions import Fraction as Fr
import numpy as np, mpmath as mp
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from numfield import K
import certify_touch as T
import round_touch as RT
PP = os.path.join(HERE, "data"); OUT = os.path.join(PP, "exact_touch")

def solid_k(name, SOL):
    e = SOL[name]
    if e.get("D") is not None: raise ValueError(f"{name}：2 段の拡大体（D あり）")
    co = [Fr(c) for c in e["minpoly"]]
    if len(co) == 2: th = K(-co[0]/co[1])
    elif len(co) == 3:
        c0, c1, c2 = co; disc = c1*c1 - 4*c0*c2; lo, hi = (float(Fr(t)) for t in e["interval"])
        th = next(r for r in ((K(-c1) + K.sqrt(disc))/(2*c2), (K(-c1) - K.sqrt(disc))/(2*c2)) if lo - 1e-9 <= float(r) <= hi + 1e-9)
    else: raise ValueError(f"{name}：最小多項式が 3 次以上")
    V = [[sum((K(Fr(c))*th**0 if k == 0 else K(Fr(c))*pw(th, k) for k, c in enumerate(comp)), K(0)) for comp in v] for v in e["V"]]
    N, H = [], []
    for f in e["faces"]:
        nn = T.cross(T.sub(V[f[1]], V[f[0]]), T.sub(V[f[2]], V[f[0]])); h = T.dot(nn, V[f[0]])
        if h.sign() < 0: nn = [-t for t in nn]; h = -h
        assert all(T.dot(nn, v) == h for v in (V[j] for j in f)), "面が平らでない"
        N.append(nn); H.append(h)
    E = sorted({tuple(sorted((f[k], f[(k+1) % len(f)]))) for f in e["faces"] for k in range(len(f))})
    return V, N, H, E
def pw(x, k):
    r = K(1)
    for _ in range(k): r = r*x
    return r
K.__pow__ = lambda s, k: pw(s, k)

# certify_touch の数の型を K に差し替える（check・emin などは型によらない）
T.num = K.from_json; T.solid = solid_k; T.Q2 = lambda a=0, b=0: K(Fr(a)) + K(Fr(b))*K.sqrt(2); T.q2 = lambda x: x if isinstance(x, K) else K(x)

def parse_k(cf):
    if not cf or re.search(r"\*\*|I|√\((?!\d+\))", cf): return None
    e = re.sub(r"√\((\d+)\)", r"S(\1)", cf); e = re.sub(r"√(\d+)", r"S(\1)", e)
    if "√" in e: return None
    e = re.sub(r"(?<![\w(])(\d+)", lambda m: f"F({m.group(1)})", e).replace("S(F(", "S((")
    try: return eval(e, {"F": lambda k: K(k), "S": lambda k: K.sqrt(k)})
    except Exception: return None

def field_primes(SOL, *names):
    ps = set()
    for nm in names:
        co = [Fr(c) for c in SOL[nm]["minpoly"]]
        if len(co) == 3: ps |= K.sqrt(co[1]**2 - 4*co[0]*co[2]).primes()
    return ps

BASES = [[d] for d in (2, 3, 5, 6, 10, 15)] + [[2, 3, 6], [2, 5, 10], [3, 5, 15]]
def recog_rot_all(x):
    """厳しい条件で認識し、だめなら少し緩める（記録の回転は 1e-9 程度の精度のことがある）"""
    return recog_rot(x) or recog_rot(x, 1e-8, 40)
def snap_rot_k(Rf):
    """ほぼ揃った向きの回転（遊んでいる駒）を、成分の高さの小さい回転に丸める（1/√3、(1+√5)/4 なども含む）。直交のときだけ返す"""
    R = [[recog_rot(t, 2e-4, 12) for t in row] for row in Rf]
    if any(t is None for row in R for t in row): return None
    if not all(T.dot(R[a], R[b]) == (1 if a == b else 0) for a in range(3) for b in range(3)): return None
    if T.dot(R[0], T.cross(R[1], R[2])) != 1: return None
    return R
def recog_rot(x, tol=1e-10, maxc=120):
    """回転の成分 x を (p + Σ q_d √d)/r として認識（基底を順に試し、高さ最小）。K or None"""
    if abs(x - round(x*64)/64) < tol: return K(Fr(round(x*64), 64))
    best = None; mp.mp.dps = 30
    for ds in BASES:
        rel = mp.pslq([mp.mpf(x), 1] + [mp.sqrt(d) for d in ds], tol=tol, maxcoeff=maxc if len(ds) == 1 else maxc//3, maxsteps=5000)
        if rel and rel[0]:
            v = (K(-rel[1]) + sum((K(-c)*K.sqrt(d) for c, d in zip(rel[2:], ds)), K(0)))/K(rel[0])
            if abs(float(v) - x) < tol*10:
                h = max(abs(t) for t in rel)
                if best is None or h < best[0]: best = (h, v)
    return best[1] if best else None

def grid_k(xs, Ds, d, tol=3e-7, bmax=4, Gmix=200):
    """round_touch.grid の √2 を √d にしたもの。(形, G, [K], 外れた数, 最大のずれ)"""
    rd = math.sqrt(d); X = np.array(xs, float); N = len(xs); cand = []
    for G in sorted({k for k in range(1, 49)} | {k*Ds for k in range(1, 49)}):
        for kind, v in (("rat", X), ("sqd", X/rd)): cand.append((int((np.abs(np.round(v*G)/G - v) < tol).sum()), G, kind))
        if G <= Gmix:
            A = X[:, None] - (np.arange(-bmax*G, bmax*G + 1)/G*rd)[None, :]
            cand.append((int((np.abs(np.round(A*G)/G - A).min(1) < tol).sum()), G, "mix"))
    top = max(c[0] for c in cand); order = {"rat": 0, "sqd": 1, "mix": 2}
    near = sorted((c for c in cand if c[0] >= top - max(1, N//100)), key=lambda c: (c[1], order[c[2]]))
    rest = sorted((c for c in cand if c[0] < top - max(1, N//100)), key=lambda c: (-c[0], c[1], order[c[2]]))
    for (fit, G, kind), fine in itertools.product((near + rest)[:4], (False, True)):
        # 格子から外れた座標（滑れる駒）：fine = False なら b = 0 として a を格子 1/G に（±1 の係数の関係を保つ）、
        # True なら格子 1/(50G) で誤差最小の a + b√d に（動かす量を小さくする）
        out, free, worst = [], 0, 0.0; Gf = 50*G
        for x in xs:
            if kind == "rat": a, b = Fr(round(x*G), G), Fr(0)
            elif kind == "sqd": a, b = Fr(0), Fr(round(x/rd*G), G)
            else:
                bs = np.arange(-bmax*G, bmax*G + 1); A = x - bs/G*rd; e = np.abs(np.round(A*G)/G - A); k = int(e.argmin())
                a, b = (Fr(round(A[k]*G), G), Fr(int(bs[k]), G)) if e[k] < tol else (Fr(round(x*G), G), Fr(0))
            if fine and abs(float(a) + float(b)*rd - x) >= tol:
                if kind == "rat": a, b = Fr(round(x*Gf), Gf), Fr(0)
                elif kind == "sqd": a, b = Fr(0), Fr(round(x/rd*Gf), Gf)
                else:
                    bs = np.arange(-bmax*Gf, bmax*Gf + 1); A = x - bs/Gf*rd; e = np.abs(np.round(A*Gf)/Gf - A); k = int(e.argmin())
                    a, b = Fr(round(A[k]*Gf), Gf), Fr(int(bs[k]), Gf)
            err = abs(float(a) + float(b)*rd - x)
            if err < tol: worst = max(worst, err)
            else: free += 1
            out.append(K(a) + K(b)*K.sqrt(d))
        if fine and free == 0: continue                   # 外れた座標がなければ同じ結果
        yield kind, G, out, free, worst, fine

# ---------- 接触の一次方程式で中心を正確に解く ----------
# 回転と s を正確な数に決めると、接触の条件はどれも中心の一次式になる：
#   壁：n·(c + R v) = s h、頂点と面：N_B·(c_A + R_A v − c_B) = H（N_B = R_B n）、辺と辺：m·((c_A + R_A p) − (c_B + R_B q)) = 0（m = R_A d × R_B e）
# 浮動小数の配置で接触を拾い（隙間 < TOLC）、ガウス・ジョルダンで解く。自由な変数（滑れる向き）は浮動小数の値を分母 L の分数に丸めて決める
TOLC = 1e-6
def gj_k(eqs, ncol):
    piv = {}
    for r in eqs:
        r = {c: v for c, v in r.items() if v != 0}
        while True:
            hit = next((c for c in r if c in piv), None)
            if hit is None: break
            f = r[hit]
            for c, v in piv[hit].items():
                t = r.get(c, K(0)) - f*v
                if t == 0: r.pop(c, None)
                else: r[c] = t
        cols = [c for c in r if c != ncol]
        if not cols:
            if ncol in r: return None
            continue
        pc = min(cols); inv = r[pc].inv(); r = {c: v*inv for c, v in r.items()}
        for q, row in piv.items():
            if pc in row:
                f = row[pc]
                for c, v in r.items():
                    t = row.get(c, K(0)) - f*v
                    if t == 0: row.pop(c, None)
                    else: row[c] = t
        piv[pc] = r
    return piv

def exact_centers(cf_float, Rk, s, SOL, piece, cont, L=2**16, skip=()):
    PV, PN, PH, PE = solid_k(piece, SOL); CV, CN, CH = solid_k(cont, SOL)[:3]; faces = SOL[piece]["faces"]
    n = len(cf_float); f = lambda v: np.array([float(t) for t in v])
    Rf = [np.array([[float(t) for t in row] for row in R]) for R in Rk]; Vf = np.array([f(v) for v in PV])
    Nf = np.array([f(v) for v in PN]); Hf = np.array([float(h) for h in PH]); CNf = np.array([f(v) for v in CN]); CHf = np.array([float(h) for h in CH])
    W = [np.array(c) + Vf @ R.T for c, R in zip(cf_float, Rf)]
    rv = [[T.rot(R, v) for v in PV] for R in Rk]                     # 正確な R v
    eqs = []; ncol = 3*n
    def addeq(coef, rhs):                                            # coef: {列: K}、右辺 rhs
        if any(c//3 in skip for c in coef): return                      # 回転を近似した駒の接触は使わない（その駒は浮動小数の位置を丸める）
        r = {c: v for c, v in coef.items() if v != 0}; r[ncol] = rhs; eqs.append(r)
    sf = float(s)
    for i in range(n):
        for k, x in enumerate(W[i]):
            for g in range(len(CN)):
                if abs(sf*CHf[g]/np.linalg.norm(CNf[g]) - CNf[g] @ x/np.linalg.norm(CNf[g])) < TOLC:
                    addeq({3*i + j: CN[g][j] for j in range(3)}, s*CH[g] - T.dot(CN[g], rv[i][k]))
    rad = float(np.linalg.norm(Vf, axis=1).max())
    for A, B in itertools.combinations(range(n), 2):
        if np.linalg.norm(np.array(cf_float[A]) - np.array(cf_float[B])) > 2*rad + 1e-6: continue
        for a_, b_ in ((A, B), (B, A)):
            for k, x in enumerate(W[a_]):
                for g, fc in enumerate(faces):
                    Nw = Rf[b_] @ Nf[g]; nn = np.linalg.norm(Nw)
                    if abs(Nw @ (x - cf_float[b_]) - Hf[g])/nn > TOLC: continue
                    P = W[b_][fc]; inside = all(np.dot(Nw, np.cross(P[(t+1) % len(fc)] - P[t], x - P[t])) > -TOLC*nn for t in range(len(fc))) or                                             all(np.dot(Nw, np.cross(P[(t+1) % len(fc)] - P[t], x - P[t])) < TOLC*nn for t in range(len(fc)))
                    if not inside: continue
                    NB = T.rot(Rk[b_], PN[g])
                    addeq({**{3*a_ + j: NB[j] for j in range(3)}, **{3*b_ + j: -NB[j] for j in range(3)}}, PH[g] - T.dot(NB, rv[a_][k]))
        for (i0, i1), (j0, j1) in itertools.product(PE, PE):
            p0, d0 = W[A][i0], W[A][i1] - W[A][i0]; q0, e0 = W[B][j0], W[B][j1] - W[B][j0]
            m = np.cross(d0, e0); mn = np.linalg.norm(m)
            if mn < 1e-6: continue
            if abs(m @ (p0 - q0))/mn > TOLC: continue
            lam = np.dot(np.cross(q0 - p0, e0), m)/mn**2; mu = np.dot(np.cross(q0 - p0, d0), m)/mn**2
            if not (-1e-9 < lam < 1 + 1e-9 and -1e-9 < mu < 1 + 1e-9): continue
            mk = T.cross(T.sub(rv[A][i1], rv[A][i0]), T.sub(rv[B][j1], rv[B][j0]))
            addeq({**{3*A + j: mk[j] for j in range(3)}, **{3*B + j: -mk[j] for j in range(3)}}, -T.dot(mk, T.sub(rv[A][i0], rv[B][j0])))
    piv = gj_k(eqs, ncol)
    if piv is None: return None, len(eqs), None
    flat = [x for c in cf_float for x in c]
    free = {c: K(Fr(round(flat[c]*L), L)) for c in range(ncol) if c not in piv}
    val = dict(free)
    for c, r in piv.items(): val[c] = r.get(ncol, K(0)) - sum((v*free[q] for q, v in r.items() if q != ncol and q != c), K(0))
    dev = max(abs(float(val[c]) - flat[c]) for c in range(ncol))
    return [[val[3*i + j] for j in range(3)] for i in range(n)], len(eqs), (len(free), dev)

def one(rid):
    cf = json.load(open(os.path.join(PP, "closed_forms", rid + ".json"), encoding="utf-8"))
    s = parse_k(cf.get("closed_form"))
    if s is None: return "式が多重 2 次体でない", ""
    SOL = json.load(open(os.path.join(PP, "src", "solids_exact.json"), encoding="utf-8"))
    try: field_primes(SOL, cf["piece"], cf["container"]); solid_k(cf["piece"], SOL); solid_k(cf["container"], SOL)
    except Exception as ex: return "立体を扱えない", str(ex)
    fn = RT.config_for(rid, float(s))
    if not fn: return "配置ファイルがない", ""
    rec = json.load(open(fn, encoding="utf-8")); lam = float(rec["s_full"])/float(s); cc = rec.get("container_center", [0, 0, 0])
    pieces, snapped = [], 0; ratq = set()
    for i, p in enumerate(rec["pieces"]):
        Rf = RT.quatR(p[3:7]); R = [[recog_rot_all(t) for t in row] for row in Rf]
        ok = not any(t is None for row in R for t in row) and all(T.dot(R[a], R[b]) == (1 if a == b else 0) for a in range(3) for b in range(3))
        if not ok:
            R2 = snap_rot_k(Rf)
            if R2 is None:                                     # 0, ±1, ±1/2, ±√2/2 への丸め（ずれ 0.15 まで）
                Rs = RT.snap_rot(Rf); R2 = [[K.from_json(t) for t in row] for row in Rs] if Rs else None
            if R2 is None:                                     # 最後の手：四元数を分数に近似（H(q)/|q|² は正確に直交）。遊んでいる駒向け
                q = [Fr(t).limit_denominator(10**7) for t in p[3:7]]; w_, a_, b_, d_ = q; L_ = w_*w_ + a_*a_ + b_*b_ + d_*d_
                H_ = [[w_*w_+a_*a_-b_*b_-d_*d_, 2*(a_*b_-w_*d_), 2*(a_*d_+w_*b_)], [2*(a_*b_+w_*d_), w_*w_-a_*a_+b_*b_-d_*d_, 2*(b_*d_-w_*a_)],
                      [2*(a_*d_-w_*b_), 2*(b_*d_+w_*a_), w_*w_-a_*a_-b_*b_+d_*d_]]
                R2 = [[K(t/L_) for t in row] for row in H_]; ratq.add(i)
            R = R2; snapped += 1
        pieces.append(dict(c=[(p[k] - cc[k])/lam for k in range(3)], R=R))
    # まず接触の一次方程式で中心を正確に解く
    cen, neq, fr_ = exact_centers([p["c"] for p in pieces], [p["R"] for p in pieces], s, SOL, rec["piece"], rec["container"], skip=ratq)
    if cen is not None:
        dd = dict(record_id=rid, piece=rec["piece"], container=rec["container"], n=len(pieces), s=s.to_json(), source=os.path.relpath(fn, HERE),
                  closed_form=cf["closed_form"], pieces=[dict(c=[t.to_json() for t in c], R=[[t.to_json() for t in row] for row in p["R"]]) for c, p in zip(cen, pieces)])
        code, msg = T.check(dd, SOL, quiet=True)
        info = f"接触の一次方程式 {neq} 本で中心を正確に解いた（自由な座標 {fr_[0]}、浮動小数からのずれ {fr_[1]:.1e}）、回転を丸めた駒 {snapped}（うち四元数の分数近似 {len(ratq)}）"
        if code == 0:
            os.makedirs(OUT, exist_ok=True); json.dump(dd, open(os.path.join(OUT, rid + ".json"), "w", encoding="utf-8"), indent=1)
            return "EXACT-VALID", info
        lin_fail = msg + "。" + info
    else: lin_fail = f"接触の一次方程式 {neq} 本が矛盾した"
    # 座標の √d の候補：s と立体に現れる根号（1 つずつと、その積）
    ps = sorted(s.primes() | field_primes(SOL, cf["piece"], cf["container"])) or [2]
    ds = sorted({math.prod(c) for r in range(1, len(ps) + 1) for c in itertools.combinations(ps, r)})
    Ds = math.lcm(*[c.denominator for c in s.t.values()]) if s.t else 1
    xs = [x for p in pieces for x in p["c"]]; first = None
    for d in ds:
        for kind, G, vals, free, worst, fine in grid_k(xs, Ds, d):
            it = iter(vals)
            dd = dict(record_id=rid, piece=rec["piece"], container=rec["container"], n=len(pieces), s=s.to_json(), source=os.path.relpath(fn, HERE),
                      closed_form=cf["closed_form"], pieces=[dict(c=[next(it).to_json() for _ in range(3)], R=[[t.to_json() for t in row] for row in p["R"]]) for p in pieces])
            code, msg = T.check(dd, SOL, quiet=True)
            info = f"座標 a + b√{d}（{kind}、格子 1/{G}{'、外れた座標は細かい格子' if fine else ''}）、ずれの最大 {worst:.1e}、格子から外れた座標 {free}、回転を丸めた駒 {snapped}"
            if code == 0 or first is None: first = (dd, code, msg, info)
            if code == 0: break
        if first[1] == 0: break
    dd, code, msg, info = first
    os.makedirs(OUT, exist_ok=True); json.dump(dd, open(os.path.join(OUT, rid + ".json"), "w", encoding="utf-8"), indent=1)
    return ("EXACT-VALID" if code == 0 else msg.split("：")[0]), (msg + "。" if code else "") + info + ("" if code == 0 else f"［一次方程式：{lin_fail}］")

def targets():
    done = set()
    p = os.path.join(OUT, "summary.tsv")
    if os.path.exists(p):
        done = {l.split("\t")[0] for l in open(p, encoding="utf-8") if "\tEXACT-VALID\t" in l}
    SOL = json.load(open(os.path.join(PP, "src", "solids_exact.json"), encoding="utf-8")); out = []
    for f in sorted(glob.glob(os.path.join(PP, "closed_forms", "*.json"))):
        d = json.load(open(f, encoding="utf-8")); rid = d["record"]
        if rid in done or rid.startswith("ours_") or str(d.get("determined")) != "True" or parse_k(d.get("closed_form")) is None: continue
        if any(SOL[x].get("D") is not None or len(SOL[x]["minpoly"]) > 3 for x in (d["piece"], d["container"])): continue
        out.append(rid)
    return out

if __name__ == "__main__":
    if sys.argv[1] != "--all":
        st, info = one(sys.argv[1]); print(f"{sys.argv[1]}\t{st}\t{info}"); sys.exit(0)
    from concurrent.futures import ThreadPoolExecutor
    NOWIN = getattr(subprocess, "CREATE_NO_WINDOW", 0); ids = targets(); print(len(ids), "件", flush=True)
    def run(rid):
        try:
            o = subprocess.run([sys.executable, "-I", "-X", "utf8", os.path.abspath(__file__), rid], capture_output=True, text=True,
                               encoding="utf-8", timeout=3600, creationflags=NOWIN)
            line = (o.stdout.strip().splitlines() or [f"{rid}\t異常終了\t{(o.stderr.strip().splitlines() or [''])[-1]}"])[-1]
        except subprocess.TimeoutExpired: line = f"{rid}\t時間切れ\t"
        print(line, flush=True); return line
    with ThreadPoolExecutor(max(1, os.cpu_count() - 4)) as ex: res = list(ex.map(run, ids))
    with open(os.path.join(OUT, "summary_k.tsv"), "w", encoding="utf-8") as w:
        w.write("record\tstatus\tinfo\n"); w.writelines(l + "\n" for l in res)
