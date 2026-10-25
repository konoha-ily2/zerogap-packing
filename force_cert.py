# 局所最適性の一次の証明書（力のつり合い）を作って、正確な数で確かめる（2026-10-25）
# 使い方：python force_cert.py 配置.json   … certify_touch.py と同じ形式（Q(√2) の正確な座標、触れ合う配置）
#
# 動き：箱の大きさの変化 ds と、各駒の平行移動 t・回転 ω（無限小）。触れ合いの隙間 g ≥ 0 を一次で近似した δg = a·(動き) + w ds。
# 力 λ ≥ 0 で Σ λ_i a_i = 0（どの駒にも力とねじる力が残らない）、Σ λ_i w_i = 1 なら、δg ≥ 0 を守る動きは必ず ds = Σ λ_i δg_i ≥ 0。
# つまり「一次では箱を縮められない」。λ がなければ、一次で縮める動きがある（ファルカスの補題）。
#
# 使う触れ合い（駒どうしが重ならないことから、局所的に必ず従う条件だけ。そうでないものは外す＝条件を弱める側なので、通れば正しい）：
#   壁：頂点が容器の面の上（容器は凸なので、どの頂点にも必ず成り立つ）
#   頂点と面：A の頂点が B の面の内部（境界を除く）にある
#   辺と辺：両方の辺の内部どうしが交わる（平行でない）
# 外す（退化した）触れ合い：頂点が相手の面の境界（辺・頂点）の上、平行な辺の重なり。これらは「どちらかの面の外」という
# 場合分けになり、一次式の条件にならない
import json, sys, os, itertools
from fractions import Fraction as Fr
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import certify_touch as T
from certify_touch import Q2, dot, sub, cross, rot

def contacts(d, SOL):
    PV, PN, PH, PE = T.solid(d["piece"], SOL); CV, CN, CH = T.solid(d["container"], SOL)[:3]
    faces = SOL[d["piece"]]["faces"]; s = T.num(d["s"]); P = d["pieces"]; n = len(P)
    Rs = [T.quat([T.num(t) for t in p["q"]]) if "q" in p else [[T.num(t) for t in r] for r in p["R"]] for p in P]
    cs = [[T.num(t) for t in p["c"]] for p in P]
    X = [[[c[k] + t for k, t in enumerate(rot(R, v))] for v in PV] for c, R in zip(cs, Rs)]
    r2 = T.emax([dot(v, v) for v in PV])
    rows, degen = [], {"面の境界の上の頂点": 0, "平行な辺の重なり・端どうし": 0}
    def row(): return [Q2(0)]*(1 + 6*n)
    # 壁
    for i in range(n):
        for x in X[i]:
            for nn, h in zip(CN, CH):
                if s*h - dot(nn, x) == 0:
                    a = row(); a[0] = h; r = sub(x, cs[i])
                    for k in range(3): a[1+6*i+k] = -nn[k]
                    for k, t in enumerate(cross(r, nn)): a[4+6*i+k] = -t
                    rows.append(("壁", i, a, (x, nn)))
    touching_pairs, nondeg_pairs = set(), set()
    for A, B in itertools.combinations(range(n), 2):
        dc = sub(cs[A], cs[B])
        if (dot(dc, dc) - 4*r2).sign() > 0: continue
        # 頂点と面（両向き）
        for a_, b_ in ((A, B), (B, A)):
            NB = [rot(Rs[b_], t) for t in PN]
            for p in X[a_]:
                for g, (N, H) in enumerate(zip(NB, PH)):
                    if dot(N, sub(p, cs[b_])) - H != 0: continue
                    f = faces[g]; fv = [X[b_][j] for j in f]
                    sg = [dot(N, cross(sub(fv[(k+1) % len(f)], fv[k]), sub(p, fv[k]))).sign() for k in range(len(f))]
                    if any(t < 0 for t in sg) and any(t > 0 for t in sg): continue      # 面の平面の上だが、面の外
                    touching_pairs.add((A, B))
                    if 0 in sg: degen["面の境界の上の頂点"] += 1; continue
                    nondeg_pairs.add((A, B)); a = row(); rp = sub(p, cs[a_]); rb = sub(p, cs[b_])
                    for k in range(3): a[1+6*a_+k] = N[k]; a[1+6*b_+k] = -N[k]
                    for k, t in enumerate(cross(rp, N)): a[4+6*a_+k] = t
                    for k, t in enumerate(cross(N, rb)): a[4+6*b_+k] = t
                    rows.append(("頂点と面", (a_, b_), a, (p, N, H)))
        # 面と面（平らに重なる）・辺が面の上に乗る：重なりの領域の内部では、A が B の面の平面を割り込まないことが局所的に必ず従い、
        # 動きの速度は点について一次なので、領域の閉包（多角形・線分の端点）でも成り立つ。その端点を頂点と面の接触と同じ式で加える
        for a_, b_ in ((A, B), (B, A)):
            NA_ = [rot(Rs[a_], t) for t in PN]; NB_ = [rot(Rs[b_], t) for t in PN]
            for gb, fb in enumerate(faces):
                N, H = NB_[gb], PH[gb]; fbv = [X[b_][j] for j in fb]
                onp = [dot(N, sub(y, cs[b_])) - H == 0 for y in X[a_]]
                if sum(onp) < 2: continue
                cen = [sum((v[k] for v in fbv), Q2(0))*Q2(Fr(1, len(fb))) for k in range(3)]
                sig = dot(N, cross(sub(fbv[1], fbv[0]), sub(cen, fbv[0]))).sign()
                def inside_val(y, k): return dot(N, cross(sub(fbv[(k+1) % len(fb)], fbv[k]), sub(y, fbv[k])))*sig
                def clip(poly):
                    for k in range(len(fb)):
                        out = []
                        for t in range(len(poly)):
                            P_, Q_ = poly[t], poly[(t+1) % len(poly)]; vp, vq = inside_val(P_, k), inside_val(Q_, k)
                            if vp.sign() >= 0: out.append(P_)
                            if vp.sign()*vq.sign() < 0:
                                tt = vp/(vp - vq); out.append([P_[j] + (Q_[j] - P_[j])*tt for j in range(3)])
                        poly = out
                        if not poly: break
                    return poly
                pts = []
                for ga, fa in enumerate(faces):                         # A の面が B の面と同じ平面で向かい合う
                    if all(onp[j] for j in fa):
                        poly = clip([X[a_][j] for j in fa]); uniq = []
                        for y in poly:
                            if not any(all(y[j] == z[j] for j in range(3)) for z in uniq): uniq.append(y)
                        if len(uniq) >= 3 and not T.iszero(cross(sub(uniq[1], uniq[0]), sub(uniq[2], uniq[0]))): pts += uniq
                for (i0, i1) in PE:                                     # A の辺が B の面の平面の上
                    if onp[i0] and onp[i1]:
                        seg = clip([X[a_][i0], X[a_][i1]])
                        if len(seg) >= 2 and not T.iszero(sub(seg[0], seg[1])):
                            mid = [(seg[0][j] + seg[1][j])*Q2(Fr(1, 2)) for j in range(3)]
                            if all(inside_val(mid, k).sign() > 0 for k in range(len(fb))): pts += seg[:2]
                for p in pts:
                    touching_pairs.add((A, B)); nondeg_pairs.add((A, B)); a = row(); rp = sub(p, cs[a_]); rb = sub(p, cs[b_])
                    for k in range(3): a[1+6*a_+k] = N[k]; a[1+6*b_+k] = -N[k]
                    for k, t in enumerate(cross(rp, N)): a[4+6*a_+k] = t
                    for k, t in enumerate(cross(N, rb)): a[4+6*b_+k] = t
                    rows.append(("面の重なりの端", (a_, b_), a, (p, N, H)))
        # 辺と辺
        for (i0, i1), (j0, j1) in itertools.product(PE, PE):
            p0, q0 = X[A][i0], X[B][j0]; dd, ee = sub(X[A][i1], p0), sub(X[B][j1], q0); m = cross(dd, ee); w = sub(p0, q0)
            if T.iszero(m):
                continue
            if dot(w, m) != 0: continue
            mm = dot(m, m); lam = dot(cross(sub(q0, p0), ee), m)/mm; mu = dot(cross(sub(q0, p0), dd), m)/mm
            inside = [lam.sign() > 0, (1 - lam).sign() > 0, mu.sign() > 0, (1 - mu).sign() > 0]
            onseg = [lam.sign() >= 0, (1 - lam).sign() >= 0, mu.sign() >= 0, (1 - mu).sign() >= 0]
            if not all(onseg): continue
            touching_pairs.add((A, B))
            if not all(inside): degen["平行な辺の重なり・端どうし"] += 1; continue
            sig = sum((dot(m, sub(y, q0)) for y in X[A]), Q2(0)).sign()
            if sig == 0: continue
            nondeg_pairs.add((A, B)); a = row()
            for k in range(3): a[1+6*A+k] = m[k]*sig; a[1+6*B+k] = -m[k]*sig
            oa = [u + v for u, v in zip(cross(sub(p0, cs[A]), m), cross(dd, cross(ee, w)))]
            ob = [-u + v for u, v in zip(cross(sub(q0, cs[B]), m), cross(ee, cross(w, dd)))]
            for k in range(3): a[4+6*A+k] = oa[k]*sig; a[4+6*B+k] = ob[k]*sig
            rows.append(("辺と辺", (A, B), a, (p0, q0, dd, ee, sig)))
        # 平行な辺の重なり（外積 0 で同じ直線上）を数える
        for (i0, i1), (j0, j1) in itertools.product(PE, PE):
            p0, q0 = X[A][i0], X[B][j0]; dd, ee = sub(X[A][i1], p0), sub(X[B][j1], q0)
            if T.iszero(cross(dd, ee)) and T.iszero(cross(dd, sub(q0, p0))):
                t0 = dot(sub(q0, p0), dd)/dot(dd, dd); t1 = dot(sub(X[B][j1], p0), dd)/dot(dd, dd)
                lo, hi = (t0, t1) if (t1 - t0).sign() > 0 else (t1, t0)
                if (hi).sign() > 0 and (1 - lo).sign() > 0:
                    touching_pairs.add((A, B)); degen["平行な辺の重なり・端どうし"] += 1
    return n, rows, degen, touching_pairs, nondeg_pairs, cs

def simplex_phase1(A, b, tol=1e-10):
    """A λ = b、λ ≥ 0 の解を単体法（第 1 段、ブランドの規則）で探す。(解 or None, 基底の列)"""
    m, k = A.shape; A = A.copy(); b = b.copy()
    neg = b < 0; A[neg] *= -1; b[neg] *= -1
    Tb = np.hstack([A, np.eye(m), b[:, None]]); basis = list(range(k, k + m))
    cost = np.concatenate([np.zeros(k), np.ones(m), [0.0]]); z = cost - cost[basis] @ Tb
    for it in range(20000):
        enter = next((j for j in range(k + m) if z[j] < -tol), None)
        if enter is None: break
        col = Tb[:, enter]; ratios = [(Tb[r, -1]/col[r], basis[r], r) for r in range(m) if col[r] > tol]
        if not ratios: return None, basis
        _, _, r = min(ratios)
        Tb[r] /= Tb[r, enter]
        for q in range(m):
            if q != r and Tb[q, enter] != 0: Tb[q] -= Tb[q, enter]*Tb[r]
        z -= z[enter]*Tb[r]; basis[r] = enter
    if -z[-1] > 1e-8: return None, basis
    lam = np.zeros(k)
    for r, j in enumerate(basis):
        if j < k: lam[j] = Tb[r, -1]
    return lam, [j for j in basis if j < k]

def lp_max(c, A, b, tol=1e-10):
    """max c·x、A x = b、x ≥ 0 を二段階の単体法（密な表、最初の段はブランドの規則、次の段は最大の改善）で。x or None"""
    m, k = A.shape; A = A.copy(); b = b.copy(); neg = b < 0; A[neg] *= -1; b[neg] *= -1
    Tb = np.hstack([A, np.eye(m), b[:, None]]); basis = list(range(k, k + m))
    def run(cost, allowed, bland):
        z = cost - cost[basis] @ Tb
        for it in range(50000):
            cand = [j for j in allowed if z[j] < -tol]
            if not cand: return True
            enter = cand[0] if bland else min(cand, key=lambda j: z[j])
            col = Tb[:, enter]; rs = [(Tb[r, -1]/col[r], basis[r], r) for r in range(m) if col[r] > tol]
            if not rs: return False                       # 有界でない
            _, _, r = min(rs); Tb[r] /= Tb[r, enter]
            for q in range(m):
                if q != r and Tb[q, enter] != 0: Tb[q] -= Tb[q, enter]*Tb[r]
            z -= z[enter]*Tb[r]; basis[r] = enter
        return True
    c1 = np.concatenate([np.zeros(k), np.ones(m), [0.0]]); run(c1, range(k + m), True)
    if Tb[[r for r, j in enumerate(basis) if j >= k], -1].sum() > 1e-8: return None
    c2 = np.concatenate([-np.asarray(c, float), np.zeros(m), [0.0]]); run(c2, range(k), True)
    x = np.zeros(k)
    for r, j in enumerate(basis):
        if j < k: x[j] = Tb[r, -1]
    return x

def max_support(A, b, kmax=1500):
    """力のかかる触れ合いができるだけ多い λ：max Σ s_j、A λ = b、s_j ≤ λ_j、s_j ≤ 1。その λ の正の列（基底）。大きすぎれば None"""
    m, k = A.shape
    if k > kmax: return None
    I, Z = np.eye(k), np.zeros((k, k))
    # 変数 [λ, s, w, v]：A λ = b；λ − s − w = 0；s + v = 1
    big = np.block([[A, np.zeros((m, 3*k))], [I, -I, -I, Z], [Z, I, Z, I]])
    rhs = np.concatenate([b, np.zeros(k), np.ones(k)])
    x = lp_max(np.concatenate([np.zeros(k), np.ones(k), np.zeros(2*k)]), big, rhs)
    return None if x is None else x[:k]

def gauss_jordan(eqs, ncol):
    """疎な行（{列: Q2}、右辺は列 ncol）のガウス・ジョルダン消去。{ピボットの列: 簡約した行} か、矛盾すれば None"""
    piv = {}
    for r in eqs:
        r = {c: v for c, v in r.items() if v != 0}
        while True:
            hit = next((c for c in r if c in piv), None)
            if hit is None: break
            f = r[hit]
            for c, v in piv[hit].items():
                t = r.get(c, Q2(0)) - f*v
                if t == 0: r.pop(c, None)
                else: r[c] = t
        cols = [c for c in r if c != ncol]
        if not cols:
            if ncol in r: return None
            continue
        pc = min(cols, key=lambda c: (len(r), c)); inv = r[pc].inv(); r = {c: v*inv for c, v in r.items()}
        for q, row in piv.items():
            if pc in row:
                f = row[pc]
                for c, v in r.items():
                    t = row.get(c, Q2(0)) - f*v
                    if t == 0: row.pop(c, None)
                    else: row[c] = t
        piv[pc] = r
    return piv

def exact_solve(cols, rows, nvar):
    """選んだ列（触れ合い）だけで Σ λ_j a_j = e_0 を Q(√2) で正確に解く。解 or None"""
    eqs = []
    for v in range(nvar):
        r = {k: rows[j][2][v] for k, j in enumerate(cols) if rows[j][2][v] != 0}
        if v == 0: r[len(cols)] = Q2(1)
        if r: eqs.append(r)
    piv = gauss_jordan(eqs, len(cols))
    if piv is None: return None
    lam = [Q2(0)]*len(cols)
    for c, r in piv.items(): lam[c] = r.get(len(cols), Q2(0))        # 自由な列は 0
    return lam

# ---------- 二次の検査 ----------
# 回転を exp([θ]×) R₀ で表すと、向き v は v + εθ×v + ε²/2·θ×(θ×v)、中心は c + εt。隙間 g の ε² の係数 q_j(d) を正確に出し、
# 怪しい方向の空間 S（ds = 0、力のかかった触れ合いを一次で保つ。力のかからない駒の変数は外す）の上で
# −Σ λ_j q_j が正定値なら、二次の十分条件（同じ λ で）により、s はこの点で厳密な局所最小（力のかからない駒の動きを除く）
def ser_add(a, b): return [[x + y for x, y in zip(u, v)] for u, v in zip(a, b)]
def ser_sub(a, b): return [[x - y for x, y in zip(u, v)] for u, v in zip(a, b)]
def ser_dot(a, b): return [dot(a[0], b[0]), dot(a[0], b[1]) + dot(a[1], b[0]), dot(a[0], b[2]) + dot(a[1], b[1]) + dot(a[2], b[0])]
def ser_cross(a, b):
    z = lambda *vs: [sum((v[k] for v in vs), Q2(0)) for k in range(3)]
    return [cross(a[0], b[0]), z(cross(a[0], b[1]), cross(a[1], b[0])), z(cross(a[0], b[2]), cross(a[1], b[1]), cross(a[2], b[0]))]
def ser_rot(v, th):
    w = cross(th, v); return [v, w, [t*Q2(1, 0)/2 for t in cross(th, w)]]
def ser_const(v): return [v, [Q2(0)]*3, [Q2(0)]*3]
def ser_lin(v, t): return [v, t, [Q2(0)]*3]

def q2coef(row, cs, d):
    """触れ合い row の隙間の、動き d（長さ 1 + 6n）に沿った ε² の係数"""
    kind, ids, _, meta = row
    tt = lambda i: d[1+6*i:4+6*i]; th = lambda i: d[4+6*i:7+6*i]
    if kind == "壁":
        x, nn = meta; i = ids; r = sub(x, cs[i]); rr = ser_rot(r, th(i))
        return -dot(nn, rr[2])
    if kind in ("頂点と面", "面の重なりの端"):
        p, N, H = meta; a_, b_ = ids
        xa = ser_add(ser_lin(cs[a_], tt(a_)), ser_rot(sub(p, cs[a_]), th(a_))); cb = ser_lin(cs[b_], tt(b_))
        return ser_dot(ser_rot(N, th(b_)), ser_sub(xa, cb))[2]
    p0, q0, dd, ee, sig = meta; A, B = ids
    P0 = ser_add(ser_lin(cs[A], tt(A)), ser_rot(sub(p0, cs[A]), th(A))); Q0 = ser_add(ser_lin(cs[B], tt(B)), ser_rot(sub(q0, cs[B]), th(B)))
    return ser_dot(ser_sub(P0, Q0), ser_cross(ser_rot(dd, th(A)), ser_rot(ee, th(B))))[2]*sig

def nullspace(rows_, ncol):
    """Q(√2) の行列の核の基底（疎なガウス・ジョルダン消去）"""
    piv = gauss_jordan([{c: v for c, v in enumerate(r) if v != 0} for r in rows_], ncol)
    free = [c for c in range(ncol) if c not in piv]; basis = []
    for fcol in free:
        v = [Q2(0)]*ncol; v[fcol] = Q2(1)
        for c, r in piv.items():
            if fcol in r: v[c] = -r[fcol]
        basis.append(v)
    return basis

def second_order(n, rows, cs, pos, lam, loaded):
    """(怪しい方向の次元, 平行移動だけの方向の次元, 判定)。判定は
    "strict"：−Σλq が S の上で正定値 → s は厳密な局所最小（力のかからない駒を除く）
    "slides"：−Σλq が S の上で半正定値で、核がちょうど平行移動だけの方向（滑れる駒）→ 滑れる動きを除いて局所最小
             （平行移動に対して隙間は一次式なので、その方向には s を変えずに正確に動ける配置の族がある）
    "no"：上のどちらでもない（この λ では決着しない）"""
    nvar = 1 + 6*n; keep = [0] + [1 + 6*i + k for i in loaded for k in range(6)]
    R = [[rows[j][2][v] for v in keep] for j in pos] + [[Q2(1)] + [Q2(0)]*(len(keep) - 1)]
    B = nullspace(R, len(keep)); m = len(B)
    if m == 0: return 0, 0, "strict"
    rotcols = [c for c, v in enumerate(keep) if v > 0 and (v - 1) % 6 >= 3]
    BT = nullspace(R + [[Q2(1) if c == rc else Q2(0) for c in range(len(keep))] for rc in rotcols], len(keep))
    def full(v):
        d = [Q2(0)]*nvar
        for k, t in zip(keep, v): d[k] = t
        return d
    Bf = [full(v) for v in B]
    Qf = lambda d: -sum((l*q2coef(rows[j], cs, d) for j, l in zip(pos, lam)), Q2(0))
    diag = [Qf(v) for v in Bf]; M = [[None]*m for _ in range(m)]
    for i in range(m):
        M[i][i] = diag[i]
        for j in range(i+1, m): M[i][j] = M[j][i] = (Qf([x + y for x, y in zip(Bf[i], Bf[j])]) - diag[i] - diag[j])/2
    # 対称な消去（対角のピボット）。ピボットが 0 なら、その行が全部 0 でなければ不定
    A = [r[:] for r in M]; act = list(range(m)); zero = 0; pivs = []
    while act:
        k = next((i for i in act if A[i][i].sign() != 0), None)
        if k is None:
            if any(A[i][j] != 0 for i in act for j in act): return m, len(BT), "no"
            zero += len(act); break
        if A[k][k].sign() < 0: return m, len(BT), "no"
        pivs.append(A[k][k]); act.remove(k)
        for i in act:
            f = A[i][k]/A[k][k]
            for j in act: A[i][j] = A[i][j] - f*A[k][j]
    if zero == 0: return m, len(BT), "strict"
    return m, len(BT), ("slides" if zero == len(BT) else "no")

def main(fn):
    d = json.load(open(fn, encoding="utf-8")); SOL = json.load(open(os.path.join(HERE, "data", "src", "solids_exact.json"), encoding="utf-8"))
    n, rows, degen, tp, ndp, cs = contacts(d, SOL); nvar = 1 + 6*n
    kinds = {}
    for r in rows: kinds[r[0]] = kinds.get(r[0], 0) + 1
    print(f"{d.get('record_id', fn)}：{n} 個、s = {T.num(d['s'])}")
    print(f"  使う触れ合い：{kinds}。外した退化した触れ合い：{degen}")
    print(f"  触れ合う組 {len(tp)}、そのうち退化していない触れ合いを持つ組 {len(ndp)}")
    A = np.array([[float(t) for t in r[2]] for r in rows]).T; b = np.zeros(nvar); b[0] = 1.0
    lam, basis = simplex_phase1(A, b)
    if lam is None:
        print("  力が見つからない → 使った触れ合いだけでは、一次で箱を縮める動きが残る（退化した触れ合いの場合分けが要るか、本当に縮められる）")
        return 1
    # 力の組は一通りではない。列の順番を変えて単体法を何回か解き（ブランドの規則は列の順に依る）、正確な解を平均して力を散らす
    # （正しい力の平均もまた正しい力。力のかかる触れ合いが増えると怪しい方向が減り、二次の検査が通りやすくなる）
    sols = []; rng = np.random.default_rng(0); NRUN = int(os.environ.get("NRUN", 6))
    ms = max_support(A, b)
    if ms is not None:
        # 正の列だけで正確に解く。台が基底より大きいので解は一通りでない：浮動小数の解に近い頂点を、台の中で単体法の第 1 段で取り直し、
        # 列の順を変えた複数の頂点の平均にする（いずれも台の上の正しい力）
        supp = [j for j in range(A.shape[1]) if ms[j] > 1e-9]
        # 台の列すべてで連立方程式を立て、自由な列は浮動小数の解を分数に丸めた値に固定し、残りを正確に解く
        eqs = []
        for v in range(nvar):
            r = {q: rows[j][2][v] for q, j in enumerate(supp) if rows[j][2][v] != 0}
            if v == 0: r[len(supp)] = Q2(1)
            if r: eqs.append(r)
        piv = gauss_jordan(eqs, len(supp))
        if piv is not None:
            fv = {q: Q2(Fr(float(ms[j])).limit_denominator(10**6)) for q, j in enumerate(supp) if q not in piv}
            val = dict(fv)
            for q, r in piv.items(): val[q] = r.get(len(supp), Q2(0)) - sum((c*fv[t] for t, c in r.items() if t != len(supp) and t != q), Q2(0))
            if all(v.sign() > 0 for v in val.values()): sols.append({supp[q]: v for q, v in val.items()})
        print(f"  力のかかる触れ合いを最大にする線形計画：台 {len(supp)} 本" + ("" if sols else "（正確な正の解にならなかった）"))
        for run in range(NRUN if not sols else 0):
            order = supp if run == 0 else list(rng.permutation(supp))
            lr, br = simplex_phase1(A[:, order], b)
            if lr is None: continue
            bcols = [order[j] for j in br]; ex = exact_solve(bcols, rows, nvar)
            if ex is not None and all(t.sign() >= 0 for t in ex): sols.append(dict(zip(bcols, ex)))
    for run in range(NRUN if not sols else 0):
        perm = np.arange(A.shape[1]) if run == 0 else rng.permutation(A.shape[1])
        lr, br = (lam, basis) if run == 0 else simplex_phase1(A[:, perm], b)
        if lr is None: continue
        bcols = list(br) if run == 0 else [int(perm[j]) for j in br]
        ex = exact_solve(bcols, rows, nvar)
        if ex is None or any(t.sign() < 0 for t in ex): continue
        sols.append(dict(zip(bcols, ex)))
    if not sols: print("  浮動小数点では力が見つかったが、正確な数で 0 以上の解にならなかった"); return 1
    lamd = {}
    for sd in sols:
        for j, l in sd.items(): lamd[j] = lamd.get(j, Q2(0)) + l*Q2(Fr(1, len(sols)))
    full = [Q2(0)]*nvar
    for j, l in lamd.items():
        for v in range(nvar): full[v] = full[v] + l*rows[j][2][v]
    assert full[0] == 1 and all(t == 0 for t in full[1:]), "つり合いの検算に失敗"
    pos = [j for j, l in lamd.items() if l.sign() > 0]
    print(f"  力のつり合い：正の力 {len(pos)} 本（壁 {sum(1 for j in pos if rows[j][0] == '壁')}、{len(sols)} 通りの解の平均）で、すべての駒の力とねじる力が正確に 0、壁の仕事が 1")
    print("  FIRST-ORDER-CERTIFIED：一次では箱を縮められない")
    loaded = sorted({i for j in pos for i in ([rows[j][1]] if rows[j][0] == "壁" else list(rows[j][1]))})
    lam_pos = [lamd[j] for j in pos]
    m, mt, verdict = second_order(n, rows, cs, pos, lam_pos, loaded)
    print(f"  力のかかった駒 {len(loaded)} / {n}。怪しい方向（ds = 0、力のかかった触れ合いを一次で保つ）{m} 次元、そのうち平行移動だけの方向（滑れる駒）{mt} 次元")
    if verdict == "strict":
        print("  SECOND-ORDER-CERTIFIED：怪しい方向すべてで −Σλq が正（または怪しい方向がない）→ s は厳密な局所最小（力のかからない駒を除く）")
    elif verdict == "slides":
        print("  LOCAL-MIN（滑れる駒を除く）：−Σλq は平行移動だけの方向で 0、残りの方向で正 → 滑れる動きを除いて局所最小")
    else:
        print("  二次の検査で決着しない（この λ では −Σλq が半正定値にならない）")
    return 0 if verdict != "no" else 2


def run_all():
    """data/exact_touch で EXACT-VALID の配置すべてを CPU−4 並列（窓を出さない子プロセス）で調べ、data/exact_touch/force_summary.tsv に書く"""
    import csv, subprocess
    from concurrent.futures import ThreadPoolExecutor
    D = os.path.join(HERE, "data", "exact_touch")
    ids = [r["record"] for r in csv.DictReader(open(os.path.join(D, "summary.tsv"), encoding="utf-8"), delimiter="	") if r["status"] == "EXACT-VALID"]
    NOWIN = getattr(subprocess, "CREATE_NO_WINDOW", 0); print(len(ids), "件", flush=True)
    KEYS = [("SECOND-ORDER-CERTIFIED", "厳密な局所最小"), ("LOCAL-MIN", "局所最小（滑れる駒を除く）"), ("力が見つからない", "力が見つからない"),
            ("決着しない", "二次で決着しない"), ("正確な数で解けなかった", "正確に解けない"), ("負のもの", "正確な力が負")]
    def one(rid):
        try:
            o = subprocess.run([sys.executable, "-I", "-X", "utf8", os.path.abspath(__file__), os.path.join(D, rid + ".json")],
                               capture_output=True, text=True, encoding="utf-8", timeout=1800, creationflags=NOWIN).stdout
            st = next((lab for k, lab in KEYS if k in o), "異常終了")
            info = " / ".join(l.strip() for l in o.splitlines() if "使う触れ合い" in l or "怪しい方向" in l)
        except subprocess.TimeoutExpired: st, info = "時間切れ", ""
        line = f"{rid}	{st}	{info}"; print(line, flush=True); return line
    with ThreadPoolExecutor(max(1, os.cpu_count() - 4)) as ex: res = list(ex.map(one, ids))
    with open(os.path.join(D, "force_summary.tsv"), "w", encoding="utf-8") as w:
        w.write("record\tstatus\tinfo\n"); w.writelines(l + "\n" for l in res)

if __name__ == "__main__":
    if sys.argv[1] == "--all": run_all(); sys.exit(0)
    sys.exit(main(sys.argv[1]))
