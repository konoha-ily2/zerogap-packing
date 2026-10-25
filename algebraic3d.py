# 凸多面体の詰め込みの S を、接触の多項式の連立方程式として高精度に解き、S を根にもつ整数係数の多項式と閉じた式を探す
#   python algebraic3d.py <記録> [--dps 120] [--tol 自動] [--maxdeg 24] [--quick]
#   --quick：ふるい分け用（60 桁、出発点をずらした解き直しなし）。結果に quick: true と書く。式が出たものは通常の設定で確かめ直す
#   <記録> は次のどれか
#     ・platonic-packing の記録の ID（例 cubincub_n37）：records/<問題>/<ID>.json を data/records/ に取ってきて使う
#     ・platonic-packing 形式のファイル（piece, container, s_full, pieces = [x, y, z, qw, qx, qy, qz] の並び）
#     ・単純な立方体の形式のファイル（n, S, X, Y, Z, R。立方体を立方体に入れる配置）
#   立体の厳密な座標は platonic-packing の src/solids_exact.json（data/src/ に取ってくる）。座標は最小多項式の根 θ の多項式
#
# 1. 接触を拾う（倍精度）：頂点が容器の面に乗る・頂点が相手の面に乗る・辺と辺が交わる。
#    向こうの配置は隙間 1e-7 ほどの余裕つきなので、接触とみなす幅は隙間の分布の切れ目から決める（--tol で指定も可）
# 2. 変数：S と、各立体の中心 c と四元数 q（スカラーが先）。回転は M(q)（q の 2 次式）で、|q|² = 1 を別の式として課す
#      頂点 v が容器の面（外向き法線 n、標準の容器での距離 h）に乗る：n·v - S h = 0
#      頂点 v が B の面（標準の B での法線 n、距離 h）に乗る：(R_B n)·(v - c_B) - h = 0
#      辺 (p, d) と辺 (p', d') が交わる：(p - p')·(d × d') = 0
# 3. 剛性：倍精度のヤコビ行列の核（接触を保ったままの動き）に S の成分がなければ、S は接触だけで決まる
# 4. 混合精度の反復改良：倍精度のヤコビ行列の擬似逆行列で修正量を求め、残差だけを高精度で計算する（複素数のステップで微分）。
#    核の向き（遊んでいる立体など）は擬似逆行列の最小ノルムの解で動かさない。出発点をずらした解き直しで S の一意性も確かめる。
#    接触を保つ動きの族があって S が族の上で 2 次で増える配置（最小点が特異点）は、止まった残差から S の信頼できる桁数を見積もる
# 5. mpmath.findpoly で整数係数の多項式を探し、4 次以下なら sympy で根号の式にする。結果は data/closed_forms/<ID>.json
import json, sys, os, re, math, random, argparse, itertools, urllib.request
from fractions import Fraction
import numpy as np
import mpmath as mp

HERE = os.path.dirname(os.path.abspath(__file__)); PP = os.path.join(HERE, "data")
RAW = "https://raw.githubusercontent.com/alejandrozu/platonic-packing/main/"
ap = argparse.ArgumentParser()
ap.add_argument("src"); ap.add_argument("--dps", type=int, default=120)
ap.add_argument("--tol", type=float, default=0.0); ap.add_argument("--maxdeg", type=int, default=24)
ap.add_argument("--quick", action="store_true")
args = ap.parse_args(); DPS = 60 if args.quick else args.dps
mp.mp.dps = DPS + 20

def fetch(rel):
    p = os.path.join(PP, *rel.split("/"))
    if not os.path.exists(p):
        os.makedirs(os.path.dirname(p), exist_ok=True); urllib.request.urlretrieve(RAW + rel, p)
    return p

# ---------- 記録を読む ----------
src = args.src
if not os.path.exists(src):
    src = fetch(f"records/{src.rsplit('_n', 1)[0]}/{src}.json")
rec = json.load(open(src, encoding="utf-8"))
if "pieces" in rec:
    rid = rec.get("record_id") or os.path.splitext(os.path.basename(src))[0]
    PIECE, CONT, S0 = rec["piece"], rec["container"], float(rec["s_full"])
    P0 = [list(map(float, p)) for p in rec["pieces"]]
else:   # 単純な立方体の形式：箱 [0, S]³ の立方体。軸 a_k = R[9i+3k : 9i+3k+3]、頂点 = c + Σ ±a_k/2
    rid = "ours_cubincub_n%d" % rec["n"]; PIECE = CONT = "cube"; S0 = float(rec["S"]); P0 = []
    for i in range(rec["n"]):
        a = [rec["R"][9*i+3*k: 9*i+3*k+3] for k in range(3)]
        M = np.array(a).T
        if np.linalg.det(M) < 0: M[:, 2] = -M[:, 2]          # 立方体としては同じ
        w = math.sqrt(max(0.0, 1 + M[0, 0] + M[1, 1] + M[2, 2]))/2
        if w > 0.1: q = [w, (M[2, 1]-M[1, 2])/(4*w), (M[0, 2]-M[2, 0])/(4*w), (M[1, 0]-M[0, 1])/(4*w)]
        else:
            k = int(np.argmax([M[0, 0], M[1, 1], M[2, 2]])); i1, i2 = (k+1) % 3, (k+2) % 3
            t = math.sqrt(max(0.0, 1 + M[k, k] - M[i1, i1] - M[i2, i2]))/2; q = [0.0]*4
            q[0] = (M[i2, i1]-M[i1, i2])/(4*t); q[1+k] = t; q[1+i1] = (M[i1, k]+M[k, i1])/(4*t); q[1+i2] = (M[i2, k]+M[k, i2])/(4*t)
        c = [rec["X"][i]-S0/2, rec["Y"][i]-S0/2, rec["Z"][i]-S0/2]; P0.append(c + q)
n = len(P0)

# ---------- 立体（厳密な座標を高精度で） ----------
SOL = json.load(open(fetch("src/solids_exact.json"), encoding="utf-8"))
def fr(s): f = Fraction(s); return mp.mpf(f.numerator)/f.denominator
def solid(name):
    e = SOL[name]
    if e.get("D") is not None: sys.exit(f"{name} は 2 段の拡大体（D あり）で、まだ扱えない")
    co = [fr(c) for c in e["minpoly"]]; lo, hi = (fr(t) for t in e["interval"])
    f = lambda t: sum(c*t**k for k, c in enumerate(co))
    th = -co[0]/co[1] if len(co) == 2 else mp.findroot(f, (lo+hi)/2)
    assert lo - mp.mpf(10)**-25 <= th <= hi + mp.mpf(10)**-25
    V = [[sum(fr(c)*th**k for k, c in enumerate(comp)) for comp in v] for v in e["V"]]
    F, Nn, H = e["faces"], [], []
    for f_ in F:
        a, b, c = (V[j] for j in f_[:3])
        u = [b[k]-a[k] for k in range(3)]; w = [c[k]-a[k] for k in range(3)]
        nn = [u[1]*w[2]-u[2]*w[1], u[2]*w[0]-u[0]*w[2], u[0]*w[1]-u[1]*w[0]]
        L = mp.sqrt(sum(t*t for t in nn)); nn = [t/L for t in nn]; h = sum(nn[k]*a[k] for k in range(3))
        if h < 0: nn = [-t for t in nn]; h = -h
        Nn.append(nn); H.append(h)
    E = sorted({tuple(sorted((f_[k], f_[(k+1) % len(f_)]))) for f_ in F for k in range(len(f_))})
    flt = lambda m: [[float(t) for t in r] for r in m]
    return dict(V=V, N=Nn, H=H, E=E, Vf=flt(V), Nf=flt(Nn), Hf=[float(h) for h in H],
                rad=max(math.sqrt(sum(float(t)**2 for t in v)) for v in V))
PC, CC = solid(PIECE), solid(CONT)

# ---------- 幾何（float と mpf のどちらでも動く） ----------
NV = 1 + 7*n
def geo(x, i):
    o = 1 + 7*i; c = x[o:o+3]; w, a, b, d = x[o+3:o+7]
    R = [[w*w+a*a-b*b-d*d, 2*(a*b-w*d), 2*(a*d+w*b)],
         [2*(a*b+w*d), w*w-a*a+b*b-d*d, 2*(b*d-w*a)],
         [2*(a*d-w*b), 2*(b*d+w*a), w*w-a*a-b*b+d*d]]
    return c, R
def rot(R, v): return [R[k][0]*v[0]+R[k][1]*v[1]+R[k][2]*v[2] for k in range(3)]
def wv(g, v): c, R = g; r = rot(R, v); return [c[k]+r[k] for k in range(3)]
def dot(u, v): return u[0]*v[0]+u[1]*v[1]+u[2]*v[2]
def cross(u, v): return [u[1]*v[2]-u[2]*v[1], u[2]*v[0]-u[0]*v[2], u[0]*v[1]-u[1]*v[0]]

def eqval(e, x, G, hp):
    V, N, H = (PC["V"], PC["N"], PC["H"]) if hp else (PC["Vf"], PC["Nf"], PC["Hf"])
    t = e[0]
    if t == "q":
        o = 1 + 7*e[1]; return sum(z*z for z in x[o+3:o+7]) - 1
    if t == "w":
        _, i, v, f = e; CN, CH = (CC["N"], CC["H"]) if hp else (CC["Nf"], CC["Hf"])
        return dot(CN[f], wv(G[i], V[v])) - x[0]*CH[f]
    if t == "vf":
        _, a, v, b, g = e; p = wv(G[a], V[v]); cb = G[b][0]
        return dot(rot(G[b][1], N[g]), [p[k]-cb[k] for k in range(3)]) - H[g]
    _, a, ea, b, eb = e
    p0, p1 = wv(G[a], V[PC["E"][ea][0]]), wv(G[a], V[PC["E"][ea][1]])
    q0, q1 = wv(G[b], V[PC["E"][eb][0]]), wv(G[b], V[PC["E"][eb][1]])
    d = [p1[k]-p0[k] for k in range(3)]; d2 = [q1[k]-q0[k] for k in range(3)]
    return dot([p0[k]-q0[k] for k in range(3)], cross(d, d2))

# ---------- 1. 接触を拾う ----------
x0 = [S0] + [v for p in P0 for v in p]
for i in range(n):                                    # 四元数を正規化しておく
    o = 1 + 7*i; L = math.sqrt(sum(t*t for t in x0[o+3:o+7])); x0[o+3:o+7] = [t/L for t in x0[o+3:o+7]]
Gf = [geo(x0, i) for i in range(n)]
WV = [[wv(Gf[i], v) for v in PC["Vf"]] for i in range(n)]
cand = []                                             # (隙間, 式)
for i in range(n):
    for v, p in enumerate(WV[i]):
        for f in range(len(CC["Nf"])):
            gap = S0*CC["Hf"][f] - dot(CC["Nf"][f], p)
            if gap < 1e-2: cand.append((gap, ("w", i, v, f)))
for A, B in itertools.combinations(range(n), 2):
    if math.dist(Gf[A][0], Gf[B][0]) > 2*PC["rad"] + 1e-2: continue
    for a, b in ((A, B), (B, A)):                     # a の頂点が b の面に乗るか
        NB = [rot(Gf[b][1], nn) for nn in PC["Nf"]]
        for v, p in enumerate(WV[a]):
            vals = [dot(NB[g], [p[k]-Gf[b][0][k] for k in range(3)]) - PC["Hf"][g] for g in range(len(NB))]
            d = max(vals)
            if d < 1e-2:
                for g, t in enumerate(vals):
                    if t > -1e-2: cand.append((max(d, -t), ("vf", a, v, b, g)))
    for ea, (a0, a1) in enumerate(PC["E"]):           # 辺と辺（両方の辺の内部で、平行でないもの）
        p, u = WV[A][a0], [WV[A][a1][k]-WV[A][a0][k] for k in range(3)]
        for eb, (b0, b1) in enumerate(PC["E"]):
            q, w = WV[B][b0], [WV[B][b1][k]-WV[B][b0][k] for k in range(3)]
            uu, ww, uw = dot(u, u), dot(w, w), dot(u, w); r = [p[k]-q[k] for k in range(3)]
            den = uu*ww - uw*uw
            if den < 1e-8*uu*ww: continue
            s = (uw*dot(w, r) - ww*dot(u, r))/den; t = (uu*dot(w, r) - uw*dot(u, r))/den
            if not (1e-6 < s < 1-1e-6 and 1e-6 < t < 1-1e-6): continue
            dist = math.dist([p[k]+s*u[k] for k in range(3)], [q[k]+t*w[k] for k in range(3)])
            if dist < 1e-2: cand.append((dist, ("ee", A, ea, B, eb)))
gaps = sorted(max(abs(g), 1e-16) for g, _ in cand) + [1e-2]   # 1e-2 は番兵（それより大きい隙間は拾っていない）
if args.tol: TOL = args.tol
else:   # 1e-4 より下で、隣どうしの比がいちばん大きい切れ目
    best = (0, 1e-6)
    for g1, g2 in zip(gaps, gaps[1:]):
        if g1 < 1e-4 and g2/g1 > best[0]: best = (g2/g1, math.sqrt(g1*g2))
    TOL = best[1]
    lo = max([g for g in gaps if g < TOL], default=0); hi = min([g for g in gaps if g > TOL], default=float("inf"))
    print(f"接触の幅 {TOL:.1e}（接触の最大の隙間 {lo:.1e}、接触でない最小の隙間 {hi:.1e}、比 {best[0]:.1e}）")
eqs = [("q", i) for i in range(n)] + list(dict.fromkeys(e for g, e in cand if abs(g) < TOL))
kinds = {k: sum(1 for e in eqs if e[0] == k) for k in ("w", "vf", "ee")}
print(f"{rid}：{PIECE} {n} 個 in {CONT}、S0 = {S0}。接触：容器の面 {kinds['w']}、頂点と面 {kinds['vf']}、辺と辺 {kinds['ee']}")
of = [[] for _ in range(n)]                            # 立体ごとに、それを含む式
for k, e in enumerate(eqs):
    for i in ({e[1]} if e[0] in ("q", "w") else {e[1], e[3]}): of[i].append(k)
wall = [k for k, e in enumerate(eqs) if e[0] == "w"]

# ---------- 2〜3. ヤコビ行列（倍精度、複素数のステップ）と剛性 ----------
# 式はすべて多項式なので、f(x + ih) の虚部 / h が引き算の桁落ちなしに微分を与える（差分だと誤差 1e-9 で、反復改良が 1e-20 で止まった）
def jac(x):
    x = [float(t) for t in x]; G = [geo(x, i) for i in range(n)]; J = np.zeros((len(eqs), NV)); h = 1e-30
    for k in wall: J[k, 0] = -CC["Hf"][eqs[k][3]]
    for i in range(n):
        for j in range(7):
            col = 1 + 7*i + j; xp = list(x); xp[col] += 1j*h
            Gp = list(G); Gp[i] = geo(xp, i)
            for k in of[i]: J[k, col] = eqval(eqs[k], xp, Gp, False).imag/h
    return J
def pinv_info(J, rel=1e-8, krel=None):
    # 式が変数より多いときは薄い特異値分解（n = 81 の 4395 × 568 で、左特異ベクトルを全部作ると 1 回 42 秒かかっていた）
    U, sv, Vt = np.linalg.svd(J, full_matrices=J.shape[0] < J.shape[1])
    rank = int((sv > rel*sv[0]).sum()); krank = rank if krel is None else int((sv > krel*sv[0]).sum())
    P = (Vt[:rank].T / sv[:rank]) @ U[:, :rank].T
    return P, krank, sv, Vt[krank:]
def indep_rows(J, tol=1e-10):
    """独立な式（行）を前から貪欲に選ぶ（グラム・シュミット）"""
    B = np.zeros((J.shape[1], J.shape[1])); r = 0; keep = []
    for k, row in enumerate(J):
        nr = np.linalg.norm(row)
        if nr == 0: continue
        v = row - B[:r].T @ (B[:r] @ row); v = v - B[:r].T @ (B[:r] @ v)
        nv = np.linalg.norm(v)
        if nv > tol*nr and r < J.shape[1]: B[r] = v/nv; r += 1; keep.append(k)
    return keep
def use_rows(idx):
    """解くのに使う式を ALL[idx] にする（重複した接触の式を落とすと、高精度の残差の計算が式の数に比例して速くなる）"""
    global eqs, of, wall
    eqs = [ALL[k] for k in idx]; of = [[] for _ in range(n)]
    for k, e in enumerate(eqs):
        for i in ({e[1]} if e[0] in ("q", "w") else {e[1], e[3]}): of[i].append(k)
    wall = [k for k, e in enumerate(eqs) if e[0] == "w"]
def resid(x):
    G = [geo(x, i) for i in range(n)]; return [eqval(e, x, G, True) for e in eqs]

# ---------- 4. 混合精度の反復改良 ----------
# 接触を保つ動きの族があり、族に沿うと S が 2 次で増える配置では、最小点が族の枝の交わる特異点になる（n = 29 の 45° 回った立方体）。
# そこではヤコビ行列の特異値がほぼ 0（2e-15）になり、倍精度のヤコビ行列ではその向きのずれ t を 1e-16 より細かく扱えず、残差が t² ≈ 1e-32 で止まる。
# S の誤差も t² 程度なので、止まった残差から S の信頼できる桁数を見積もる。擬似逆行列の特異値の切り捨ては相対 1e-12
# （1e-8 では t ≈ 1e-9 の折り目の向きまで捨ててしまい、出発点をずらした解き直しが残差 1e-18 で止まった）
def solve(xs, label, quiet=False, P0=None):
    # P0：使い回す擬似逆行列（族の上の最小化での射影。止まったら一度だけ作り直す）
    x = [mp.mpf(t) for t in xs]; P = P0; fixed = P0 is not None; hist = []
    for it in range(300):
        F = resid(x); res = max(abs(t) for t in F); hist.append(res)
        if not quiet and (it % 10 == 0 or res < mp.mpf(10)**(-DPS)): print(f"  {label} {it}: 残差 {mp.nstr(res, 3)}  S = {mp.nstr(x[0], 25)}", flush=True)
        if res < mp.mpf(10)**(-DPS): return x, res
        if fixed and len(hist) > 4 and hist[-1] > hist[-3]*mp.mpf("0.1"):
            P = None; fixed = False; hist = hist[-1:]
        if len(hist) > 8 and hist[-1] > hist[-6]*mp.mpf("0.1"):
            if not quiet:
                bad = sorted(range(len(F)), key=lambda k: -abs(F[k]))[:3]
                print(f"  {label}：残差が {mp.nstr(res, 3)} で止まった。残差の大きい式：" + "、".join(f"{eqs[k]} {mp.nstr(F[k], 2)}" for k in bad))
            return x, res
        if P is None or (not fixed and (res > 1e-12 or it < 4)): P, *_ = pinv_info(jac(x), 1e-12)
        dx = P @ np.array([float(t) for t in F])
        x = [xi - mp.mpf(float(d)) for xi, d in zip(x, dx)]
    return x, res

# 接触を保つ動きの族があると、厳密に収束した解でも族の最小点とは限らない（立方体 n = 9 は最小点から t ≈ 1e-15 ずれた点に収束し、
# S が 2e-30 大きかった）。核の各向きに ±h 動かして接触の式に戻し（射影）、S の差分から高精度の勾配を出し、
# 勾配の向きに 1 次元のニュートン法で動かす（厳密な直線探索つきの最急降下。n = 9 で勾配 1.6e-15 → 2e-32 → 4e-62）。
# 完全なヘッセ行列は、核の基底で遊んでいる立体の向きと族の向きが混ざって 25 本 × 25 本になり、n = 9 で 5 分かかったのでやめた。
# 射影が残差の下限で止まる特異点（n = 29）では使えないので、そのときは止まった残差から桁数を決める
def minimize(x, label):
    exact = mp.mpf(10)**(-DPS); h = mp.mpf("1e-30")
    def proj(y):
        z, r = solve(y, "", quiet=True, P0=Pb)
        if r > exact: raise ArithmeticError
        return z
    def along(x, d, t): return proj([xi + t*di for xi, di in zip(x, d)])
    try:
        for rnd in range(12):
            Pb, _, _, V = pinv_info(jac(x), 1e-12, 1e-11)
            if not len(V): return x, "なし"
            Vm = [[mp.mpf(float(t)) for t in v] for v in V]
            g = [(along(x, v, h)[0] - along(x, v, -h)[0])/(2*h) for v in Vm]
            gn_ = max(abs(t) for t in g)
            # S の誤差は 勾配² / 曲率 程度。勾配は差分の雑音で 1e-62 ほどで止まる（h = 1e-30）ので、10^-(桁数/2 - 5) で打ち切る
            if gn_ < mp.mpf(10)**(-(DPS//2-5)): return x, "済" if rnd else "なし"
            d = [sum(g[j]*Vm[j][i] for j in range(len(Vm)))/gn_ for i in range(NV)]   # 勾配の向き（高精度）
            Sp, Sm, S0_ = along(x, d, h)[0], along(x, d, -h)[0], x[0]
            sl, cv = (Sp - Sm)/(2*h), (Sp + Sm - 2*S0_)/h**2
            if cv <= 0:
                print(f"  {label}：族の上で S の曲率が正でない（{mp.nstr(cv, 3)}）→ 鞍点"); return x, "鞍点"
            t = -sl/cv; x = along(x, d, t)
            print(f"  {label}：族の上で最小化 {rnd+1}（勾配 {mp.nstr(gn_, 2)}、曲率 {mp.nstr(cv, 3)}、動かした量 {mp.nstr(abs(t), 2)}）S = {mp.nstr(x[0], 40)}", flush=True)
    except ArithmeticError:
        print(f"  {label}：射影が収束しない（特異点）→ 族の上の最小化はしない"); return x, "特異点"
    return x, "済"

J0 = jac(x0); _, rank, sv, ker = pinv_info(J0)
sdep = float(np.abs(ker[:, 0]).max()) if len(ker) else 0.0
print(f"式 {len(eqs)} 本・変数 {NV} 個：階数 {rank}、核の次元 {len(ker)}、核の S 成分の最大 {sdep:.1e}"
      + (f"（特異値の切れ目 {sv[rank-1]:.1e} → {sv[rank]:.1e}）" if rank < len(sv) else ""))
out = dict(record=rid, piece=PIECE, container=CONT, n=n, s0=S0, tol=TOL, contacts=kinds, equations=len(eqs),
           variables=NV, rank=rank, kernel=len(ker), kernel_s_component=sdep)
ALL = list(eqs); ROWS = list(range(len(ALL))); EXACT = mp.mpf(10)**(-DPS)

def full_resid(x):
    use_rows(range(len(ALL))); r = max(abs(t) for t in resid(x)); use_rows(ROWS); return r
def main_pass(xs, label):
    """全部の式で解く → 核と S 成分 → 族の上で最小化。(x, 残差, 最小化, S 成分, 階数, 核の次元)
    最小化（射影の繰り返しで時間の大半を使う）だけは、解いた点のヤコビ行列で選んだ独立な式で行い、最後に全部の式で確かめる。
    出発点（余裕つきの配置）で選ぶと、勾配がたまたまそろう式を落として別の点に収束した（立方体→正十二面体 n = 44 で 1.2e-7、n = 29 で 3.7e-29）"""
    global ROWS
    ROWS = list(range(len(ALL))); use_rows(ROWS)
    x, res = solve(xs, label); mn = "なし"; sdep = 1.0; rank1 = nker = None
    if res < mp.mpf("1e-18"):
        J1 = jac(x); _, rank1, sv1, ker1 = pinv_info(J1, 1e-11); nker = len(ker1)
        sdep = float(np.abs(ker1[:, 0]).max()) if nker else 0.0
        print(f"{label}：解いた点で階数 {rank1}、核の次元 {nker}（特異値 {sv1[rank1-1]:.1e} 以下を 0 とみなす）、核の S 成分の最大 {sdep:.1e}")
        if sdep < 1e-6 and nker and res <= EXACT:
            rows = indep_rows(J1)
            if len(rows) < 0.8*len(ALL):
                ROWS = rows; use_rows(ROWS); print(f"  {label}：族の上の最小化は独立な式 {len(ROWS)} 本（全 {len(ALL)} 本）で")
            y, mn = minimize(x, label)
            if len(ROWS) < len(ALL):
                rf = full_resid(y)
                if rf > EXACT*mp.mpf(10)**10:
                    print(f"  {label}：落とした式の残差 {mp.nstr(rf, 3)} → 最小化を全部の式でやり直す")
                    ROWS = list(range(len(ALL))); use_rows(ROWS); y, mn = minimize(x, label)
            x = y
            if mn == "済": sdep = 0.0                 # 最小化した後は勾配が 0（最小化の前の値で判定して、正四面体 n = 6 を「決まらない」としていた）
    ROWS = list(range(len(ALL))); use_rows(ROWS)
    return x, res, mn, sdep, rank1, nker, True
x, res, mn, sdep, rank1, nker, _ = main_pass(x0, "解く")
ok = res < mp.mpf("1e-18")                            # これより大きく止まったら、接触の組が矛盾している（n = 14 は 5.7e-23 で止まる）
if ok: out.update(rank=rank1, kernel=nker, kernel_s_component=sdep)
determined = ok and sdep < 1e-8 and mn != "鞍点"
out.update(converged=bool(ok), residual=mp.nstr(res, 3), determined=determined, minimize=mn, quick=args.quick)
if ok and not determined:
    print("→ S は拾った接触だけでは決まらない（接触を保ったまま S を変えられる）")
DIG = 0
if determined and args.quick:                          # ふるい分け：解き直しをせず、残差から桁数を決める
    DIG = int(-mp.log10(max(res, EXACT))) - 3; determined = DIG >= 15
    out.update(digits=DIG, s=mp.nstr(x[0], DIG), determined=determined)
    print(f"S（ふるい分け、{DIG} 桁）= {mp.nstr(x[0], min(DIG, 60))}")
elif determined:
    # 出発点を 1e-9 ずらして解き直す（族があれば同じく最小化）。同じ S に戻れば、その桁まで信頼できる
    random.seed(1)
    x2, res2, *_ , good2 = main_pass([float(t) + random.gauss(0, 1e-9) for t in x0], "ずらして解く")
    diff = x2[0] - x[0]; floor = max(res, EXACT)
    print(f"ずらした出発点からの S との差: {mp.nstr(diff, 3)}（残差 {mp.nstr(res2, 3)}）")
    if res2 < mp.mpf("1e-18") and diff < -10*max(floor, res2, mp.mpf(10)**(-DPS+20)):
        print("→ ずらした方が S が小さい：最初の解は最小点ではない"); determined = False
    else:
        # 最初の解が特異点で止まり、ずらした解が族の上の別の点（S が大きい）に収束したときは、止まった残差から桁数を決める
        fam = mn == "特異点" or (res > EXACT and res2 < mp.mpf("1e-18") and diff > 10*max(floor, res2))
        if fam: print("→ 族の最小点が特異点：止まった残差から桁数を決める")
        err = floor if fam else max(floor, res2, abs(diff))
        DIG = int(-mp.log10(err)) - 3; determined = DIG >= 15
        out.update(recheck_diff=mp.nstr(diff, 3), family=bool(fam), digits=DIG, s=mp.nstr(x[0], DIG))
        print(f"S の信頼できる桁数 {DIG}：S = {mp.nstr(x[0], min(DIG, 60))}" + ("…" if DIG > 60 else ""))
    out.update(determined=determined)

# ---------- 5. 多項式と閉じた式 ----------
if determined:
    mp.mp.dps = DIG; S = +x[0]; poly = None; tried = 0
    for deg in range(1, args.maxdeg+1):
        maxc = int(mp.mpf(10)**min(30, int(0.75*DIG)//(deg+1)))   # 係数の桁 × (次数 + 1) を桁数の 3/4 までに（偽の関係を避ける）
        if maxc < 10: break
        tried = deg; p = mp.findpoly(S, deg, maxcoeff=maxc, maxsteps=20000)
        if p: poly = p; break
    if poly:
        if poly[0] < 0: poly = [-c for c in poly]
        print(f"多項式（次数 {len(poly)-1}、高い次数から）: {poly}"); out["poly"] = poly
        if len(poly)-1 <= 4:
            import sympy as sp
            t = sp.Symbol("x"); rts = sp.Poly(poly, t).all_roots() if len(poly)-1 == 1 else sp.roots(sp.Poly(poly, t), multiple=True)
            rts = [r for r in rts if abs(complex(sp.N(r, 30)).imag) < 1e-20] or rts
            r = min(rts, key=lambda r: abs(complex(sp.N(r, 30)) - float(S)))
            cf = sp.nsimplify(sp.radsimp(sp.simplify(r))); cf = cf if sp.count_ops(cf) <= sp.count_ops(r) else r
            out["closed_form"] = re.sub(r"√\((\d+)\)", r"√\1", str(cf).replace("sqrt", "√")); print(f"閉じた式: {out['closed_form']}")
    else:
        print(f"次数 {tried} までの多項式は見つからなかった（{DIG} 桁で探せる範囲。係数の上限は次数とともに下がる）")
        out["poly_searched_degree"] = tried
os.makedirs(os.path.join(PP, "closed_forms"), exist_ok=True)
json.dump(out, open(os.path.join(PP, "closed_forms", rid + ".json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
