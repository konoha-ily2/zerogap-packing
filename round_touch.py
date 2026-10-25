# 閉じた式の値 s ちょうどで、記録の配置を Q(√2) の正確な数に丸め、certify_touch.py で触れ合いを許して誤差なしに確かめる（2026-10-25）
# 使い方：python round_touch.py <ID>   … data/closed_forms/<ID>.json の式と、その配置（data/improved か data/records）を使う。結果を 1 行で表示
#        python round_touch.py --all  … 正四面体・正八面体・立方体どうしで、式が Q(√2) に入る記録すべてを CPU−4 並列で流し、
#                                       data/exact_touch/summary.tsv に書く（子プロセスは窓を出さない）
# 丸め方：
#   1. 配置は余裕として一様に広げてあるとみなし、中心を s_full / s で割って触れ合いの極限に戻す
#   2. 回転行列の各成分を (p + q√2)/d（d ≤ 64）として認識する（四元数は半分の角なので、Q(√2) に入らないことがある）
#   3. 中心の各座標を、有理数・√2 × 有理数・(p + q√2)/d の順に、誤差 TOL 以内で高さ（分子・分母の大きさ）が最小のものとして認識する
#   4. 認識できない座標（接触を保ったまま滑れる駒）は、その配置でいちばん多い形の共通の格子に丸める
import json, sys, os, re, glob, math, subprocess
from fractions import Fraction as Fr
from concurrent.futures import ThreadPoolExecutor
import mpmath as mp
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); PP = os.path.join(HERE, "data"); OUT = os.path.join(PP, "exact_touch")
sys.path.insert(0, HERE)
import certify_touch as T
Q2SOL = {"cube", "tetrahedron", "octahedron"}
TOL = 1e-6; DMAX = 5000; R2 = math.sqrt(2)

def parse_s(cf):
    """閉じた式（sympy の文字列）を Q2 に。√2 以外の根号を含むなら None"""
    if re.search(r"√(?!2\b)|√\(|\*\*|I", cf): return None
    e = re.sub(r"\d+", lambda m: f"F({m.group()})", cf.replace("√2", "R"))
    return eval(e, {"F": lambda k: T.Q2(k), "R": T.Q2(0, 1)})

def height(f): return max(abs(f.numerator), f.denominator)
def recog(x, dmax=DMAX, tol=TOL):
    """x ≈ a + b√2 を (高さ, a, b) で。見つからなければ None"""
    c = []
    f = Fr(x).limit_denominator(dmax)
    if abs(float(f) - x) < tol: c.append((height(f), f, Fr(0)))
    g = Fr(x/R2).limit_denominator(dmax)
    if abs(float(g)*R2 - x) < tol: c.append((height(g), Fr(0), g))
    if not c:
        mp.mp.dps = 20
        rel = mp.pslq([mp.mpf(x), 1, mp.sqrt(2)], tol=tol*10, maxcoeff=dmax, maxsteps=2000)
        if rel and rel[0]:
            a, b = Fr(-rel[1], rel[0]), Fr(-rel[2], rel[0])
            if abs(float(a) + float(b)*R2 - x) < tol: c.append((max(height(a), height(b)), a, b))
    return min(c, key=lambda t: t[0]) if c else None

def grid(xs, Ds=1, tol=3e-7, bmax=4, Gmix=200, pick=0):
    """座標の並び xs に共通の格子を探す。(形, G, [(a, b)], 格子から外れていた数, 乗ったもののずれの最大)。座標は a + b√2
    形 rat：a ∈ Z/G、b = 0。sq2：a = 0、b ∈ Z/G。mix：a, b ∈ Z/G、|b| ≤ bmax（G ≤ Gmix）。
    G の候補は k と k·Ds（Ds は s の分母、k ≤ 48）。乗る座標の数が最大のものから 1% 以内で、G が最小（同じ G なら rat、sq2、mix の順）を選ぶ。
    格子から外れた座標（滑れる駒）は、rat・sq2 ではその格子に、mix では b = 0 として a を格子に丸める
    （x + y = c のような ±1 の係数の関係は、c が格子に乗っていれば丸めても崩れない）"""
    X = np.array(xs, float); N = len(xs); cand = []
    for G in sorted({k for k in range(1, 49)} | {k*Ds for k in range(1, 49)}):
        for kind, v in (("rat", X), ("sq2", X/R2)):
            cand.append((int((np.abs(np.round(v*G)/G - v) < tol).sum()), G, kind))
        if G <= Gmix:
            A = X[:, None] - (np.arange(-bmax*G, bmax*G + 1)/G*R2)[None, :]
            cand.append((int((np.abs(np.round(A*G)/G - A).min(1) < tol).sum()), G, "mix"))
    top = max(c[0] for c in cand); order = {"rat": 0, "sq2": 1, "mix": 2}
    # pick 番目の候補：1% 以内のものを G の小さい順に並べ、その後ろに残りを乗る数の多い順に並べる（確かめて通らなければ次へ）
    near = sorted((c for c in cand if c[0] >= top - max(1, N//100)), key=lambda c: (c[1], order[c[2]]))
    rest = sorted((c for c in cand if c[0] < top - max(1, N//100)), key=lambda c: (-c[0], c[1], order[c[2]]))
    if pick >= len(near) + len(rest): return None
    fit, G, kind = (near + rest)[pick]
    out, free, worst = [], 0, 0.0
    for x in xs:
        if kind == "rat": ab = (Fr(round(x*G), G), Fr(0))
        elif kind == "sq2": ab = (Fr(0), Fr(round(x/R2*G), G))
        else:
            bs = np.arange(-bmax*G, bmax*G + 1); A = x - bs/G*R2; e = np.abs(np.round(A*G)/G - A); k = int(e.argmin())
            ab = (Fr(round(A[k]*G), G), Fr(int(bs[k]), G)) if e[k] < tol else (Fr(round(x*G), G), Fr(0))
        e = abs(float(ab[0]) + float(ab[1])*R2 - x)
        if e < tol: worst = max(worst, e)
        else: free += 1
        out.append(ab)
    return kind, G, out, free, worst

NICE = [(Fr(0), Fr(0)), (Fr(1), Fr(0)), (Fr(-1), Fr(0)), (Fr(1, 2), Fr(0)), (Fr(-1, 2), Fr(0)), (Fr(0), Fr(1, 2)), (Fr(0), Fr(-1, 2))]
def snap_rot(Rf):
    """ほぼ揃った向きの回転（遊んでいる駒）を、成分が 0, ±1, ±1/2, ±√2/2 の回転に丸める。直交で行列式 1 のときだけ返す"""
    R = []
    for row in Rf:
        r = []
        for t in row:
            ab = min(NICE, key=lambda ab: abs(float(ab[0]) + float(ab[1])*R2 - t))
            if abs(float(ab[0]) + float(ab[1])*R2 - t) > 0.15: return None
            r.append(ab)
        R.append(r)
    Q = [[T.Q2(*ab) for ab in row] for row in R]
    if not all(T.dot(Q[i], Q[j]) == (1 if i == j else 0) for i in range(3) for j in range(3)): return None
    if T.dot(Q[0], T.cross(Q[1], Q[2])) != 1: return None
    return [[[str(a), str(b)] for a, b in row] for row in R]

def quatR(q):
    w, a, b, d = q; L = math.sqrt(w*w + a*a + b*b + d*d); w, a, b, d = w/L, a/L, b/L, d/L
    return [[w*w+a*a-b*b-d*d, 2*(a*b-w*d), 2*(a*d+w*b)], [2*(a*b+w*d), w*w-a*a+b*b-d*d, 2*(b*d-w*a)], [2*(a*d-w*b), 2*(b*d+w*a), w*w-a*a-b*b+d*d]]

def config_for(rid, s):
    prob = rid.split("_n")[0]
    for p in (os.path.join(PP, "improved", rid + ".json"), os.path.join(PP, "records", prob, rid + ".json")):
        if os.path.exists(p): return p
    base = re.sub(r"(_from_n\d+)?(_desc)*$", "", rid)          # 続きから下げた結果（<ID>_desc など）は improved/<元の ID>*.json にある
    c = [(json.load(open(p, encoding="utf-8"))["s_full"], p) for p in glob.glob(os.path.join(PP, "improved", base + "*.json"))]
    c = [t for t in c if t[0] >= s]
    return min(c)[1] if c else None

def one(rid):
    cf = json.load(open(os.path.join(PP, "closed_forms", rid + ".json"), encoding="utf-8"))
    if not cf.get("closed_form"): return "式がない", ""
    s = parse_s(cf["closed_form"])
    if s is None: return "式が Q(√2) でない", ""
    fn = config_for(rid, float(s))
    if not fn: return "配置ファイルがない", ""
    rec = json.load(open(fn, encoding="utf-8"))
    lam = float(rec["s_full"])/float(s); cc = rec.get("container_center", [0, 0, 0])
    pieces = []; snapped_rot = 0
    for i, p in enumerate(rec["pieces"]):
        R = []; Rf = quatR(p[3:7])
        for row in Rf:
            r = []
            for t in row:
                g = recog(t, 64, 1e-6)
                if g is None: break
                r.append([str(g[1]), str(g[2])])
            if len(r) < 3: break
            R.append(r)
        if len(R) < 3:
            R = snap_rot(Rf); snapped_rot += 1
            if R is None:
                t = next(t for row in Rf for t in row if recog(t, 64, 1e-6) is None)
                return "回転を認識できない", f"駒 {i} の成分 {t:.9f}（{os.path.basename(fn)}）"
        pieces.append(dict(c=[(p[k] - cc[k])/lam for k in range(3)], R=R))
    xs = [x for p in pieces for x in p["c"]]
    Ds = s.a.denominator*s.b.denominator//math.gcd(s.a.denominator, s.b.denominator)
    SOL = json.load(open(os.path.join(PP, "src", "solids_exact.json"), encoding="utf-8"))
    first = None
    # 格子の候補を上から 4 つまで試し、だめなら許容値を 2e-6 に緩めてもう一度（正八面体→正八面体 n = 35 は、
    # 3e-7 では入力の誤差の大きい座標が外れて誤った格子を選び、2e-6 の √2 × 1/10 で通った）
    for tol, pick in [(t, k) for t in (3e-7, 2e-6) for k in range(4)]:
        g = grid(xs, Ds, tol=tol, pick=pick)
        if g is None: break
        kind, G, snapped, free, worst = g
        it = iter(snapped)
        ps = [dict(c=[[str(a), str(b)] for a, b in (next(it) for _ in range(3))], R=p["R"]) for p in pieces]
        d = dict(record_id=rid, piece=rec["piece"], container=rec["container"], n=len(ps), s=[str(s.a), str(s.b)],
                 source=os.path.relpath(fn, HERE), closed_form=cf["closed_form"], pieces=ps)
        code, msg = T.check(d, SOL, quiet=True)
        info = f"座標の形 {kind}、格子 1/{G}（許容値 {tol:.0e} で {pick + 1} 番目の候補）、格子からの最大のずれ {worst:.1e}、格子から外れていた（滑れる駒の）座標 {free}、回転を丸めた駒 {snapped_rot}"
        if code == 0 or first is None: first = (d, code, msg, info)
        if code == 0: break
    d, code, msg, info = first
    os.makedirs(OUT, exist_ok=True); json.dump(d, open(os.path.join(OUT, rid + ".json"), "w", encoding="utf-8"), indent=1)
    return ("EXACT-VALID" if code == 0 else msg.split("：")[0]), (msg + "。" if code else "") + info

def targets():
    out = []
    for f in sorted(glob.glob(os.path.join(PP, "closed_forms", "*.json"))):
        d = json.load(open(f, encoding="utf-8"))
        if d["piece"] in Q2SOL and d["container"] in Q2SOL and str(d.get("determined")) == "True" and d.get("closed_form") \
                and parse_s(d["closed_form"]) is not None and not d["record"].startswith("ours_"):
            out.append(d["record"])
    return out

if __name__ == "__main__":
    if sys.argv[1] != "--all":
        st, info = one(sys.argv[1]); print(f"{sys.argv[1]}\t{st}\t{info}"); sys.exit(0)
    NOWIN = getattr(subprocess, "CREATE_NO_WINDOW", 0); ids = targets(); print(len(ids), "件", flush=True)
    def run(rid):
        try:
            o = subprocess.run([sys.executable, "-I", "-X", "utf8", os.path.abspath(__file__), rid], capture_output=True, text=True,
                               encoding="utf-8", timeout=1800, creationflags=NOWIN)
            line = (o.stdout.strip().splitlines() or [f"{rid}\t異常終了\t{o.stderr.strip().splitlines()[-1] if o.stderr.strip() else ''}"])[-1]
        except subprocess.TimeoutExpired: line = f"{rid}\t時間切れ\t"
        print(line, flush=True); return line
    with ThreadPoolExecutor(max(1, os.cpu_count() - 4)) as ex: res = list(ex.map(run, ids))
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "summary.tsv"), "w", encoding="utf-8") as w:
        w.write("record\tstatus\tinfo\n"); w.writelines(l + "\n" for l in res)
