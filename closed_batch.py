# 閉じた式の一括処理：python closed_batch.py <known | unknown | confirm | ID ...> [--jobs 12] [--redo] [--quick] [--name 名前]
#   known   ：platonic-packing の SUMMARY.csv で閉じた式（closed_form_conjecture）があるもの。向こうの式と突き合わせる（検算）
#   unknown ：閉じた式がないもの
#   confirm ：--quick（60 桁のふるい分け）で多項式が出たものを、通常の設定（120 桁・解き直しあり）で確かめ直す
#   各 ID で algebraic3d.py を子プロセスで走らせる（窓は出さない。並列数の既定は CPU − 4）。
#   ログは data/logs/runs/<ID>.log、結果は data/closed_forms/<ID>.json（済んでいる ID は --redo がなければ飛ばす）、
#   要約は data/logs/batch_<名前>.tsv（終わるたびに書き直す）
import csv, json, os, re, sys, time, argparse, subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
import mpmath as mp

HERE = os.path.dirname(os.path.abspath(__file__)); PP = os.path.join(HERE, "data")
ap = argparse.ArgumentParser()
ap.add_argument("what", nargs="+"); ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 8) - 4))
ap.add_argument("--redo", action="store_true"); ap.add_argument("--quick", action="store_true"); ap.add_argument("--name", default=None); ap.add_argument("--timeout", type=int, default=3600)
args = ap.parse_args()

rows = {r["record_id"]: r for r in csv.DictReader(open(os.path.join(PP, "records", "SUMMARY.csv"), encoding="utf-8"))}
if args.what == ["known"]: ids = [k for k, r in rows.items() if r["closed_form_conjecture"]]
elif args.what == ["unknown"]: ids = [k for k, r in rows.items() if not r["closed_form_conjecture"]]
elif args.what == ["confirm"]:
    ids = []
    for k in rows:
        f = os.path.join(PP, "closed_forms", k + ".json")
        if os.path.exists(f):
            o = json.load(open(f, encoding="utf-8"))
            if o.get("quick") and o.get("poly"): ids.append(k)
    args.redo = True
else: ids = args.what
ids.sort(key=lambda k: int(rows[k]["n"]) if k in rows else 0)       # 小さい n から（速い）
name = args.name or (args.what[0] if len(args.what) == 1 else "ids")
os.makedirs(os.path.join(PP, "logs", "runs"), exist_ok=True)

mp.mp.dps = 130
def their_value(cf):
    """向こうの式（例 '(8√2)/3'、'-2 + 2√5'、'6 + √2'）を高精度の値に"""
    s = re.sub(r"(\d)\s*√", r"\1*√", cf.replace(" ", ""))
    s = re.sub(r"√(\d+)", r"sqrt(\1)", s); s = re.sub(r"√\(", "sqrt(", s)
    s = re.sub(r"(?<![\w.])(\d+)", r"M(\1)", s)       # 整数を高精度に（8/3 が倍精度の割り算になっていた）
    return mp.mpf(eval(s, {"sqrt": mp.sqrt, "M": mp.mpf}))

def run(rid):
    out = os.path.join(PP, "closed_forms", rid + ".json")
    if os.path.exists(out) and not args.redo: return rid, "skip"
    t = time.time(); log = open(os.path.join(PP, "logs", "runs", rid + ".log"), "w", encoding="utf-8")
    try:
        subprocess.run([sys.executable, "-X", "utf8", os.path.join(HERE, "algebraic3d.py"), rid] + (["--quick"] if args.quick else []), stdout=log, stderr=subprocess.STDOUT,
                       timeout=args.timeout, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        st = "ok"
    except subprocess.TimeoutExpired: st = "timeout"
    log.close(); return rid, f"{st} {time.time()-t:.0f}s"

def summary():
    lines = ["record_id\tstatus\tdigits\tclosed_form\tpoly\ttheirs\tagree_digits\ttol\tkernel\tkernel_S\tfamily"]
    for rid in ids:
        f = os.path.join(PP, "closed_forms", rid + ".json"); theirs = rows.get(rid, {}).get("closed_form_conjecture", "")
        if not os.path.exists(f):
            lines.append(f"{rid}\t{status.get(rid, 'pending')}"); continue
        o = json.load(open(f, encoding="utf-8"))
        if not o.get("converged"): st = "inconsistent"
        elif not o.get("determined"): st = "not_determined"
        elif o.get("poly"): st = "poly"
        else: st = "no_poly"
        agree = ""
        if theirs and o.get("s"):
            d = abs(their_value(theirs) - mp.mpf(o["s"])); agree = "%d" % int(-mp.log10(d)) if d > 0 else "all"
        lines.append("\t".join(str(v) for v in (rid, st, o.get("digits", ""), o.get("closed_form", ""), o.get("poly", ""), theirs, agree,
                                                 "%.1e" % o["tol"], o.get("kernel", ""), "%.1e" % o.get("kernel_s_component", 0), o.get("family", ""))))
    open(os.path.join(PP, "logs", f"batch_{name}.tsv"), "w", encoding="utf-8").write("\n".join(lines) + "\n")

status = {}
print(f"{len(ids)} 件を {args.jobs} 並列で", flush=True)
with ThreadPoolExecutor(args.jobs) as ex:
    futs = [ex.submit(run, rid) for rid in ids]
    for k, fu in enumerate(as_completed(futs), 1):
        rid, st = fu.result(); status[rid] = st
        print(f"[{k}/{len(ids)}] {rid} {st}", flush=True); summary()
summary(); print("終わり", flush=True)
