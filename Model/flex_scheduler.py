#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
flex_scheduler.py — 跨单元全局调度: 4 单元 (best + 3 seeds) 并行共享 ≤6 作业槽
最快跑完 32 折训练 + 4 次推演。作业: fold (python run.py F) 与 analysis
(python3 analysis.py, 单并发); analysis 等本单元 8 折齐后自动执行。

用法:
  python3 flex_scheduler.py                   # 全跑 4 单元
  python3 flex_scheduler.py best best_new_seed3   # 指定单元
  touch Model/.flex_stop                      # 优雅停止 (当前作业完成后退出)
断点续跑: fold 日志含 "foldF EXIT:0" 尾行 → 自动跳过; score 新于训练日志 → 跳过 analysis。
"""
import os
import subprocess
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor

PY = "/autodl-fs/data/miniconda3/bin/python3"
ROOT = "/autodl-fs/data/lingqiData/Model"
STOP = os.path.join(ROOT, ".flex_stop")
SLOTS = int(os.environ.get("CONC", "6"))
UNITS = sys.argv[1:] or ["best", "best_new_seed1", "best_new_seed2",
                         "best_new_seed3"]
FOLDS = list(range(1, 9))
LOG = os.path.join(ROOT, "flex_scheduler.log")


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def fold_done(unit, f):
    p = os.path.join(ROOT, unit, "logs", f"fold{f}.log")
    if not os.path.exists(p):
        return False
    with open(p, errors="ignore") as fh:
        return f"fold{f} EXIT:0" in fh.read()


def analysis_done(unit):
    score = os.path.join(ROOT, unit, "model_pred", "2026q3",
                         "all_zscore_score.fea")
    if not os.path.exists(score):
        return False
    newest = 0.0
    for fn in os.listdir(os.path.join(ROOT, unit, "logs")):
        if fn.startswith("fold") and fn.endswith(".log"):
            newest = max(newest, os.path.getmtime(
                os.path.join(ROOT, unit, "logs", fn)))
    return newest > 0 and os.path.getmtime(score) >= newest


def run_job(cmd, cwd, out_path, tag):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "a") as fh:
        p = subprocess.Popen(cmd, cwd=cwd, stdout=fh,
                             stderr=subprocess.STDOUT)
    log(f"  启动 {tag} (pid {p.pid})")
    rc = p.wait()
    log(f"  完成 {tag} rc={rc}")
    return rc


def worker(item):
    kind, u, f = item
    cwd = os.path.join(ROOT, u)
    if kind == "fold":
        out = os.path.join(cwd, "logs", f"fold{f}.log")
        subprocess.run(["rm", "-rf", os.path.join(
            cwd, "model_train", "2026q3", f"fold{f}")])
        log(f"  [fold] {u}/fold{f} 启动")
        rc = run_job([PY, "run.py", str(f)], cwd, out, f"{u}/fold{f}")
        with open(out, "a") as fh:
            fh.write(f"fold{f} EXIT:{rc}\n")
        return kind, u, f, rc
    out = os.path.join(cwd, "analysis_run.log")
    log(f"  [analysis] {u} 启动")
    rc = run_job([PY, "-u", "analysis.py"], cwd, out, f"{u}/analysis")
    return kind, u, None, rc


def main():
    if os.path.exists(STOP):
        os.remove(STOP)
    # 作业顺序: 各单元折 1 先行 → 其余折轮转 → analysis 尾部 (动态放行)
    order = []
    for u in UNITS:
        order.append(("fold", u, FOLDS[0]))
    for f in FOLDS[1:]:
        for u in UNITS:
            order.append(("fold", u, f))
    for u in UNITS:
        order.append(("analysis", u, None))
    q = deque(item for item in order
              if not (item[0] == "fold" and fold_done(item[1], item[2]))
              and not (item[0] == "analysis" and analysis_done(item[1])))
    log(f"调度开始: units={UNITS}, 待办={len(q)} (32折+4推演), 并发槽={SLOTS}")
    results = {}
    with ThreadPoolExecutor(max_workers=SLOTS) as ex:
        futs = {}
        while q or futs:
            for item in list(q):
                kind, u, f = item
                if kind == "analysis" and not all(
                        fold_done(u, k) for k in FOLDS):
                    continue                    # 等本单元折齐
                if kind == "analysis" and any(
                        t[0] == "analysis" for t in futs.values()):
                    continue                    # analysis 单并发
                if len(futs) >= SLOTS:
                    break
                q.remove(item)
                futs[ex.submit(worker, item)] = item
            if not futs and not q:
                break
            done = next((fu for fu in futs if fu.done()), None)
            if done is None:
                time.sleep(2)
                continue
            item = futs.pop(done)
            kind, u, f, rc = done.result()
            results[(u, kind, f)] = rc
            log(f"  [done] {u} {kind} {f or ''} rc={rc}")
            if os.path.exists(STOP):
                log("收到停止信号 — 不再派发新作业")
                q.clear()
    log("调度器退出; 汇总: " + "; ".join(
        f"{u}:{'OK' if all(results.get((u,'fold',f)) == 0 for f in FOLDS) and results.get((u,'analysis',None)) == 0 else 'CHECK'}"
        for u in UNITS))


if __name__ == "__main__":
    main()
