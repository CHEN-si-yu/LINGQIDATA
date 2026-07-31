#!/usr/bin/env python3
"""
对比两个 debug 快照 v3 — 逐日横截面数据指纹 + finance 逐字段追踪。

用法：
    python debug/compare.py debug/20260729 debug/20260730

对比逻辑：
    1. ★ 源数据逐日指纹 — 同一日期在不同快照中的 MD5 是否一致
    2. ★ finance 逐字段 MD5 — 精确到17个字段哪一个变了
    3. 因子文件 — 历史日期的 top10/bottom10 股票是否变化
    4. fac_all 因子值 — 相同日期的因子值是否漂移
    5. 模型预测 — 相同因子日期的 top-N 排名是否变化
"""

import json, sys
from pathlib import Path


def load_json(path):
    with open(path) as f:
        return json.load(f)


def compare(dir_a, dir_b):
    a_name = Path(dir_a).name
    b_name = Path(dir_b).name
    print(f"{'='*70}")
    print(f"  COMPARING: {a_name} vs {b_name}")
    print(f"{'='*70}")

    issues = []

    # ── 1. ★ 逐日横截面数据指纹（核心） ──
    print(f"\n{'─'*70}")
    print("[1] SOURCE DATE HASHES — per-date cross-sectional data fingerprint")
    print(f"{'─'*70}")

    try:
        dha = load_json(f"{dir_a}/source_date_hashes.json")
        dhb = load_json(f"{dir_b}/source_date_hashes.json")
    except FileNotFoundError as e:
        print(f"  ERROR: {e}")
        dha = dhb = {}

    if dha and dhb:
        for name in sorted(set(list(dha.keys()) + list(dhb.keys()))):
            a_info = dha.get(name, {})
            b_info = dhb.get(name, {})

            if a_info.get("error") or b_info.get("error"):
                status = "ERROR" if a_info.get("error") else "ERROR"
                print(f"  ? {name}: {a_info.get('error') or b_info.get('error')}")
                continue

            a_dates = a_info.get("last7_dates", a_info.get("dates", {}))
            b_dates = b_info.get("last7_dates", b_info.get("dates", {}))

            common_dates = sorted(set(a_dates.keys()) & set(b_dates.keys()))
            only_b = sorted(set(b_dates.keys()) - set(a_dates.keys()))

            changed_dates = []
            for d in common_dates:
                ha = a_dates[d].get("md5")
                hb = b_dates[d].get("md5")
                na = a_dates[d].get("n_stocks", 0)
                nb = b_dates[d].get("n_stocks", 0)

                if ha and hb and ha != hb:
                    changed_dates.append((d, na, nb, ha, hb))

            if changed_dates:
                print(f"\n  ★★★ {name}: {len(changed_dates)} HISTORICAL DATE(S) CHANGED! ★★★")
                for d, na, nb, ha, hb in changed_dates:
                    stocks_info = f"{na}→{nb} stocks" if na != nb else f"{nb} stocks"
                    print(f"      {d}: {stocks_info}, MD5 {ha[:16]} → {hb[:16]}")
                    issues.append(f"DATA REVISION: {name} date {d} changed!")
            else:
                n_common = len(common_dates)
                if n_common > 0:
                    print(f"  ✓ {name}: {n_common} common dates — all identical")
                else:
                    print(f"  ~ {name}: no common dates to compare")

            if only_b:
                new_dates_str = ", ".join(only_b[-3:])
                print(f"    (new dates in {b_name}: {new_dates_str})")

    # ── 2. ★ Finance 逐字段 MD5 对比 ──
    print(f"\n{'─'*70}")
    print("[2] FINANCE PER-FIELD — which specific fields changed?")
    print(f"{'─'*70}")

    if dha and dhb and "finance" in dha and "finance" in dhb:
        fa_dates = dha["finance"].get("last7_dates", dha["finance"].get("dates", {}))
        fb_dates = dhb["finance"].get("last7_dates", dhb["finance"].get("dates", {}))
        common = sorted(set(fa_dates.keys()) & set(fb_dates.keys()))

        for d in common:
            fa = fa_dates[d]
            fb = fb_dates[d]

            # 先看整体 MD5
            composite_changed = fa.get("md5") != fb.get("md5")
            if not composite_changed:
                print(f"  ✓ {d}: composite unchanged")
                continue

            # 逐字段对比
            fa_fields = fa.get("fields", {})
            fb_fields = fb.get("fields", {})

            if not fa_fields or not fb_fields:
                print(f"  ★ {d}: composite MD5 changed (no per-field data)")
                continue

            changed_fields = []
            stable_fields = []
            for field in sorted(set(list(fa_fields.keys()) + list(fb_fields.keys()))):
                fha = fa_fields.get(field, {}).get("md5")
                fhb = fb_fields.get(field, {}).get("md5")
                if fha and fhb and fha != fhb:
                    changed_fields.append(field)
                else:
                    stable_fields.append(field)

            if changed_fields:
                print(f"\n  ★★★ finance {d}: composite MD5 CHANGED ★★★")
                print(f"      变动字段 ({len(changed_fields)}/{len(changed_fields)+len(stable_fields)}):")
                for f in changed_fields:
                    print(f"        🔴 {f}")
                if stable_fields:
                    print(f"      未变动字段 ({len(stable_fields)}):")
                    # 最多显示10个未变字段
                    show_stable = stable_fields[:10]
                    for f in show_stable:
                        print(f"        ✅ {f}")
                    if len(stable_fields) > 10:
                        print(f"        ... (+{len(stable_fields)-10} more)")
                issues.append(f"FINANCE {d}: {len(changed_fields)} fields changed ({', '.join(changed_fields[:5])}...)")

    # ── 3. 因子文件对比 ──
    print(f"\n{'─'*70}")
    print("[3] FACTOR FILES — checking if historical factor VALUES changed")
    print(f"{'─'*70}")

    try:
        fac_a = load_json(f"{dir_a}/factor_sample.json")
        fac_b = load_json(f"{dir_b}/factor_sample.json")
    except FileNotFoundError:
        fac_a = fac_b = {}

    changed_factors = []
    for fn in sorted(set(list(fac_a.keys()) + list(fac_b.keys()))):
        a = fac_a.get(fn, {})
        b = fac_b.get(fn, {})

        a_vals = a.get("last3_values", {})
        b_vals = b.get("last3_values", {})

        common_dates = sorted(set(a_vals.keys()) & set(b_vals.keys()))

        for d in common_dates:
            va = a_vals.get(d, {})
            vb = b_vals.get(d, {})
            if not va or not vb:
                continue

            ta = set(va.get("top10", {}).keys())
            tb = set(vb.get("top10", {}).keys())
            new_in = tb - ta
            dropped = ta - tb

            if new_in or dropped:
                if fn not in changed_factors:
                    changed_factors.append(fn)
                print(f"\n  ★ FACTOR {fn} DATE {d}: top10 changed!")
                print(f"      +{len(new_in)} new in top10: {sorted(new_in)[:5]}...")
                print(f"      -{len(dropped)} dropped: {sorted(dropped)[:5]}...")
                issues.append(
                    f"FACTOR {fn} date {d}: top10 changed +{len(new_in)}/-{len(dropped)}"
                )

    if not changed_factors:
        print("  ✓ All common-date factor values are stable")

    # ── 4. fac_all.fea 对比 ──
    print(f"\n{'─'*70}")
    print("[4] fac_all.fea — checking if same-date factor values drifted")
    print(f"{'─'*70}")

    try:
        fall_a = load_json(f"{dir_a}/fac_all_meta.json")
        fall_b = load_json(f"{dir_b}/fac_all_meta.json")
    except FileNotFoundError:
        fall_a = fall_b = {}

    if fall_a.get("shape") != fall_b.get("shape"):
        print(f"  Shape: {fall_a.get('shape')} → {fall_b.get('shape')} (new dates added)")

    watch = fall_a.get("watched_factors", [])
    for key in sorted(fall_a.keys()):
        if not key.startswith("date_") or not key.endswith("_stats"):
            continue
        if key not in fall_b:
            continue
        date_str = key.replace("date_", "").replace("_stats", "")
        fa = fall_a[key]
        fb = fall_b[key]
        big_diffs = []
        for wf in watch:
            if wf not in fa or wf not in fb:
                continue
            mean_diff = abs(fa[wf].get("mean", 0) - fb[wf].get("mean", 0))
            n_diff = fa[wf].get("n", 0) - fb[wf].get("n", 0)
            if mean_diff > 0.0001 or n_diff != 0:
                big_diffs.append((wf, mean_diff, n_diff))
        if big_diffs:
            print(f"  ★ DATE {date_str}: {len(big_diffs)}/{len(watch)} factors drifted!")
            for wf, md, nd in big_diffs[:10]:
                print(f"      {wf}: mean δ={md:.6f}, n δ={nd}")
            issues.append(f"FAC_ALL date {date_str}: {len(big_diffs)} factors drifted")
        else:
            print(f"  ✓ DATE {date_str}: all {len(watch)} watched factors stable")

    # ── 5. 模型预测对比 ──
    print(f"\n{'─'*70}")
    print("[5] MODEL PREDICTIONS")
    print(f"{'─'*70}")

    try:
        pred_a = load_json(f"{dir_a}/model_pred_full.json")
        pred_b = load_json(f"{dir_b}/model_pred_full.json")
    except FileNotFoundError:
        pred_a = pred_b = {}

    for season in sorted(set(list(pred_a.keys()) + list(pred_b.keys()))):
        pa = pred_a.get(season, {})
        pb = pred_b.get(season, {})
        for date_str in sorted(set(list(pa.keys()) + list(pb.keys()))):
            a = pa.get(date_str, {})
            b = pb.get(date_str, {})

            if not a or not b:
                if a:
                    print(f"  ? {season}/{date_str}: only in {a_name}")
                elif b:
                    print(f"  + {season}/{date_str}: new in {b_name}")
                continue

            ta = a.get("top10", {})
            tb = b.get("top10", {})

            a_ranked = list(ta.keys())
            b_ranked = list(tb.keys())

            if a_ranked != b_ranked:
                print(f"\n  ★ {season}/{date_str}: TOP-10 RANKING CHANGED!")
                for i in range(min(len(a_ranked), len(b_ranked))):
                    if a_ranked[i] != b_ranked[i]:
                        a_score = ta.get(a_ranked[i], "?")
                        b_score = tb.get(b_ranked[i], "?")
                        print(f"    Rank {i+1}: {a_ranked[i]} ({a_score}) → {b_ranked[i]} ({b_score})")

                old_top1 = a_ranked[0] if a_ranked else "?"
                new_top1 = b_ranked[0] if b_ranked else "?"
                if old_top1 != new_top1:
                    print(f"    ★ #1 CHANGED: {old_top1} → {new_top1}")
                    issues.append(f"PRED {season}/{date_str}: #1 {old_top1} → {new_top1}")
            else:
                max_score_diff = 0
                for code in a_ranked:
                    sa = ta.get(code, 0)
                    sb = tb.get(code, 0)
                    max_score_diff = max(max_score_diff, abs(sa - sb))
                if max_score_diff > 0.01:
                    print(f"  ⚠ {season}/{date_str}: ranking same but scores drifted (max δ={max_score_diff:.4f})")
                else:
                    print(f"  ✓ {season}/{date_str}: top-10 identical")

    # ── 汇总 ──
    print(f"\n{'='*70}")
    print(f"  COMPARISON SUMMARY: {a_name} → {b_name}")
    print(f"{'='*70}")
    if issues:
        print(f"\n  ⚠ {len(issues)} issue(s) detected:")
        for i, issue in enumerate(issues, 1):
            if "DATA REVISION" in issue:
                prefix = "🔴"
            elif "FINANCE" in issue:
                prefix = "🔵"
            elif "FACTOR" in issue:
                prefix = "🟡"
            elif "PRED" in issue:
                prefix = "🟠"
            else:
                prefix = "  "
            print(f"  {prefix} {i}. {issue}")
    else:
        print(f"\n  ✓ No issues detected between snapshots.")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python compare.py <dir_a> <dir_b>")
        print("Example: python compare.py debug/20260729 debug/20260730")
        sys.exit(1)
    compare(sys.argv[1], sys.argv[2])
