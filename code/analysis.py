# -*- coding: utf-8 -*-
"""Analysis of all runs: validity against arXiv primary categories, run-to-run stability, prevalence deviation,
label stability, failure rates, cost and time. All numbers are computed from runs/<tag>/*.json and logs/api_calls.jsonl."""
import json, os, glob, itertools, argparse, collections
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score as ARI, normalized_mutual_info_score as NMI
from common import load_corpus, ROOT

RNG = np.random.default_rng(20261122)


def load_runs(tag):
    R = []
    for p in sorted(glob.glob(os.path.join(ROOT, "runs", tag, "*.json"))):
        R.append(json.load(open(p, encoding="utf-8")))
    return R


def purity_p1(pred, gold):
    """Harmonic mean of purity and inverse purity (P1), as used for topic-model alignment (Pham et al., 2024)."""
    pred = np.asarray(pred); gold = np.asarray(gold)
    def pur(a, b):
        tot = 0
        for c in np.unique(a):
            m = b[a == c]
            tot += collections.Counter(m).most_common(1)[0][1]
        return tot / len(a)
    p, ip = pur(pred, gold), pur(gold, pred)
    return 2 * p * ip / (p + ip), p, ip


def shares_aligned(a, b, K):
    """Align topics of run b to run a by Hungarian matching on the contingency table, then return
    mean and max absolute difference of topic prevalence (percentage points)."""
    a = np.asarray(a); b = np.asarray(b)
    la = sorted(set(a)); lb = sorted(set(b))
    C = np.zeros((len(la), len(lb)))
    for i, x in enumerate(la):
        for j, y in enumerate(lb):
            C[i, j] = np.sum((a == x) & (b == y))
    r, c = linear_sum_assignment(-C)
    n = len(a)
    sa = {x: np.mean(a == x) for x in la}; sb = {y: np.mean(b == y) for y in lb}
    diffs = []
    matched_a = set(); matched_b = set()
    for i, j in zip(r, c):
        diffs.append(abs(sa[la[i]] - sb[lb[j]])); matched_a.add(la[i]); matched_b.add(lb[j])
    for x in la:
        if x not in matched_a: diffs.append(sa[x])
    for y in lb:
        if y not in matched_b: diffs.append(sb[y])
    d = np.array(diffs) * 100
    return float(d.mean()), float(d.max())


def boot_ci(vals, B=2000):
    v = np.asarray(vals, float)
    if len(v) < 2:
        return (float(v.mean()) if len(v) else float("nan"), float("nan"), float("nan"))
    bs = [RNG.choice(v, len(v), replace=True).mean() for _ in range(B)]
    return float(v.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def run_boot_pairwise(runs, metric, B=2000):
    """Bootstrap over runs (not over pairs) for the mean pairwise agreement within a set of runs."""
    n = len(runs)
    if n < 2:
        return float("nan"), float("nan"), float("nan")
    M = np.full((n, n), np.nan)
    for i, j in itertools.combinations(range(n), 2):
        M[i, j] = M[j, i] = metric(runs[i], runs[j])
    point = np.nanmean(M[np.triu_indices(n, 1)])
    bs = []
    for _ in range(B):
        idx = RNG.choice(n, n, replace=True)
        sub = M[np.ix_(idx, idx)]
        mask = ~np.eye(n, dtype=bool) & (idx[:, None] != idx[None, :])
        if mask.sum() == 0:
            continue
        bs.append(np.nanmean(sub[mask]))
    return float(point), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def cliffs_delta(x, y):
    x = np.asarray(x); y = np.asarray(y)
    gt = sum((xi > y).sum() for xi in x); lt = sum((xi < y).sum() for xi in x)
    return (gt - lt) / (len(x) * len(y))


def costs(tag):
    p = os.path.join(ROOT, "logs", "api_calls.jsonl")
    agg = collections.defaultdict(lambda: dict(cost=0.0, calls=0, fails=0, pt=0, ct=0, lat=0.0))
    if not os.path.exists(p):
        return agg
    for l in open(p, encoding="utf-8"):
        x = json.loads(l)
        if x.get("tag") != tag:
            continue
        a = agg[x["run_id"]]
        a["calls"] += 1
        if not x.get("ok"):
            a["fails"] += 1; continue
        a["cost"] += x.get("cost_usd") or 0; a["pt"] += x.get("prompt_tokens") or 0; a["ct"] += x.get("completion_tokens") or 0
        a["lat"] += x.get("latency_s") or 0
    return agg


def main(tag):
    runs = load_runs(tag)
    gold_by_corpus = {}
    for c in sorted(set(r["corpus"] for r in runs)):
        rows = load_corpus(c)
        gold_by_corpus[c] = {r["id"]: r["primary"] for r in rows}
    cst = costs(tag)
    per_run = []
    for r in runs:
        gold = [gold_by_corpus[r["corpus"]][d] for d in r["doc_ids"]]
        pred = r["labels"]
        p1, pu, ip = purity_p1(pred, gold)
        c = cst.get(r["run_id"], {})
        per_run.append(dict(run_id=r["run_id"], corpus=r["corpus"], method=r["model"], cond=r["cond"], rep=r["rep"],
                            NMI=NMI(gold, pred), ARI=ARI(gold, pred), P1=p1, purity=pu, inv_purity=ip,
                            topics_used=len(set(x for x in pred if x != -1)), unassigned_pct=100 * np.mean(np.asarray(pred) == -1),
                            n_docs=len(pred), wall_s=r.get("wall_s"), cost_usd=c.get("cost", 0.0), api_calls=c.get("calls", 0),
                            api_fail=c.get("fails", 0), prompt_tokens=c.get("pt", 0), completion_tokens=c.get("ct", 0)))
    # stability within (corpus, method, cond)
    groups = collections.defaultdict(list)
    for r in runs:
        groups[(r["corpus"], r["model"], r["cond"])].append(r)
    stab = []
    ari_f = lambda a, b: ARI(a["labels"], b["labels"])
    nmi_f = lambda a, b: NMI(a["labels"], b["labels"])
    prev_f = lambda a, b: shares_aligned(a["labels"], b["labels"], a["K"])[0]
    prevmax_f = lambda a, b: shares_aligned(a["labels"], b["labels"], a["K"])[1]
    for (c, m, cond), rs in sorted(groups.items()):
        if len(rs) < 2:
            continue
        stab.append(dict(corpus=c, method=m, cond=cond, n_runs=len(rs), pairwise_ARI=run_boot_pairwise(rs, ari_f),
                         pairwise_NMI=run_boot_pairwise(rs, nmi_f), prevalence_MAD_pp=run_boot_pairwise(rs, prev_f),
                         prevalence_maxdev_pp=run_boot_pairwise(rs, prevmax_f),
                         pairwise_ARI_values=[ari_f(a, b) for a, b in itertools.combinations(rs, 2)]))
    # cross-condition agreement: perturbed run vs each base run of same model
    cross = []
    for (c, m, cond), rs in sorted(groups.items()):
        if cond == "base" or (c, m, "base") not in groups:
            continue
        base = groups[(c, m, "base")]
        vals = [ARI(p["labels"], b["labels"]) for p in rs for b in base]
        prev = [shares_aligned(p["labels"], b["labels"], p["K"])[0] for p in rs for b in base]
        cross.append(dict(corpus=c, method=m, cond=cond, n_pert=len(rs), n_base=len(base), ARI_vs_base=boot_ci(vals),
                          prevalence_MAD_vs_base=boot_ci(prev), ARI_values=vals))
    out = dict(per_run=per_run, stability=stab, cross=cross)
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    json.dump(out, open(os.path.join(ROOT, "results", f"analysis_{tag}.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", default="main")
    a = ap.parse_args()
    o = main(a.tag)
    for s in o["stability"]:
        print(s["corpus"], s["method"], s["cond"], s["n_runs"], "ARI %.3f [%.3f, %.3f]" % s["pairwise_ARI"], "prevMAD %.2f" % s["prevalence_MAD_pp"][0])
    for r in o["per_run"][:50]:
        print(r["run_id"], "NMI %.3f ARI %.3f P1 %.3f used %d unas %.1f%% cost %.4f" % (r["NMI"], r["ARI"], r["P1"], r["topics_used"], r["unassigned_pct"], r["cost_usd"]))
