# -*- coding: utf-8 -*-
"""Run all pre-specified conditions (see S04 pre-specified plan). Resumable: finished runs are skipped.
Usage: python run_all.py [--part base|perturb|all] [--budget 3.0]"""
import os, json, argparse, glob
from concurrent.futures import ThreadPoolExecutor
from llm_topics import run, RUNDIR
from common import ROOT

MODELS = ["openai/gpt-4o-mini", "google/gemini-2.5-flash-lite", "mistralai/mistral-small-3.2-24b-instruct", "deepseek/deepseek-v3.2"]
PERT_MODELS = ["openai/gpt-4o-mini", "mistralai/mistral-small-3.2-24b-instruct"]


def spent():
    p = os.path.join(ROOT, "logs", "api_calls.jsonl")
    if not os.path.exists(p):
        return 0.0
    s = 0.0
    for l in open(p, encoding="utf-8"):
        x = json.loads(l)
        if x.get("tag") == "main":
            s += x.get("cost_usd") or 0
    return s


def jobs(part):
    J = []
    if part in ("base", "all"):
        for c in ["CS8", "QBIO6"]:
            for m in MODELS:
                for r in range(1, 6):
                    J.append(dict(corpus=c, model=m, cond="base", rep=r))
    if part in ("perturb", "all"):
        for m in PERT_MODELS:
            for r in range(1, 4):
                J.append(dict(corpus="CS8", model=m, cond="temp", rep=r, temperature=0.7))
                J.append(dict(corpus="CS8", model=m, cond="prompt", rep=r, prompt_id=r))
                J.append(dict(corpus="CS8", model=m, cond="gensample", rep=r, gen_seed=r))
                J.append(dict(corpus="CS8", model=m, cond="order", rep=r, order_seed=r))
                J.append(dict(corpus="CS8", model=m, cond="fixedtopics", rep=r, order_seed=0, fixed="base_r1"))
        for r in range(1, 4):
            J.append(dict(corpus="QBIO6", model="openai/gpt-4o-mini", cond="prompt", rep=r, prompt_id=r))
            J.append(dict(corpus="QBIO6", model="openai/gpt-4o-mini", cond="gensample", rep=r, gen_seed=r))
    return J


def do(j, budget):
    if spent() > budget:
        return f"SKIP budget {j}"
    fixed = None
    if j.get("fixed"):
        ms = j["model"].split("/")[-1]
        fixed = os.path.join(RUNDIR, "main", f"{j['corpus']}__{ms}__base__r1.json")
    try:
        o = run(j["corpus"], j["model"], j["cond"], j["rep"], j.get("temperature", 0.0), j.get("prompt_id", 0), j.get("gen_seed", 0),
                j.get("order_seed", 0), fixed, None, "main")
        return f"{o['run_id']} topics={o['n_topics_returned']} missing={o['missing_docs']} invalid={o['invalid_topic']} wall={o['wall_s']}"
    except Exception as e:
        return f"FAIL {j}: {e}"


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--part", default="all"); ap.add_argument("--budget", type=float, default=3.0)
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    J = jobs(a.part)
    print("jobs", len(J), "spent so far", round(spent(), 4), flush=True)
    with ThreadPoolExecutor(a.workers) as ex:
        for msg in ex.map(lambda j: do(j, a.budget), J):
            print(msg, "| spent", round(spent(), 4), flush=True)
    print("DONE spent", round(spent(), 4))
