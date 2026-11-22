# -*- coding: utf-8 -*-
"""Two-stage LLM topic pipeline (topic generation on a document sample, then batched assignment).
Simplified TopicGPT-style design (Pham et al., 2024) with a fixed number of topics K so that
outputs are directly comparable with LDA/NMF/embedding-clustering baselines."""
import json, os, random, sys, time, argparse
from concurrent.futures import ThreadPoolExecutor
from common import load_corpus, doc_text, short_text, chat, extract_json, ROOT

RUNDIR = os.path.join(ROOT, "runs")
os.makedirs(RUNDIR, exist_ok=True)
G = 60          # documents shown in the generation stage
BATCH = 20      # documents per assignment call

GEN_PROMPTS = [
 # P0 (base)
 "You are analysing a collection of scientific paper abstracts. Below is a random sample of documents from the collection.\n"
 "Identify exactly {K} high-level research topics that together cover the whole collection. Topics must be mutually distinct and at a similar level of granularity.\n"
 "Return JSON only, in this format: {{\"topics\": [{{\"id\": 1, \"label\": \"2-5 word label\", \"description\": \"one sentence\"}}]}} with ids 1..{K}.\n\nSample documents:\n{DOCS}",
 # P1
 "Task: topic discovery for a corpus of research abstracts. Read the sampled documents below and propose {K} broad topics that jointly cover the corpus, "
 "with no overlap between topics and comparable breadth.\nOutput only JSON: {{\"topics\": [{{\"id\": 1, \"label\": \"short label\", \"description\": \"one-sentence definition\"}}]}}, ids from 1 to {K}.\n\nDocuments:\n{DOCS}",
 # P2
 "The following texts are a random subset of a scientific literature collection. Summarise the collection as a set of {K} distinct research areas "
 "(similar granularity, non-overlapping). For each area give a concise name and a one-sentence scope note.\n"
 "Respond with JSON only: {{\"topics\": [{{\"id\": 1, \"label\": \"name\", \"description\": \"scope note\"}}]}} (ids 1..{K}).\n\nTexts:\n{DOCS}",
 # P3
 "As a bibliometric analyst, derive a flat list of {K} topics for the collection represented by the sample below. "
 "Each topic should be specific enough to separate documents but general enough to cover many of them; avoid overlapping topics.\n"
 "JSON only: {{\"topics\": [{{\"id\": 1, \"label\": \"label\", \"description\": \"one sentence\"}}]}} with ids 1..{K}.\n\nSample:\n{DOCS}",
]
ASSIGN_PROMPTS = [
 "Topics:\n{TOPICS}\n\nAssign each document below to the single best-matching topic from the list.\n"
 "Return JSON only: {{\"assignments\": [{{\"doc\": <document number>, \"topic\": <topic id>}}]}} with exactly one entry per document.\n\nDocuments:\n{DOCS}",
 "Topic list:\n{TOPICS}\n\nFor every document below, choose the one topic id that fits it best.\n"
 "Output JSON only: {{\"assignments\": [{{\"doc\": n, \"topic\": id}}]}}, one entry for each document number.\n\nDocuments:\n{DOCS}",
 "Candidate research areas:\n{TOPICS}\n\nClassify each text below into exactly one of these areas.\n"
 "Respond with JSON only: {{\"assignments\": [{{\"doc\": text number, \"topic\": area id}}]}} covering all texts.\n\nTexts:\n{DOCS}",
 "Topics derived for this collection:\n{TOPICS}\n\nLabel each document in the sample below with the id of its most appropriate topic.\n"
 "JSON only: {{\"assignments\": [{{\"doc\": number, \"topic\": id}}]}}; include every document exactly once.\n\nSample:\n{DOCS}",
]
SYSTEM = "You are a careful research assistant. Follow the output format exactly."


def gen_topics(rows, K, model, temperature, prompt_id, gen_seed, meta):
    rnd = random.Random(1000 + gen_seed)
    idx = rnd.sample(range(len(rows)), min(G, len(rows)))
    docs = "\n".join(f"[{i+1}] {short_text(rows[j])}" for i, j in enumerate(idx))
    prompt = GEN_PROMPTS[prompt_id].format(K=K, DOCS=docs)
    for attempt in range(3):
        txt, rec = chat([{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}], model, temperature,
                        max_tokens=1500, meta=dict(meta, stage="generate", gen_attempt=attempt))
        js = extract_json(txt)
        if js and isinstance(js, dict) and isinstance(js.get("topics"), list) and len(js["topics"]) >= 2:
            tps = [{"id": int(t.get("id", k + 1)), "label": str(t.get("label", "")).strip(), "description": str(t.get("description", "")).strip()}
                   for k, t in enumerate(js["topics"])]
            return tps, idx, attempt
    raise RuntimeError("topic generation failed")


def assign(rows, order, topics, model, temperature, prompt_id, meta, workers=8):
    tlist = "\n".join(f"{t['id']}. {t['label']}: {t['description']}" for t in topics)
    valid = {t["id"] for t in topics}
    batches = [order[i:i + BATCH] for i in range(0, len(order), BATCH)]
    result = {}
    stats = {"parse_fail_batches": 0, "missing_docs": 0, "invalid_topic": 0}

    def run_batch(bi):
        b = batches[bi]
        docs = "\n".join(f"[{k+1}] {doc_text(rows[j])}" for k, j in enumerate(b))
        prompt = ASSIGN_PROMPTS[prompt_id].format(TOPICS=tlist, DOCS=docs)
        out = {}
        fails = 0
        for attempt in range(2):
            txt, rec = chat([{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}], model, temperature,
                            max_tokens=1200, meta=dict(meta, stage="assign", batch=bi, assign_attempt=attempt))
            js = extract_json(txt)
            arr = js.get("assignments") if isinstance(js, dict) else (js if isinstance(js, list) else None)
            if isinstance(arr, list):
                for a in arr:
                    try:
                        d = int(a.get("doc")); tpc = int(a.get("topic"))
                    except Exception:
                        continue
                    if 1 <= d <= len(b):
                        out[b[d - 1]] = tpc
                if len(out) >= len(b) * 0.9:
                    break
            fails += 1
        return bi, out, fails

    with ThreadPoolExecutor(workers) as ex:
        for bi, out, fails in ex.map(run_batch, range(len(batches))):
            stats["parse_fail_batches"] += 1 if fails >= 2 else 0
            for j in batches[bi]:
                if j not in out:
                    stats["missing_docs"] += 1; result[j] = -1
                elif out[j] not in valid:
                    stats["invalid_topic"] += 1; result[j] = -1
                else:
                    result[j] = out[j]
    return [result[j] for j in range(len(rows))], stats


def run(corpus, model, cond, rep, temperature=0.0, prompt_id=0, gen_seed=0, order_seed=0, fixed_topics_from=None, n_docs=None, tag="main"):
    rows = load_corpus(corpus)
    if n_docs:
        rows = rows[::max(1, len(rows) // n_docs)][:n_docs]
    K = len(sorted(set(r["primary"] for r in rows)))
    mshort = model.split("/")[-1]
    run_id = f"{corpus}__{mshort}__{cond}__r{rep}"
    path = os.path.join(RUNDIR, tag, run_id + ".json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        return json.load(open(path, encoding="utf-8"))
    meta = dict(run_id=run_id, corpus=corpus, cond=cond, rep=rep, tag=tag)
    t0 = time.time()
    if fixed_topics_from:
        src = json.load(open(fixed_topics_from, encoding="utf-8"))
        topics, gen_idx, gen_attempts = src["topics"], src["gen_sample"], 0
    else:
        topics, gen_idx, gen_attempts = gen_topics(rows, K, model, temperature, prompt_id, gen_seed, meta)
    order = list(range(len(rows)))
    random.Random(2000 + order_seed).shuffle(order)
    labels, st = assign(rows, order, topics, model, temperature, prompt_id, meta)
    out = dict(run_id=run_id, corpus=corpus, model=model, cond=cond, rep=rep, K=K, temperature=temperature, prompt_id=prompt_id,
               gen_seed=gen_seed, order_seed=order_seed, fixed_topics_from=os.path.basename(fixed_topics_from) if fixed_topics_from else None,
               topics=topics, n_topics_returned=len(topics), gen_sample=gen_idx, gen_attempts=gen_attempts, labels=labels,
               doc_ids=[r["id"] for r in rows], wall_s=round(time.time() - t0, 2), **st,
               finished=time.strftime("%Y-%m-%d %H:%M:%S"))
    json.dump(out, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=0)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus"); ap.add_argument("--model"); ap.add_argument("--cond", default="base"); ap.add_argument("--rep", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.0); ap.add_argument("--prompt_id", type=int, default=0)
    ap.add_argument("--gen_seed", type=int, default=0); ap.add_argument("--order_seed", type=int, default=0)
    ap.add_argument("--fixed_topics_from", default=None); ap.add_argument("--n_docs", type=int, default=None); ap.add_argument("--tag", default="main")
    a = ap.parse_args()
    o = run(a.corpus, a.model, a.cond, a.rep, a.temperature, a.prompt_id, a.gen_seed, a.order_seed, a.fixed_topics_from, a.n_docs, a.tag)
    print(o["run_id"], "K", o["K"], "topics", o["n_topics_returned"], "missing", o["missing_docs"], "invalid", o["invalid_topic"], "wall", o["wall_s"])


