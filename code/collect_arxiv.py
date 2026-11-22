"""Collect arXiv metadata (CC0 1.0) for two labelled corpora, sampled across quarterly windows to avoid temporal clustering."""
import urllib.request,urllib.parse,xml.etree.ElementTree as ET,time,json,random,sys,os
out=sys.argv[1]
ns={'a':'http://www.w3.org/2005/Atom','arxiv':'http://arxiv.org/schemas/atom'}
CORPORA={
 "CS8":dict(cats=["cs.CV","cs.CL","cs.CR","cs.RO","cs.DB","cs.NI","cs.SE","cs.DS"],years=[2024],per_q=150),
 "QBIO6":dict(cats=["q-bio.NC","q-bio.PE","q-bio.BM","q-bio.GN","q-bio.MN","q-bio.QM"],years=[2023,2024],per_q=80),
}
QS=[("01010000","03312359"),("04010000","06302359"),("07010000","09302359"),("10010000","12312359")]
N_PER=100; SEED=20261122
def get(u):
    for t in range(5):
        try: return urllib.request.urlopen(u,timeout=120).read()
        except Exception as e: print('retry',e); time.sleep(10)
def fetch(cat,years,per_q):
    recs={}
    for y in years:
        for a,b in QS:
            q=f"cat:{cat} AND submittedDate:[{y}{a} TO {y}{b}]"
            u="http://export.arxiv.org/api/query?"+urllib.parse.urlencode({"search_query":q,"start":0,"max_results":per_q,"sortBy":"submittedDate","sortOrder":"ascending"})
            x=get(u); time.sleep(3.1)
            for e in ET.fromstring(x).findall('a:entry',ns):
                prim=e.find('arxiv:primary_category',ns).attrib['term']
                if prim!=cat: continue
                i=e.find('a:id',ns).text.split('/abs/')[-1]
                recs[i]=dict(id=i,title=' '.join(e.find('a:title',ns).text.split()),abstract=' '.join(e.find('a:summary',ns).text.split()),
                    primary=prim,categories=[c.attrib['term'] for c in e.findall('a:category',ns)],published=e.find('a:published',ns).text[:10])
    return list(recs.values())
meta={}
for name,cfg in CORPORA.items():
    rows=[]
    for c in cfg['cats']:
        r=sorted(fetch(c,cfg['years'],cfg['per_q']),key=lambda d:d['id']); rnd=random.Random(SEED); rnd.shuffle(r)
        pick=r[:N_PER]; rows+=pick; meta[c]=dict(primary_matches_pool=len(r),sampled=len(pick))
        print(name,c,'pool',len(r),'sampled',len(pick),flush=True)
    json.dump(rows,open(os.path.join(out,f"{name}.json"),'w',encoding='utf-8'),ensure_ascii=False)
json.dump(dict(source="arXiv API export.arxiv.org (metadata released under CC0 1.0)",corpora={k:dict(cats=v['cats'],years=v['years']) for k,v in CORPORA.items()},sampling="quarterly windows, earliest-submitted records per quarter returned by API, primary-category filter, then seeded shuffle",n_per_category=N_PER,seed=SEED,per_category=meta,collected=time.strftime('%Y-%m-%d %H:%M:%S')),open(os.path.join(out,"collection_log.json"),'w',encoding='utf-8'),indent=1)
