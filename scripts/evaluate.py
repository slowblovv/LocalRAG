import json
import statistics
import time
from pathlib import Path

from localrag.config import settings
from localrag.index import LocalIndex
from localrag.evaluation.metrics import aggregate, hit_rate_at_k, mrr, ndcg_at_k, precision_at_k, recall_at_k
from localrag.retrieval.hybrid import reciprocal_rank_fusion
from localrag.retrieval.rerank import create_reranker, rerank


def p95(values):
    return statistics.quantiles(values, n=20, method='inclusive')[-1] if len(values) >= 2 else (values[0] if values else 0.0)


def main():
    idx = LocalIndex(); info = idx.build('demo_corpus'); idx.load_current()
    qs = json.loads(Path('evaluation/queries.json').read_text()); rel = json.loads(Path('evaluation/relevance.json').read_text())
    modes = {}
    reranker = create_reranker(settings.reranker_model)
    for mode in ['lexical','vector','hybrid']:
        rows=[]; times=[]
        for q,r in zip(qs,rel):
            t=time.perf_counter()
            lex=idx.lexical.search(q['query'],settings.top_k_lexical) if mode in {'lexical','hybrid'} else []
            vec=idx.vector.search(q['query'],settings.top_k_vector,idx.chunks) if mode in {'vector','hybrid'} else []
            res=lex if mode=='lexical' else vec if mode=='vector' else reciprocal_rank_fusion([lex,vec],settings.rrf_k,20)
            times.append((time.perf_counter()-t)*1000); ids=[x.chunk_id for x in res]; relevant=r['relevant_chunk_ids']
            rows.append({'Recall@10':recall_at_k(ids,relevant,10),'Precision@10':precision_at_k(ids,relevant,10),'MRR':mrr(ids,relevant),'NDCG@10':ndcg_at_k(ids,relevant,10),'HitRate@10':hit_rate_at_k(ids,relevant,10)})
        modes[mode] = {**aggregate(rows),'p95_retrieval_latency_ms':p95(times)}
    rows=[]; times=[]
    for q,r in zip(qs,rel):
        t=time.perf_counter(); lex=idx.lexical.search(q['query'],20); vec=idx.vector.search(q['query'],20,idx.chunks); base=reciprocal_rank_fusion([lex,vec],settings.rrf_k,40); res=rerank(q['query'],base,reranker,10); times.append((time.perf_counter()-t)*1000); ids=[x.chunk_id for x in res]; relevant=r['relevant_chunk_ids']; rows.append({'Recall@10':recall_at_k(ids,relevant,10),'Precision@10':precision_at_k(ids,relevant,10),'MRR':mrr(ids,relevant),'NDCG@10':ndcg_at_k(ids,relevant,10),'HitRate@10':hit_rate_at_k(ids,relevant,10)})
    modes['hybrid+reranker']={**aggregate(rows),'p95_reranker_plus_retrieval_latency_ms':p95(times),'reranker_model':reranker.name}
    print(json.dumps({'index':info,'embedding_model':idx.embedder.name,'reranker_model':reranker.name,'results':modes},indent=2))
    print('These values are measured on the repository fixture; none are hard-coded.')
if __name__=='__main__': main()
