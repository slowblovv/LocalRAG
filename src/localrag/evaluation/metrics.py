from __future__ import annotations
import math

def recall_at_k(results,relevant,k): return len({r for r in results[:k]} & set(relevant))/max(len(set(relevant)),1)
def precision_at_k(results,relevant,k): return len({r for r in results[:k]} & set(relevant))/max(k,1)
def hit_rate_at_k(results,relevant,k): return 1.0 if ({r for r in results[:k]} & set(relevant)) else 0.0
def mrr(results,relevant):
    rel=set(relevant)
    for i,r in enumerate(results,1):
        if r in rel: return 1/i
    return 0.0

def ndcg_at_k(results,relevant,k):
    rel=set(relevant); dcg=0.0
    for i,r in enumerate(results[:k],1):
        if r in rel: dcg+=1/math.log2(i+1)
    ideal=sum(1/math.log2(i+1) for i in range(1,min(len(rel),k)+1))
    return dcg/ideal if ideal else 0.0

def aggregate(rows):
    if not rows: return {}
    return {k:sum(x[k] for x in rows)/len(rows) for k in rows[0]}
