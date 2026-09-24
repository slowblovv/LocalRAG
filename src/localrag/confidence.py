from __future__ import annotations
from .retrieval.tokenizer import tokenize

def retrieval_confidence(query,reranked,lexical,vector):
    if not reranked: return {'score':0.0,'top_score':0.0,'score_gap':0.0,'support_count':0,'query_term_coverage':0.0,'lexical_vector_agreement':0.0}
    top=reranked[0].score; second=reranked[1].score if len(reranked)>1 else 0.0; gap=max(top-second,0.0)
    q=set(tokenize(query)); doc=set(tokenize(reranked[0].text)); coverage=len(q&doc)/max(len(q),1)
    lex_ids={x.chunk_id for x in lexical}; vec_ids={x.chunk_id for x in vector}; agreement=1.0 if reranked[0].chunk_id in lex_ids and reranked[0].chunk_id in vec_ids else 0.0
    support=sum(1 for x in reranked if x.score >= max(top*0.8,0.01))
    normalized_score=min(top/0.5,1.0)
    score=0.30*normalized_score+0.45*coverage+0.15*agreement+0.05*min(support/3,1.0)+0.05*min(gap/0.5,1.0)
    return {'score':score,'top_score':top,'score_gap':gap,'support_count':support,'query_term_coverage':coverage,'lexical_vector_agreement':agreement}
