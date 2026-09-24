import json
from pathlib import Path
import numpy as np
from localrag.ingest.chunker import ingest_file, chunk_records
from localrag.retrieval.hybrid import reciprocal_rank_fusion
from localrag.retrieval.context import build_context
from localrag.retrieval.embed import HashEmbedder
from localrag.retrieval.vector import VectorIndex
from localrag.models import Chunk, SearchResult
from localrag.index import LocalIndex
from localrag.retrieval.rerank import OverlapReranker
from localrag.engine import RagEngine

def test_chunk_ids_are_deterministic(tmp_path):
    p=tmp_path/'a.md'; p.write_text('# T\n\nHello world.\n\nMore text.',encoding='utf8'); a=ingest_file(p)[2]; b=ingest_file(p)[2]; assert [x.id for x in a]==[x.id for x in b]

def test_rrf_fuses_rankings():
    a=SearchResult('a',1,'lexical','a',{}); b=SearchResult('b',.5,'lexical','b',{}); c=SearchResult('c',1,'vector','c',{})
    out=reciprocal_rank_fusion([[a,b],[c,a]],60,3); assert out[0].chunk_id=='a'

def test_context_budget_and_redundancy():
    r1=SearchResult('1',1,'x','alpha beta gamma',{'section':'A'}); r2=SearchResult('2',.9,'x','alpha beta gamma',{'section':'A'}); r3=SearchResult('3',.8,'x','delta epsilon',{'section':'B'})
    out,t=build_context([r1,r2,r3],max_tokens=3,max_chunks=5,redundancy_threshold=.8); assert [x.chunk_id for x in out]==['1']

def test_vector_incompatible_metadata(tmp_path):
    e=HashEmbedder(8); idx=VectorIndex(e); c=Chunk.make('s',0,'hello','s'); idx.build([c],'v1'); idx.save(tmp_path)
    with __import__('pytest').raises(ValueError): VectorIndex.load(tmp_path,HashEmbedder(16))

def test_context_source_diversity_is_by_source_id():
    from localrag.retrieval.context import build_context
    rows=[SearchResult(str(i),1-i*.1,'reranked',f'content {i}',{'source_id':'doc-a','section':str(i)}) for i in range(4)]
    rows += [SearchResult('x',.5,'reranked','other content',{'source_id':'doc-b','section':'1'})]
    out,_=build_context(rows,max_tokens=100,max_chunks=5,max_per_source=2)
    assert [x.metadata['source_id'] for x in out].count('doc-a')==2
    assert 'doc-b' in [x.metadata['source_id'] for x in out]

def test_cache_hit_gets_new_request_id(tmp_path, monkeypatch):
    from localrag.engine import RagEngine
    from localrag.cache import DiskCache
    from localrag.providers.llm import FakeLLMProvider
    idx=LocalIndex(tmp_path/'indexes', embedder=HashEmbedder(32)); chunk=Chunk.make('sha',0,'TCP provides reliable ordered delivery.','sha'); idx.chunks={chunk.id:chunk}; idx.documents={'sha':{'id':'sha','filename':'a.md'}}; idx.lexical.build([chunk]); idx.vector.build([chunk],'v1'); idx.index_version='v1'; idx.root.mkdir(parents=True,exist_ok=True); idx.query_cache = None
    # Persist a real vector index so the same engine can be rebuilt without a global model.
    idx.query_cache = DiskCache(idx.root/'cache'/'queries',300)
    e=RagEngine(idx,reranker=OverlapReranker(),llm=FakeLLMProvider()); a=e.query('What does TCP provide?','lexical',1,True); b=e.query('What does TCP provide?','lexical',1,True)
    assert a['request_id']!=b['request_id'] and b['cache_hit'] is True


def test_query_updates_real_metrics(tmp_path):
    from localrag.cache import DiskCache
    from localrag.metrics import cache_misses, generation_latency, query_requests_total, retrieval_latency, search_requests_total
    from localrag.providers.llm import FakeLLMProvider
    idx = LocalIndex(tmp_path / 'indexes', embedder=HashEmbedder(32)); chunk = Chunk.make('m',0,'TCP reliable ordered delivery.', 'm')
    idx.chunks={chunk.id:chunk}; idx.documents={'m':{'id':'m','filename':'m.md'}}; idx.lexical.build([chunk]); idx.vector.build([chunk],'v1'); idx.index_version='v1'; idx.query_cache=DiskCache(idx.root/'cache'/'queries',300)
    engine=RagEngine(idx,reranker=OverlapReranker(),llm=FakeLLMProvider())
    q0=query_requests_total._value.get(); s0=search_requests_total.labels(mode='hybrid')._value.get(); r0=retrieval_latency._sum.get(); g0=generation_latency._sum.get(); c0=cache_misses._value.get()
    out=engine.query('What does TCP provide?','hybrid',1,True)
    assert out['latency_ms']['retrieval'] >= 0 and out['latency_ms']['context'] >= 0
    assert query_requests_total._value.get()==q0+1 and search_requests_total.labels(mode='hybrid')._value.get()==s0+1
    assert retrieval_latency._sum.get() >= r0 and generation_latency._sum.get() >= g0 and cache_misses._value.get()==c0+1


def test_cache_identity_includes_refusal_policy(tmp_path, monkeypatch):
    from localrag.cache import DiskCache
    from localrag.config import settings
    from localrag.providers.llm import FakeLLMProvider
    idx=LocalIndex(tmp_path/'indexes', embedder=HashEmbedder(32)); chunk=Chunk.make('m',0,'TCP reliable ordered delivery.', 'm')
    idx.chunks={chunk.id:chunk}; idx.documents={'m':{'id':'m','filename':'m.md'}}; idx.lexical.build([chunk]); idx.vector.build([chunk],'v1'); idx.index_version='v1'; idx.query_cache=DiskCache(idx.root/'cache'/'queries',300)
    engine=RagEngine(idx,reranker=OverlapReranker(),llm=FakeLLMProvider())
    monkeypatch.setattr(settings,'refuse_top_score',0.5)
    first=engine.query('What is the salary of the networking engineer?','hybrid',1,True)
    monkeypatch.setattr(settings,'refuse_top_score',0.65)
    second=engine.query('What is the salary of the networking engineer?','hybrid',1,True)
    assert first['cache_hit'] is False and second['cache_hit'] is False
    assert second['trace']['citation_validation_result']=='refused'


def test_real_model_adapters_are_wired_to_sentence_transformers(monkeypatch):
    import sys, types
    import numpy as np

    class FakeST:
        def __init__(self, name, local_files_only=False):
            self.name = name; self.local_files_only = local_files_only
        def get_sentence_embedding_dimension(self): return 4
        def encode(self, values, **kwargs): return np.ones((len(values), 4), dtype=np.float32)

    class FakeCE:
        def __init__(self, name, **kwargs):
            self.name = name; self.kwargs = kwargs
        def predict(self, pairs, **kwargs): return np.full(len(pairs), 0.5, dtype=np.float32)

    fake_st = types.ModuleType('sentence_transformers')
    fake_st.SentenceTransformer = FakeST; fake_st.CrossEncoder = FakeCE
    monkeypatch.setitem(sys.modules, 'sentence_transformers', fake_st)
    from localrag.retrieval.embed import SentenceTransformerEmbedder
    from localrag.retrieval.rerank import CrossEncoderReranker
    e = SentenceTransformerEmbedder('sentence-transformers/test', local_files_only=True)
    r = CrossEncoderReranker('cross-encoder/test', local_files_only=True)
    assert e.name == 'sentence-transformers/test' and e.dimension == 4
    assert r.name == 'cross-encoder/test' and 'activation_fn' in r.model.kwargs

