from __future__ import annotations
import math, json
from collections import Counter, defaultdict
from .tokenizer import tokenize
from ..models import Chunk, SearchResult

class BM25Index:
    def __init__(self,k1=1.5,b=0.75): self.k1=k1; self.b=b; self.docs=[]; self.tf=[]; self.df=defaultdict(int); self.avgdl=0.0; self.inverted={}
    def build(self,chunks:list[Chunk]):
        self.docs=[c.id for c in chunks]; self.tf=[]; self.df=defaultdict(int); self.inverted=defaultdict(list)
        for i,c in enumerate(chunks):
            cnt=Counter(tokenize(c.text)); self.tf.append(cnt)
            for t in cnt: self.df[t]+=1; self.inverted[t].append(i)
        self.avgdl=sum(sum(x.values()) for x in self.tf)/max(len(self.tf),1)
    def search(self,q:str,top_k=20)->list[tuple[str,float]]:
        qtokens=tokenize(q); scores=[0.0]*len(self.docs); n=len(self.docs)
        for term in qtokens:
            df=self.df.get(term,0)
            if not df: continue
            idf=math.log(1+(n-df+0.5)/(df+0.5))
            for i in self.inverted.get(term,[]):
                f=self.tf[i][term]; dl=sum(self.tf[i].values())
                scores[i]+=idf*(f*(self.k1+1))/(f+self.k1*(1-self.b+self.b*dl/max(self.avgdl,1e-9)))
        order=sorted(range(n),key=lambda i:scores[i],reverse=True)
        return [(self.docs[i],scores[i]) for i in order[:top_k] if scores[i]>0]
    def dump(self,path):
        data={'k1':self.k1,'b':self.b,'docs':self.docs,'tf':[dict(x) for x in self.tf],'df':dict(self.df),'avgdl':self.avgdl}
        path.write_text(json.dumps(data),encoding='utf8')
    @classmethod
    def load(cls,path):
        d=json.loads(path.read_text()); x=cls(d['k1'],d['b']); x.docs=d['docs']; x.tf=[Counter(t) for t in d['tf']]; x.df=defaultdict(int,d['df']); x.avgdl=d['avgdl']; x.inverted=defaultdict(list); [x.inverted[t].append(i) for i,tf in enumerate(x.tf) for t in tf]; return x
