from __future__ import annotations
from dataclasses import dataclass
import os, re
import httpx

@dataclass
class LLMResponse:
    text:str
    model:str

class LLMProvider:
    name='base'
    def generate(self,system_prompt,user_prompt,max_tokens,temperature)->LLMResponse: raise NotImplementedError

class FakeLLMProvider(LLMProvider):
    name='fake'
    def generate(self,system_prompt,user_prompt,max_tokens,temperature):
        # Deterministic extractive behavior: select the source sentence with greatest query-term overlap.
        qmatch=re.search(r'<QUESTION>\n(.+?)\n</QUESTION>', user_prompt, flags=re.S)
        question=qmatch.group(1) if qmatch else ''
        qterms={t.casefold() for t in re.findall(r'\b\w+\b',question) if t.casefold() not in {'what','does','the','a','an','is','in','of','to','why','how','can','more','than','and','for'}}
        sources=re.findall(r'\[SOURCE (\d+)\] (.+?)\n(.+?)(?=\n\n\[SOURCE |\Z)',user_prompt,flags=re.S)
        best=None
        for sid,meta,text in sources:
            for sent in re.split(r'(?<=[.!?])\s+',text.strip()):
                score=len(qterms & {t.casefold() for t in re.findall(r'\b\w+\b',sent)})
                if best is None or score>best[0]: best=(score,sid,sent.strip())
        if not best or best[0]==0: return LLMResponse("I don't have enough evidence in the indexed documents to answer this question.",'fake-v1')
        return LLMResponse(f"{best[2]} [{best[1]}]",'fake-v1')

class OllamaProvider(LLMProvider):
    name='ollama'
    def __init__(self,base_url='http://localhost:11434',model='llama3.1:8b'): self.base_url=base_url.rstrip('/'); self.model=model
    def generate(self,system_prompt,user_prompt,max_tokens,temperature):
        payload={'model':self.model,'stream':False,'options':{'temperature':temperature,'num_predict':max_tokens},'system':system_prompt,'prompt':user_prompt}
        r=httpx.post(self.base_url+'/api/generate',json=payload,timeout=120); r.raise_for_status(); data=r.json(); return LLMResponse(data.get('response',''),self.model)

def create_provider(name,model): return FakeLLMProvider() if name=='fake' else OllamaProvider(os.getenv('OLLAMA_BASE_URL','http://localhost:11434'),model)
