from __future__ import annotations
import re
from .parsers import parse_document
from ..models import Chunk

def _tokens(text:str)->list[str]: return re.findall(r"\S+", text)

def chunk_records(records:list[dict], document_sha256:str, source_id:str, target_tokens:int=700, overlap_tokens:int=100)->list[Chunk]:
    out=[]; idx=0
    for rec in records:
        text=rec.get('text','').strip()
        if not text: continue
        paragraphs=[p.strip() for p in re.split(r'\n\s*\n',text) if p.strip()]
        current=[]; n=0
        for para in paragraphs:
            pn=len(_tokens(para))
            if current and n+pn>target_tokens:
                joined='\n\n'.join(current).strip(); out.append(Chunk.make(document_sha256,idx,joined,source_id,rec.get('page_number'),rec.get('section'))); idx+=1
                carry=_tokens(joined)[-overlap_tokens:] if overlap_tokens else []
                current=[' '.join(carry)] if carry else []; n=len(carry)
            current.append(para); n+=pn
        if current:
            joined='\n\n'.join(current).strip()
            if joined: out.append(Chunk.make(document_sha256,idx,joined,source_id,rec.get('page_number'),rec.get('section'))); idx+=1
    return out

def ingest_file(path, target_tokens=700, overlap_tokens=100):
    from pathlib import Path
    p=Path(path); import hashlib, os
    raw=p.read_bytes(); sha=hashlib.sha256(raw).hexdigest()
    records=parse_document(p)
    doc_id=sha
    chunks=chunk_records(records,sha,doc_id,target_tokens,overlap_tokens)
    return doc_id, sha, chunks
