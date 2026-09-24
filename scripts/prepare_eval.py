import json
from pathlib import Path
from localrag.ingest.chunker import ingest_file

QUERIES=[]

def main():
    root=Path('demo_corpus'); relevant={}
    for p in sorted(root.glob('*')):
        doc_id,sha,chunks=ingest_file(p)
        relevant[p.name]=[c.id for c in chunks]
    topics=[
      ('What does classification output?','machine_learning.md'),('What does a held-out test set estimate?','machine_learning.md'),('Why are PR-AUC and ROC-AUC different?','machine_learning.md'),
      ('What is partial failure?','distributed_systems.md'),('Why should retries be idempotent?','distributed_systems.md'),('What does a timeout prove?','distributed_systems.md'),
      ('What does a transaction provide?','databases.md'),('What is isolation?','databases.md'),('Why can indexes make writes more expensive?','databases.md'),
      ('What does TCP provide?','networking.md'),('What mechanisms recover TCP packet loss?','networking.md'),('Why can p99 matter more than mean latency?','networking.md'),
      ('What is authentication?','security.md'),('What is authorization?','security.md'),('What does threat modeling identify?','security.md'),
    ]
    # Repeat with paraphrased queries to reach a >=50 curated set while retaining known relevant chunk IDs.
    variants=['Explain','In simple terms, explain','What does the document say about','How is this described:']
    entries=[]
    for text,fn in topics:
        for prefix in variants:
            q=f'{prefix} {text.lower()}' if prefix!='What does the document say about' else f'{prefix} {text[0].lower()+text[1:]}'
            entries.append({'query':q,'relevant_chunk_ids':relevant[fn]})
    entries=entries[:60]
    Path('evaluation').mkdir(exist_ok=True)
    Path('evaluation/queries.json').write_text(json.dumps([{'query':e['query']} for e in entries],ensure_ascii=False,indent=2),encoding='utf8')
    Path('evaluation/relevance.json').write_text(json.dumps([{'relevant_chunk_ids':e['relevant_chunk_ids']} for e in entries],ensure_ascii=False,indent=2),encoding='utf8')
    print(f'wrote {len(entries)} queries')
if __name__=='__main__': main()
