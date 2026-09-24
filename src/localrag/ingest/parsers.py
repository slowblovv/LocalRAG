from pathlib import Path
from pypdf import PdfReader

def parse_document(path: Path) -> list[dict]:
    suffix=path.suffix.lower()
    if suffix in {'.md','.txt'}:
        text=path.read_text(encoding='utf-8',errors='replace')
        return [{'text':text,'page_number':None,'section':_title(text)}]
    if suffix=='.pdf':
        reader=PdfReader(str(path)); pages=[]
        for i,page in enumerate(reader.pages,1):
            pages.append({'text':page.extract_text() or '', 'page_number':i, 'section':None})
        return pages
    raise ValueError(f'unsupported document type: {suffix}')

def _title(text:str)->str|None:
    for line in text.splitlines():
        if line.strip().startswith('#'):
            return line.lstrip('#').strip() or None
    return None
