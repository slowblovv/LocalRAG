import re
STOPWORDS={'a','an','and','are','as','at','be','by','for','from','how','in','is','it','of','on','or','that','the','to','what','when','where','which','who','why','with','does','do','can','this','than','more','about','into','their','they'}

def normalize_query(text:str)->str:
    text=text.casefold(); text=re.sub(r"[^\w\s-]"," ",text,flags=re.UNICODE); return re.sub(r"\s+"," ",text).strip()
def tokenize(text:str)->list[str]: return [t for t in normalize_query(text).split() if t not in STOPWORDS]
