from localrag.citations import validate_citations, citation_source_coverage

def test_valid_citations():
    ids,ok=validate_citations('Answer [1] and [2].',2); assert ids==[1,2] and ok

def test_invalid_citation():
    _,ok=validate_citations('Answer [999].',2); assert not ok

def test_coverage():
    assert citation_source_coverage([1,1,2],4)==.5
