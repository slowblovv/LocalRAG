from pathlib import Path
from localrag.index import LocalIndex
from localrag.retrieval.embed import HashEmbedder

def test_incremental_and_deleted_document(tmp_path):
    root=tmp_path/'knowledge'; root.mkdir(); (root/'a.md').write_text('# A\n\nhello world')
    idx=LocalIndex(tmp_path/'indexes', embedder=HashEmbedder(32)); first=idx.build(root); idx.load_current(); ids1=set(idx.chunks)
    second=idx.build(root); idx.load_current(); assert first['index_version']==second['index_version']; assert ids1==set(idx.chunks)
    (root/'a.md').write_text('# A\n\nchanged world'); idx.build(root); idx.load_current(); assert set(idx.chunks)!=ids1
    (root/'a.md').unlink(); idx.build(root); idx.load_current(); assert not idx.chunks


def test_identical_documents_are_explicitly_deduplicated(tmp_path):
    root = tmp_path / "knowledge"; root.mkdir()
    (root / "a.md").write_text("same bytes", encoding="utf8")
    (root / "copy.txt").write_text("same bytes", encoding="utf8")
    idx = LocalIndex(tmp_path / "indexes", embedder=HashEmbedder(32))
    result = idx.build(root); idx.load_current()
    assert result["documents"] == 1
    assert len(idx.documents) == 1

