from pathlib import Path

from localrag.jobs import JobStore
from localrag.metrics import cache_hits


def test_job_store_lifecycle(tmp_path):
    store = JobStore(tmp_path / 'jobs.db'); job_id = store.enqueue('index_documents', str(tmp_path / 'knowledge'))
    job = store.get(job_id); assert job and job.status == 'PENDING'
    claimed = store.claim(); assert claimed and claimed[0].id == job_id and claimed[0].status == 'RUNNING'
    store.finish(job_id, True); assert store.get(job_id).status == 'SUCCEEDED'


def test_cache_counter_has_prometheus_metric_family():
    before = cache_hits._value.get()
    cache_hits.inc()
    assert cache_hits._value.get() == before + 1


def test_stale_running_job_is_recovered(tmp_path):
    from datetime import datetime, timedelta, timezone
    store = JobStore(tmp_path / 'jobs.db')
    job_id = store.enqueue('index_documents', str(tmp_path / 'knowledge'))
    claimed = store.claim(); assert claimed
    stale = (datetime.now(timezone.utc) - timedelta(seconds=3600)).isoformat()
    with store._connect() as con:
        con.execute("UPDATE jobs SET started_at=? WHERE id=?", (stale, job_id))
    recovered = store.claim(); assert recovered and recovered[0].id == job_id and recovered[0].status == 'RUNNING'
    assert recovered[0].attempts == 2


def test_async_index_api_and_worker_lifecycle(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from localrag.api import app
    from localrag.config import settings
    from localrag.worker import process_once

    knowledge = tmp_path / 'knowledge'; knowledge.mkdir(); (knowledge / 'a.md').write_text('TCP is reliable and ordered.')
    job_db = tmp_path / 'jobs.db'; index_root = tmp_path / 'indexes'
    monkeypatch.setattr(settings, 'data_root', str(knowledge))
    monkeypatch.setattr(settings, 'index_root', str(index_root))
    monkeypatch.setattr(settings, 'job_db_path', str(job_db))

    client = TestClient(app)
    accepted = client.post('/api/v1/index', json={'path': './knowledge'})
    assert accepted.status_code == 200 and accepted.json()['status'] == 'accepted'
    job_id = accepted.json()['job_id']
    before = client.get(f'/api/v1/jobs/{job_id}').json()
    assert before['status'] == 'PENDING'

    class FakeIndex:
        def __init__(self): pass
        def build(self, path):
            assert Path(path).resolve() == knowledge.resolve()
            return {'index_version': 'test-index', 'documents': 1, 'chunks': 1}

    monkeypatch.setattr('localrag.worker.LocalIndex', FakeIndex)
    assert process_once() is True
    after = client.get(f'/api/v1/jobs/{job_id}').json()
    assert after['status'] == 'SUCCEEDED' and after['attempts'] == 1


def test_external_tinyqueue_job_is_visible(tmp_path, monkeypatch):
    from localrag.jobs import TinyQueueAdapter
    from localrag.config import settings

    module = tmp_path / 'fakequeue.py'
    module.write_text('def enqueue(job_type, payload):\n    return "external-123"\n')
    import sys
    sys.path.insert(0, str(tmp_path))
    try:
        monkeypatch.setattr(settings, 'job_db_path', str(tmp_path / 'jobs.db'))
        adapter = TinyQueueAdapter(external_target='fakequeue:enqueue')
        job_id = adapter.submit_index(tmp_path / 'knowledge')
        assert job_id == 'external-123'
        job = adapter.get(job_id)
        assert job and job.status == 'EXTERNAL'
    finally:
        sys.path.remove(str(tmp_path))
