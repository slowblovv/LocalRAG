'use client';

import { useState } from 'react';

const API = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export default function IndexPage() {
  const [path, setPath] = useState('./knowledge');
  const [job, setJob] = useState<any>(null);
  const [loading, setLoading] = useState(false);

  async function start() {
    setLoading(true);
    try {
      const res = await fetch(`${API}/api/v1/index`, {
        method: 'POST',
        headers: {'content-type': 'application/json'},
        body: JSON.stringify({path}),
      });
      const created = await res.json();
      setJob(created);
      if (created.job_id) poll(created.job_id);
    } finally {
      setLoading(false);
    }
  }

  async function poll(id: string) {
    const res = await fetch(`${API}/api/v1/jobs/${id}`);
    const current = await res.json();
    setJob(current);
    if (current.status === 'PENDING' || current.status === 'RUNNING') {
      window.setTimeout(() => poll(id), 1000);
    }
  }

  return <main className="wrap">
    <h1>Index documents</h1>
    <div className="card">
      <p className="muted">Indexing is asynchronous: API → TinyQueue/local durable job → worker → atomic index swap.</p>
      <label>Document root</label>
      <input value={path} onChange={e => setPath(e.target.value)} />
      <button onClick={start} disabled={loading}>{loading ? 'Scheduling…' : 'Start indexing'}</button>
      {job && <><h3>Job</h3><pre>{JSON.stringify(job, null, 2)}</pre></>}
    </div>
  </main>;
}
