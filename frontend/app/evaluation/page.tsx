'use client';
import {useEffect,useState} from 'react';
export default function Evaluation(){const [text,setText]=useState('Run scripts/evaluate.py to populate measured fixture results.'); useEffect(()=>{},[]); return <main className="wrap"><h1>Evaluation</h1><div className="card"><p>{text}</p><p className="muted">BM25, vector, hybrid and reranker results are produced by executable evaluation scripts; no benchmark numbers are claimed here unless measured in the current environment.</p></div></main>}
