import './globals.css';
import Link from 'next/link';
export default function Layout({children}:{children:React.ReactNode}){return <><nav className="nav"><Link href="/">LocalRAG</Link><Link href="/query">Query</Link><Link href="/index">Index</Link><Link href="/docs">Docs</Link><Link href="/evaluation">Evaluation</Link><Link href="/settings">Settings</Link></nav>{children}</>}
