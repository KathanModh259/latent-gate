import React, { useState, useCallback } from 'react';
import { Terminal, Zap, ChevronRight, Activity, ArrowRight, Code, Loader2, Check, Copy, AlertTriangle, Cloud } from 'lucide-react';
import { loadOptimizer, optimize, PACKAGE_VERSION } from './utils/optimizer';

const MAX_INPUT_CHARS = parseInt(import.meta.env.VITE_MAX_INPUT_CHARS || '200000', 10);
const HOSTED_URL = 'https://mcprush.com/kathanmodh259/latentgate-hosted-mcp';
const GITHUB_URL = 'https://github.com/KathanModh259/latent-gate';
const CLAUDE_CODE_CMD = 'claude mcp add --scope user latent-gate -- uvx --from "latent-gate[mcp,tokens]" latent-gate-mcp';

const LEVELS = {
  lossless: 'Only whitespace, JSON formatting and exact duplicates change. Every line survives.',
  balanced: 'Also folds runs of near-identical lines (first, last and value ranges kept) and drops filler. The default.',
  aggressive: 'Also keeps only the most relevant sentences, about half. Use when you just need the gist.',
};

// Deterministic sample inputs, so visitors can see a result without pasting anything
const SAMPLES = [
  {
    label: '400-line error log',
    make: () => {
      let seed = 7;
      const rand = () => (seed = (seed * 1103515245 + 12345) % 2147483648) / 2147483648;
      const lines = ['2026-10-08 09:00:00 INFO  app started, 3 workers'];
      for (let i = 0; i < 400; i++) {
        const t = `${String(Math.floor(i / 60)).padStart(2, '0')}:${String(i % 60).padStart(2, '0')}`;
        lines.push(`2026-10-08 09:${t} ERROR order ${10000 + i} failed: payment gateway timeout after ${2900 + Math.floor(rand() * 200)}ms (retry ${(i % 3) + 1}/3)`);
      }
      lines.push('2026-10-08 09:07:00 FATAL circuit breaker open for payments-api');
      return lines.join('\n');
    },
  },
  {
    label: 'Pretty-printed API JSON',
    make: () => JSON.stringify({
      status: 'ok',
      page: 1,
      orders: Array.from({ length: 12 }, (_, i) => ({
        id: 5000 + i,
        customer: { id: 900 + i, tier: i % 3 ? 'standard' : 'gold' },
        items: [{ sku: `SKU-${100 + i}`, qty: 1 + (i % 4), price: 19.99 + i }],
        status: i % 5 ? 'shipped' : 'pending',
        created_at: `2026-10-0${1 + (i % 8)}T10:${String(i).padStart(2, '0')}:00Z`,
      })),
    }, null, 2),
  },
  {
    label: 'Chatty prompt, pasted twice',
    make: () => {
      const body = 'Hi! Hope you are doing well.\n\nI was wondering if you could possibly help me with something. The checkout page should validate the card number before submitting, show an inline error under the field, and keep the Pay button disabled until the form is valid. It also needs to work on mobile.\n\nThanks so much in advance!';
      return `${body}\n\n${body}`;
    },
  },
];

function CopyLine({ text }) {
  const [copied, setCopied] = useState(false);
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      setCopied(false);
    }
  };
  return (
    <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'stretch' }}>
      {/* minWidth 0 lets the flex item shrink, so a long command scrolls instead of widening the page */}
      <code className="pixel-input" style={{ flex: 1, minWidth: 0, fontSize: '0.75rem', overflowX: 'auto', whiteSpace: 'nowrap', padding: '0.6rem' }}>{text}</code>
      <button className="pixel-btn pixel-btn-secondary" onClick={copy} style={{ padding: '0 0.8rem', fontSize: '11px' }} aria-label={copied ? 'Copied' : 'Copy command'}>
        {copied ? <Check size={14} /> : <Copy size={14} />}
      </button>
    </div>
  );
}

function App() {
  const [inputText, setInputText] = useState('');
  const [level, setLevel] = useState('balanced');
  const [result, setResult] = useState(null);
  const [status, setStatus] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const [copied, setCopied] = useState(false);

  // Start downloading the runtime as soon as someone shows interest, so "Optimize" feels instant
  const warmUp = useCallback(() => {
    loadOptimizer(setStatus).then(() => setStatus('ready'), () => setStatus(''));
  }, []);

  const run = async (text = inputText, lvl = level) => {
    if (!text.trim()) return;
    setBusy(true);
    setError(null);
    try {
      await loadOptimizer(setStatus);
      setStatus('ready');
      setResult(await optimize(text, lvl));
    } catch (err) {
      setError(`${err.message}. Check your connection and try again.`);
      setStatus('');
    } finally {
      setBusy(false);
    }
  };

  const pickSample = (sample) => {
    const text = sample.make();
    setInputText(text);
    run(text);
  };

  const pickLevel = (lvl) => {
    setLevel(lvl);
    if (result) run(inputText, lvl);
  };

  const copyOutput = async () => {
    try {
      await navigator.clipboard.writeText(result.text);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      setCopied(false);
    }
  };

  const saved = result ? Math.round((1 - result.optimized / Math.max(result.original, 1)) * 100) : 0;
  const tooLong = inputText.length > MAX_INPUT_CHARS;

  return (
    <div className="app-wrapper">
      <nav style={{ padding: '1.5rem', display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '1rem' }}>
        <div className="pixel-font" style={{ fontSize: '1.2rem', color: 'var(--primary-neon)' }}>
          LatentGate<span className="cursor-blink">_</span>
        </div>
        <div style={{ display: 'flex', gap: '1rem', flexWrap: 'wrap' }}>
          <a href="#try" className="pixel-btn pixel-btn-secondary" style={{ padding: '0.5rem 1rem', fontSize: '12px' }}>Try it</a>
          <a href="#install" className="pixel-btn pixel-btn-secondary" style={{ padding: '0.5rem 1rem', fontSize: '12px' }}>Install</a>
          <a href="#pricing" className="pixel-btn pixel-btn-secondary" style={{ padding: '0.5rem 1rem', fontSize: '12px' }}>Pricing</a>
        </div>
      </nav>

      <header className="container" style={{ textAlign: 'center', padding: '4rem 0 5rem 0' }}>
        <h1 className="pixel-font text-gradient" style={{ fontSize: '2.2rem', marginBottom: '1.5rem', lineHeight: '1.5' }}>
          GIVE YOUR AI<br />THE SIGNAL,<br />NOT THE NOISE.
        </h1>
        <p style={{ fontSize: '1.15rem', color: '#64748b', maxWidth: '640px', margin: '0 auto 2.5rem auto' }}>
          An MCP server that shrinks logs, JSON, docs and web pages before Claude reads them.
          Offline, deterministic, a few milliseconds, no API key. A 400-line error log goes from 13,641 tokens to 145.
        </p>
        <div style={{ display: 'flex', gap: '1rem', justifyContent: 'center', flexWrap: 'wrap' }}>
          <a href="#try" className="pixel-btn" onMouseEnter={warmUp} onFocus={warmUp}><Zap size={18} /> Try it in your browser</a>
          <a href={GITHUB_URL} target="_blank" rel="noreferrer" className="pixel-btn pixel-btn-secondary"><Code size={18} /> GitHub</a>
        </div>
      </header>

      <section id="try" className="container" style={{ padding: '3rem 0' }} onMouseEnter={warmUp}>
        <div className="text-center mb-8">
          <h2 className="pixel-font" style={{ fontSize: '1.6rem', color: 'var(--secondary-neon)' }}>&gt; TRY_IT</h2>
          <p style={{ color: '#64748b', marginTop: '1rem' }}>
            The real optimizer (latent-gate {PACKAGE_VERSION}) runs in your browser. Nothing you paste leaves your machine.
          </p>
        </div>

        <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap', justifyContent: 'center', marginBottom: '1.5rem' }}>
          <span style={{ color: '#64748b', alignSelf: 'center', fontSize: '0.9rem' }}>No text handy? Try:</span>
          {SAMPLES.map((s) => (
            <button key={s.label} className="pixel-btn pixel-btn-secondary" style={{ padding: '0.5rem 0.9rem', fontSize: '11px' }} onClick={() => pickSample(s)} disabled={busy}>
              {s.label}
            </button>
          ))}
        </div>

        <div className="grid grid-cols-2 gap-8" style={{ alignItems: 'stretch' }}>
          <div className="pixel-border" style={{ display: 'flex', flexDirection: 'column' }}>
            <h3 className="pixel-font mb-4" style={{ fontSize: '0.9rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <Terminal size={16} color="var(--primary-neon)" /> INPUT
            </h3>
            <textarea
              className="pixel-input"
              style={{ flex: 1, minHeight: '260px', fontFamily: 'monospace', fontSize: '0.8rem' }}
              placeholder="Paste a log, a JSON response, a long prompt or a doc…"
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              onFocus={warmUp}
              aria-label="Text to optimize"
            />
            <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '0.5rem', fontSize: '0.75rem', color: tooLong ? '#ef4444' : '#64748b' }}>
              <span>{inputText.length.toLocaleString()} / {MAX_INPUT_CHARS.toLocaleString()} chars</span>
              {tooLong && <span><AlertTriangle size={12} /> Too long for the browser demo</span>}
            </div>

            <div role="radiogroup" aria-label="Optimization level" style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem', marginTop: '1rem' }}>
              {Object.keys(LEVELS).map((l) => (
                <button key={l} role="radio" aria-checked={level === l} onClick={() => pickLevel(l)} disabled={busy}
                  className={level === l ? 'pixel-btn' : 'pixel-btn pixel-btn-secondary'}
                  style={{ flex: '1 1 auto', justifyContent: 'center', padding: '0.5rem', fontSize: '11px' }}>
                  {l}
                </button>
              ))}
            </div>
            <p style={{ fontSize: '0.8rem', color: '#64748b', marginTop: '0.5rem', minHeight: '2.5em' }}>{LEVELS[level]}</p>

            <button className="pixel-btn" style={{ justifyContent: 'center', marginTop: '0.5rem' }} onClick={() => run()} disabled={busy || !inputText.trim() || tooLong} aria-busy={busy}>
              {busy ? <><Loader2 size={16} className="animate-spin" /> WORKING…</> : <>OPTIMIZE <ArrowRight size={16} /></>}
            </button>
          </div>

          <div className="pixel-border-alt" style={{ display: 'flex', flexDirection: 'column' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
              <h3 className="pixel-font" style={{ fontSize: '0.9rem', display: 'flex', alignItems: 'center', gap: '0.5rem', margin: 0 }}>
                <Activity size={16} color="var(--secondary-neon)" /> WHAT CLAUDE READS
              </h3>
              {result && !busy && (
                <button className="pixel-btn pixel-btn-secondary" onClick={copyOutput} style={{ padding: '0.4rem 0.8rem', fontSize: '11px' }}>
                  {copied ? <><Check size={14} /> COPIED</> : <><Copy size={14} /> COPY</>}
                </button>
              )}
            </div>

            {error && (
              <div role="alert" style={{ padding: '0.75rem', border: '2px solid #ef4444', marginBottom: '1rem', color: '#ef4444', display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
                <AlertTriangle size={16} /> {error}
              </div>
            )}

            <div className="pixel-input" aria-live="polite"
              style={{ flex: 1, minHeight: '260px', backgroundColor: '#f8fafc', color: '#0f172a', overflowY: 'auto', fontFamily: 'monospace', fontSize: '0.8rem', whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}>
              {busy && status && status !== 'ready' ? (
                <div style={{ textAlign: 'center', marginTop: '4rem', color: '#64748b' }}>
                  <Loader2 size={28} className="animate-spin" /><div style={{ marginTop: '1rem' }}>{status}</div>
                </div>
              ) : result ? result.text : (
                <span style={{ color: '#94a3b8' }}>Pick a sample or paste your own text, then press Optimize.</span>
              )}
            </div>

            {result && (
              <div style={{ marginTop: '1rem', padding: '1rem', border: '2px dashed var(--primary-neon)' }}>
                <div className="grid grid-cols-3 gap-4 text-center">
                  <div><div style={{ fontSize: '0.75rem', color: '#64748b' }}>Before</div><div className="pixel-font" style={{ color: '#ef4444' }}>{result.original.toLocaleString()}</div></div>
                  <div><div style={{ fontSize: '0.75rem', color: '#64748b' }}>After</div><div className="pixel-font" style={{ color: 'var(--primary-neon)' }}>{result.optimized.toLocaleString()}</div></div>
                  <div><div style={{ fontSize: '0.75rem', color: '#64748b' }}>Saved</div><div className="pixel-font" style={{ color: 'var(--accent-neon)' }}>{saved}%</div></div>
                </div>
                <p style={{ fontSize: '0.75rem', color: '#64748b', marginTop: '0.75rem', textAlign: 'center' }}>
                  Tokens {result.counter === 'heuristic' ? 'estimated (about ±6%; the installed package counts exactly with tiktoken)' : `counted with ${result.counter}`} · {result.ms.toFixed(1)} ms in your browser
                </p>
              </div>
            )}
          </div>
        </div>
      </section>

      <section id="install" className="container" style={{ padding: '4rem 0' }}>
        <div className="text-center mb-8">
          <h2 className="pixel-font" style={{ fontSize: '1.6rem', color: 'var(--primary-neon)' }}>&gt; USE_IT</h2>
          <p style={{ color: '#64748b', marginTop: '1rem' }}>Then just ask Claude: “read app.log and tell me why orders fail.”</p>
        </div>
        <div className="grid grid-cols-2 gap-8">
          <div className="pixel-border">
            <h3 className="pixel-font mb-4" style={{ fontSize: '0.9rem' }}><Terminal size={16} /> Claude Code, Cursor, Zed… (free)</h3>
            <CopyLine text={CLAUDE_CODE_CMD} />
            <p style={{ fontSize: '0.8rem', color: '#64748b', marginTop: '0.75rem' }}>Runs on your machine. Needs <a href="https://docs.astral.sh/uv/" target="_blank" rel="noreferrer">uv</a>.</p>
          </div>
          <div className="pixel-border">
            <h3 className="pixel-font mb-4" style={{ fontSize: '0.9rem' }}><Cloud size={16} /> claude.ai web &amp; mobile (hosted)</h3>
            <p style={{ color: '#64748b', fontSize: '0.9rem', marginBottom: '1rem' }}>Nothing to install. Adds a URL-fetch tool that returns pages as clean text.</p>
            <a href={HOSTED_URL} target="_blank" rel="noreferrer" className="pixel-btn" style={{ justifyContent: 'center' }}>Get the hosted version <ArrowRight size={16} /></a>
          </div>
        </div>
      </section>

      <section id="pricing" className="container" style={{ padding: '4rem 0 6rem 0' }}>
        <div className="text-center mb-8">
          <h2 className="pixel-font" style={{ fontSize: '1.6rem', color: 'var(--primary-neon)' }}>&gt; PRICING</h2>
        </div>
        <div className="grid grid-cols-2 gap-8">
          <div className="pixel-border pricing-card" style={{ borderColor: '#cbd5e1', boxShadow: '4px 4px 0 0 #cbd5e1' }}>
            <h3 className="pixel-font mb-2">LOCAL</h3>
            <div className="pixel-font text-gradient mb-4" style={{ fontSize: '2rem' }}>Free</div>
            <ul style={{ listStyle: 'none', padding: 0, marginBottom: '2rem', display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
              {['Every optimizer tool, offline', 'Exact token counts', 'Optional local Ollama tools for images and text', 'Personal and internal business use'].map((f) => (
                <li key={f} style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}><ChevronRight size={16} color="var(--primary-neon)" /> {f}</li>
              ))}
            </ul>
            <a href="#install" className="pixel-btn pixel-btn-secondary" style={{ width: '100%', justifyContent: 'center' }}>Install</a>
          </div>
          <div className="pixel-border pricing-card">
            <h3 className="pixel-font mb-2">HOSTED</h3>
            <div className="pixel-font text-gradient mb-4" style={{ fontSize: '2rem' }}>from $0</div>
            <ul style={{ listStyle: 'none', padding: 0, marginBottom: '2rem', display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
              {['Free · 300 calls / month', 'Starter · $9 · 3,000 calls', 'Pro · $19 · 15,000 calls', 'Team · $49 · 60,000 calls'].map((f) => (
                <li key={f} style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}><ChevronRight size={16} color="var(--secondary-neon)" /> {f}</li>
              ))}
            </ul>
            <a href={HOSTED_URL} target="_blank" rel="noreferrer" className="pixel-btn" style={{ width: '100%', justifyContent: 'center' }}>Choose a plan</a>
          </div>
        </div>
      </section>

      <footer style={{ borderTop: '2px dashed var(--border-color)', padding: '3rem 0', textAlign: 'center', marginTop: '2rem' }}>
        <div className="pixel-font" style={{ fontSize: '1.3rem', color: 'var(--primary-neon)', marginBottom: '1rem' }}>LatentGate</div>
        <p style={{ color: '#64748b', marginBottom: '1.5rem' }}>Give your AI the signal, not the noise.</p>
        <p style={{ color: '#475569', fontSize: '0.9rem' }}>© 2026 Kathan Modh · Proprietary licence · <a href={GITHUB_URL} target="_blank" rel="noreferrer">GitHub</a></p>
      </footer>
    </div>
  );
}

export default App;
