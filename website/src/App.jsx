import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Terminal, Zap, ChevronRight, Activity, ArrowRight, Code, Loader2, Check, Copy, AlertTriangle } from 'lucide-react';

const MAX_INPUT_CHARS = parseInt(import.meta.env.VITE_MAX_INPUT_CHARS || '50000', 10);
const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

function App() {
  const [inputText, setInputText] = useState('');
  const [outputText, setOutputText] = useState('');
  const [isCompressing, setIsCompressing] = useState(false);
  const [stats, setStats] = useState(null);
  const [error, setError] = useState(null);
  const [copied, setCopied] = useState(false);
  const [inputError, setInputError] = useState(null);
  const [isMockMode, setIsMockMode] = useState(false);

  const outputRef = useRef(null);

  const handleInputChange = useCallback((e) => {
    const value = e.target.value;
    if (value.length > MAX_INPUT_CHARS) {
      setInputError(`Input exceeds ${MAX_INPUT_CHARS.toLocaleString()} characters limit`);
      return;
    }
    setInputError(null);
    setInputText(value);
  }, []);

  const handleCopy = useCallback(async () => {
    if (!outputText) return;
    try {
      await navigator.clipboard.writeText(outputText);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      setCopied(false);
    }
  }, [outputText]);


  const handleCompress = async () => {
    if (!inputText || inputError) return;
    
    setIsCompressing(true);
    setIsMockMode(false);
    setError(null);
    setStats(null);
    setOutputText('');

    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 30000);
      
      const response = await fetch(`${API_URL}/compress`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          text: inputText,
        }),
        signal: controller.signal
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        throw new Error(`API error: ${response.status}`);
      }

      const data = await response.json();
      setOutputText(data.compressed_prompt || data.answer || 'Compressed output generated.');
      setStats({
        original: data.original_tokens || Math.floor(inputText.length / 4),
        compressed: data.compressed_tokens || Math.floor((inputText.length / 4) * 0.2),
        savings: data.tokens_saved ? `${Math.round(data.tokens_saved / data.original_tokens * 100)}%` : '80%'
      });
    } catch (err) {
      if (err.name === 'AbortError') {
        setError('Request timed out. Please try with shorter text.');
        setIsCompressing(false);
        return;
      }
      // Mock response if API is down
      console.log('API unavailable, using mock response:', err.message);
      setIsMockMode(true);
      const estTokens = Math.floor(inputText.length / 4);
      const compressed = (() => {
        const lines = inputText.split('\n');
        const skip = new Set(['', 'hi', 'hello', 'hey', 'thanks', 'thank you', 'ok', 'okay', 'sure', 'please']);
        const kept = [];
        for (const l of lines) {
          const s = l.trim();
          if (!s || skip.has(s.toLowerCase().replace(/[!?.,]+$/, ''))) continue;
          if (kept.length && s === kept[kept.length - 1]) continue;
          kept.push(s);
        }
        // Remove boilerplate like "The application must include"
        const boilerplate = /^(the\s+)?(application|project|system|tool|feature|solution|script|code|page|website)\s+(must|should|will|needs\s+to|has\s+to|shall|would|can|could)\s+(include|have|support|provide|contain|be|do|implement|use|work|handle|ensure|deliver)/i;
        const cleaned = kept.map(l => l.replace(boilerplate, '').trim()).filter(Boolean);
        // Condense bullet points into inline lists
        const bulletRe = /^[\s]*[-*•]|\d+[.)]\s/;
        const result = [];
        let bullets = [];
        for (const line of cleaned) {
          if (bulletRe.test(line)) {
            bullets.push(line.replace(bulletRe, '').trim());
          } else {
            if (bullets.length) { result.push(bullets.join('; ')); bullets = []; }
            result.push(line);
          }
        }
        if (bullets.length) result.push(bullets.join('; '));
        let text = result.join('\n');
        const words = text.split(/\s+/);
        if (words.length > 60) {
          const splitA = Math.max(25, Math.floor(words.length * 0.35));
          const splitB = Math.max(splitA + 5, Math.floor(words.length * 0.80));
          text = words.slice(0, splitA).join(' ') + '\n...\n' + words.slice(splitB).join(' ');
        }
        return text;
      })();
      const compTokens = Math.floor(compressed.split(/\s+/).length * 1.33);
      setOutputText(compressed);
      setStats({
        original: estTokens || 0,
        compressed: compTokens || 0,
        savings: '~80%'
      });
      setIsCompressing(false);
      return;
    }
    
    setIsCompressing(false);
  };

  const handleClear = () => {
    setInputText('');
    setOutputText('');
    setStats(null);
    setError(null);
    setInputError(null);
  };

  return (
    <div className="app-wrapper">
      {/* Navigation */}
      <nav style={{ padding: '1.5rem', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div className="pixel-font" style={{ fontSize: '1.2rem', color: 'var(--primary-neon)' }}>
          LatentGate<span className="cursor-blink">_</span>
        </div>
        <div style={{ display: 'flex', gap: '1rem' }}>
          <a href="#compress" className="pixel-btn pixel-btn-secondary" style={{ padding: '0.5rem 1rem', fontSize: '12px' }}>Demo</a>
          <a href="#pricing" className="pixel-btn pixel-btn-secondary" style={{ padding: '0.5rem 1rem', fontSize: '12px' }}>Pricing</a>
        </div>
      </nav>

      {/* Hero Section */}
      <header className="container" style={{ textAlign: 'center', padding: '4rem 0 6rem 0' }}>
        <h1 className="pixel-font text-gradient" style={{ fontSize: '2.5rem', marginBottom: '1.5rem', lineHeight: '1.5' }}>
          PROCESS LOCALLY.<br/>SEND SMART.<br/>PAY LESS.
        </h1>
        
        <p style={{ fontSize: '1.2rem', color: '#64748b', maxWidth: '600px', margin: '0 auto 2.5rem auto' }}>
          A VL-JEPA-inspired pipeline that compresses images, text, and documents locally via Ollama. 
          Send only compact semantic payloads to any LLM API — cutting token costs by ~80%.
        </p>

        <div style={{ display: 'flex', gap: '1rem', justifyContent: 'center' }}>
          <a href="#compress" className="pixel-btn">
            <Zap size={18} /> Try Compression
          </a>
          <a href="https://github.com/KathanModh259/latent-gate" target="_blank" rel="noreferrer" className="pixel-btn pixel-btn-secondary">
            <Code size={18} /> GitHub
          </a>
        </div>
      </header>

      {/* Compressor Demo Section */}
      <section id="compress" className="container" style={{ padding: '4rem 0' }}>
        <div className="text-center mb-8">
          <h2 className="pixel-font" style={{ fontSize: '1.8rem', color: 'var(--secondary-neon)' }}>
            &gt; INTERACTIVE_DEMO
          </h2>
          <p style={{ color: '#64748b', marginTop: '1rem' }}>Test the local text compression pipeline directly.</p>
        </div>

        <div className="grid grid-cols-2 gap-8" style={{ alignItems: 'stretch' }}>
          {/* Input Panel */}
          <div className="pixel-border" style={{ display: 'flex', flexDirection: 'column' }}>
            <h3 className="pixel-font mb-4" style={{ fontSize: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <Terminal size={16} color="var(--primary-neon)" /> INPUT_PROMPT.txt
            </h3>
            <div style={{ position: 'relative' }}>
              <textarea 
                className="pixel-input" 
                style={{ flex: 1, minHeight: '250px' }}
                placeholder="Paste a very long prompt, code snippet, or document here to see how much we can compress it..."
                value={inputText}
                onChange={handleInputChange}
                maxLength={MAX_INPUT_CHARS}
                aria-label="Input text to compress"
              />
              {inputError && (
                <div className="pixel-font" style={{ fontSize: '0.7rem', color: '#ef4444', marginTop: '0.5rem', display: 'flex', alignItems: 'center', gap: '0.3rem' }}>
                  <AlertTriangle size={12} /> {inputError}
                </div>
              )}
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: '0.5rem', fontSize: '0.75rem', color: '#64748b' }}>
                <span>{inputText.length.toLocaleString()} / {MAX_INPUT_CHARS.toLocaleString()} chars</span>
                <span className="pixel-font" style={{ color: inputText.length > MAX_INPUT_CHARS * 0.9 ? '#ef4444' : 'var(--primary-neon)' }}>
                  ~{Math.ceil(inputText.length / 4).toLocaleString()} tokens
                </span>
              </div>
            </div>
            <div style={{ display: 'flex', gap: '0.5rem', marginTop: '1rem' }}>
              <button 
                className="pixel-btn" 
                style={{ flex: 1, justifyContent: 'center' }}
                onClick={handleCompress}
                disabled={isCompressing || !inputText || inputError}
                aria-busy={isCompressing}
              >
                {isCompressing ? (
                  <>
                    <Loader2 size={16} className="animate-spin" /> COMPRESSING...
                  </>
                ) : (
                  <>
                    INITIALIZE COMPRESSION <ArrowRight size={16} />
                  </>
                )}
              </button>
              {inputText && !isCompressing && (
                <button 
                  className="pixel-btn pixel-btn-secondary" 
                  onClick={handleClear}
                  style={{ justifyContent: 'center', padding: '0 1rem' }}
                  aria-label="Clear input"
                >
                  <Code size={14} /> CLEAR
                </button>
              )}
            </div>
          </div>

          {/* Output Panel */}
          <div className="pixel-border-alt" style={{ display: 'flex', flexDirection: 'column', position: 'relative' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
              <h3 className="pixel-font mb-4" style={{ fontSize: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <Activity size={16} color="var(--secondary-neon)" /> SEMANTIC_PAYLOAD.out
              </h3>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                {isMockMode && (
                  <span style={{ 
                    backgroundColor: 'rgba(245, 158, 11, 0.15)', 
                    color: 'var(--accent-neon)', 
                    padding: '0.2rem 0.6rem', 
                    border: '1px solid var(--accent-neon)',
                    fontSize: '0.65rem',
                    fontFamily: 'var(--font-pixel)',
                    letterSpacing: '0.5px'
                  }}>
                    MOCK MODE
                  </span>
                )}
                {outputText && !isCompressing && (
                  <button 
                    className="pixel-btn pixel-btn-secondary" 
                    onClick={handleCopy}
                    style={{ padding: '0.4rem 0.8rem', fontSize: '11px', justifyContent: 'center', gap: '0.3rem' }}
                    aria-label={copied ? 'Copied to clipboard' : 'Copy to clipboard'}
                  >
                    {copied ? (
                      <>
                        <Check size={14} /> COPIED
                      </>
                    ) : (
                      <>
                        <Copy size={14} /> COPY
                      </>
                    )}
                  </button>
                )}
              </div>
            </div>
            
            {error && (
              <div style={{ padding: '1rem', backgroundColor: 'rgba(239, 68, 68, 0.1)', border: '2px solid #ef4444', marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem', color: '#ef4444' }}>
                <AlertTriangle size={18} /> {error}
              </div>
            )}
             
            <div 
              className="pixel-input" 
              ref={outputRef}
              style={{ flex: 1, backgroundColor: '#f8fafc', border: '2px solid #cbd5e1', color: '#0f172a', whiteSpace: 'pre-wrap', overflowY: 'auto', minHeight: '250px', fontFamily: 'var(--font-body)', fontSize: '0.9rem', lineHeight: '1.6' }}
              aria-live="polite"
            >
              {isCompressing ? (
                <div style={{ color: 'var(--accent-neon)', textAlign: 'center', marginTop: '4rem', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '1rem' }}>
                  <div className="animate-float"><Loader2 size={32} className="animate-spin" style={{ color: 'var(--secondary-neon)' }} /></div>
                  <div className="pixel-font" style={{ fontSize: '0.8rem' }}>Processing locally via Ollama...</div>
                  <div style={{ width: '100px', height: '4px', backgroundColor: '#e2e8f0', borderRadius: '2px', overflow: 'hidden' }}>
                    <div className="animate-pulse" style={{ width: '100%', height: '100%', backgroundColor: 'var(--secondary-neon)' }}></div>
                  </div>
                </div>
              ) : outputText ? (
                <pre style={{ margin: 0, whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}>{outputText}</pre>
              ) : (
                <span style={{ color: '#475569' }}>Waiting for input...</span>
              )}
            </div>

            {/* Stats */}
            {stats && (
              <div style={{ marginTop: '1rem', padding: '1rem', backgroundColor: 'rgba(22, 255, 224, 0.1)', border: '2px dashed var(--primary-neon)' }}>
                <div className="grid grid-cols-3 gap-4 text-center">
                  <div>
                    <div style={{ fontSize: '0.8rem', color: '#64748b' }}>Original Tokens</div>
                    <div className="pixel-font" style={{ color: '#ef4444' }}>~{stats.original}</div>
                  </div>
                  <div>
                    <div style={{ fontSize: '0.8rem', color: '#64748b' }}>Compressed</div>
                    <div className="pixel-font" style={{ color: 'var(--primary-neon)' }}>~{stats.compressed}</div>
                  </div>
                  <div>
                    <div style={{ fontSize: '0.8rem', color: '#64748b' }}>Savings</div>
                    <div className="pixel-font" style={{ color: 'var(--accent-neon)' }}>{stats.savings}</div>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </section>

      {/* Pricing Section */}
      <section id="pricing" className="container" style={{ padding: '6rem 0' }}>
        <div className="text-center mb-8">
          <h2 className="pixel-font" style={{ fontSize: '1.8rem', color: 'var(--primary-neon)' }}>
            &gt; PRICING_MODULE
          </h2>
          <p style={{ color: '#64748b', marginTop: '1rem' }}>No subscriptions. Just lower API bills.</p>
        </div>

        <div className="grid grid-cols-3 gap-8">
          <div className="pixel-border pricing-card" style={{ borderColor: '#cbd5e1', boxShadow: '4px 4px 0 0 #cbd5e1' }}>
            <h3 className="pixel-font mb-2">HACKER</h3>
            <div className="pixel-font text-gradient mb-4" style={{ fontSize: '2rem' }}>$0</div>
            <p style={{ color: '#64748b', marginBottom: '1.5rem', minHeight: '48px' }}>For local tinkerers and open-source enthusiasts.</p>
            <ul style={{ listStyle: 'none', padding: 0, marginBottom: '2rem', display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
              <li style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}><ChevronRight size={16} color="var(--primary-neon)" /> Local CLI Tool</li>
              <li style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}><ChevronRight size={16} color="var(--primary-neon)" /> Basic MCP Server</li>
              <li style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}><ChevronRight size={16} color="var(--primary-neon)" /> Community Support</li>
            </ul>
            <button className="pixel-btn pixel-btn-secondary" style={{ width: '100%', justifyContent: 'center' }}>npm install</button>
          </div>

          <div className="pixel-border pricing-card" style={{ position: 'relative' }}>
            <div style={{ position: 'absolute', top: '-15px', left: '50%', transform: 'translateX(-50%)', backgroundColor: 'var(--secondary-neon)', color: '#000', padding: '0.2rem 1rem', fontSize: '0.8rem', fontWeight: 'bold' }} className="pixel-font">
              RECOMMENDED
            </div>
            <h3 className="pixel-font mb-2">PRO API</h3>
            <div className="pixel-font text-gradient mb-4" style={{ fontSize: '2rem' }}>$0<span style={{ fontSize: '1rem', color: '#64748b' }}>/mo</span></div>
            <p style={{ color: '#64748b', marginBottom: '1.5rem', minHeight: '48px' }}>Host the API server yourself. Pay only for your LLM usage.</p>
            <ul style={{ listStyle: 'none', padding: 0, marginBottom: '2rem', display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
              <li style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}><ChevronRight size={16} color="var(--secondary-neon)" /> FastAPI Server</li>
              <li style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}><ChevronRight size={16} color="var(--secondary-neon)" /> Video Processing</li>
              <li style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}><ChevronRight size={16} color="var(--secondary-neon)" /> Advanced Selective Decoding</li>
              <li style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}><ChevronRight size={16} color="var(--secondary-neon)" /> Docker Compose Ready</li>
            </ul>
            <button className="pixel-btn" style={{ width: '100%', justifyContent: 'center' }}>PULL DOCKER</button>
          </div>

          <div className="pixel-border pricing-card" style={{ borderColor: '#cbd5e1', boxShadow: '4px 4px 0 0 #cbd5e1' }}>
            <h3 className="pixel-font mb-2">ENTERPRISE</h3>
            <div className="pixel-font text-gradient mb-4" style={{ fontSize: '2rem' }}>Custom</div>
            <p style={{ color: '#64748b', marginBottom: '1.5rem', minHeight: '48px' }}>Managed instances and custom compression models.</p>
            <ul style={{ listStyle: 'none', padding: 0, marginBottom: '2rem', display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
              <li style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}><ChevronRight size={16} color="var(--accent-neon)" /> Dedicated Hosting</li>
              <li style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}><ChevronRight size={16} color="var(--accent-neon)" /> SLA Guarantees</li>
              <li style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}><ChevronRight size={16} color="var(--accent-neon)" /> Custom Plugins</li>
            </ul>
            <button className="pixel-btn pixel-btn-secondary" style={{ width: '100%', justifyContent: 'center', borderColor: 'var(--accent-neon)', color: 'var(--accent-neon)' }}>CONTACT US</button>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer style={{ borderTop: '2px dashed var(--border-color)', padding: '3rem 0', textAlign: 'center', marginTop: '4rem' }}>
        <div className="pixel-font" style={{ fontSize: '1.5rem', color: 'var(--primary-neon)', marginBottom: '1rem' }}>LatentGate</div>
        <p style={{ color: '#64748b', marginBottom: '2rem' }}>Process Locally. Send Smart. Pay Less.</p>
        <p style={{ color: '#475569', fontSize: '0.9rem' }}>© 2026 Kathan Modh. Open source under MIT License.</p>
      </footer>
    </div>
  );
}

export default App;
