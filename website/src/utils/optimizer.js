// Runs the real LatentGate optimizer in the visitor's browser (Pyodide = CPython on
// WebAssembly). Same code as the PyPI package, so the demo shows exactly what users get;
// nothing is uploaded and there is no server to pay for or abuse.
const PYODIDE_URL = 'https://cdn.jsdelivr.net/pyodide/v0.27.7/full/';
export const PACKAGE_VERSION = '1.4.2';

let runtime = null;

function loadScript(src) {
  return new Promise((resolve, reject) => {
    const s = document.createElement('script');
    s.src = src;
    s.onload = resolve;
    s.onerror = () => reject(new Error('Could not download the Python runtime'));
    document.head.appendChild(s);
  });
}

// Loads once; later calls reuse it. A failed load is forgotten so the user can retry.
export function loadOptimizer(onStatus = () => {}) {
  runtime ??= (async () => {
    onStatus('Downloading the Python runtime (about 10 MB, first time only)…');
    await loadScript(`${PYODIDE_URL}pyodide.js`);
    const py = await window.loadPyodide({ indexURL: PYODIDE_URL });
    onStatus(`Installing latent-gate ${PACKAGE_VERSION} from PyPI…`);
    await py.loadPackage('micropip');
    // deps=False: the optimizer only needs the standard library
    await py.runPythonAsync(
      `import micropip\nawait micropip.install("latent-gate==${PACKAGE_VERSION}", deps=False)`
    );
    py.runPython('import json\nfrom latent_gate.optimizer import TokenOptimizer');
    return py;
  })().catch((err) => {
    runtime = null;
    throw err;
  });
  return runtime;
}

export async function optimize(text, level) {
  const py = await loadOptimizer();
  py.globals.set('_text', text);
  py.globals.set('_level', level);
  const started = performance.now();
  const out = py.runPython(
    'r = TokenOptimizer(_level).optimize(_text)\n' +
      'json.dumps({"text": r.text, "original": r.original_tokens, ' +
      '"optimized": r.optimized_tokens, "counter": r.counter})'
  );
  return { ...JSON.parse(out), ms: performance.now() - started };
}
