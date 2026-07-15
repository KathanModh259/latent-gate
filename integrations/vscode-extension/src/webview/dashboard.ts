import * as vscode from 'vscode';
import { LatentGateClient } from '../client';

export class DashboardProvider implements vscode.WebviewViewProvider {
    public static readonly viewType = 'latentGate.dashboard';
    private _view?: vscode.WebviewView;
    private _extensionUri: vscode.Uri;
    private client: LatentGateClient;

    constructor(extensionUri: vscode.Uri, client: LatentGateClient) {
        this._extensionUri = extensionUri;
        this.client = client;
    }

    public resolveWebviewView(webviewView: vscode.WebviewView) {
        this._view = webviewView;

        webviewView.webview.options = {
            enableScripts: true,
            localResourceRoots: [this._extensionUri],
        };

        webviewView.webview.html = this._getHtmlForWebview(webviewView.webview);

        webviewView.webview.onDidReceiveMessage(async (message) => {
            switch (message.command) {
                case 'compressText':
                    if (message.text && message.text.trim()) {
                        vscode.commands.executeCommand('latentGate.compressSelection');
                    }
                    break;
                case 'compressImage':
                    vscode.commands.executeCommand('latentGate.compressImage');
                    break;
                case 'compressDocument':
                    vscode.commands.executeCommand('latentGate.compressDocument');
                    break;
                case 'quickCompress':
                    await this.handleQuickCompress(message.text);
                    break;
                case 'checkHealth':
                    const healthy = await this.client.checkHealth();
                    this._view?.webview.postMessage({ type: 'health', healthy });
                    break;
                case 'getStats':
                    const stats = await this.client.getSessionStats();
                    this._view?.webview.postMessage({ type: 'stats', stats });
                    break;
            }
        });

        this.client.onDidChangeStats((stats) => {
            this._view?.webview.postMessage({ type: 'stats', stats });
        });
    }

    private async handleQuickCompress(text: string) {
        this._view?.webview.postMessage({ type: 'loading', loading: true });
        try {
            const result = await this.client.compressText(text);
            this.client.updateStats(result);
            this._view?.webview.postMessage({
                type: 'compressResult',
                result: {
                    originalTokens: result.tokensEstimated,
                    compressedPrompt: result.compactPrompt,
                    compressionRatio: result.compressionRatio || '1.0x',
                    tokensSaved: result.tokensSaved || 0,
                },
            });
        } catch (err: any) {
            this._view?.webview.postMessage({ type: 'error', message: err.message });
        }
    }

    public show() {
        this._view?.show();
    }

    public refresh() {
        if (this._view) {
            const stats = this.client.getSessionStats();
            this._view.webview.postMessage({ type: 'stats', stats });
        }
    }

    private _getHtmlForWebview(webview: vscode.Webview): string {
        const nonce = getNonce();
        return `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LatentGate Dashboard</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: var(--vscode-font-family);
            font-size: var(--vscode-font-size);
            color: var(--vscode-foreground);
            background: var(--vscode-sideBar-background);
            padding: 12px;
        }

        .header {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 8px 0 12px;
            border-bottom: 1px solid var(--vscode-widget-border);
            margin-bottom: 12px;
        }
        .header-logo {
            width: 24px;
            height: 24px;
            background: var(--vscode-terminal-ansiCyan);
            border-radius: 6px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 14px;
            font-weight: 700;
            color: #fff;
        }
        .header-text h1 {
            font-size: 13px;
            font-weight: 600;
            line-height: 1.2;
        }
        .header-text span {
            font-size: 11px;
            color: var(--vscode-descriptionForeground);
        }

        .status-bar {
            display: flex;
            align-items: center;
            gap: 6px;
            padding: 6px 10px;
            border-radius: 6px;
            background: var(--vscode-input-background);
            margin-bottom: 12px;
            font-size: 12px;
        }
        .status-dot {
            width: 7px;
            height: 7px;
            border-radius: 50%;
            background: var(--vscode-errorForeground);
            flex-shrink: 0;
        }
        .status-dot.healthy { background: var(--vscode-terminal-ansiGreen); }
        .status-dot.loading { background: var(--vscode-terminal-ansiYellow); animation: pulse 1s infinite; }
        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.4; } }

        .stats-row {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 6px;
            margin-bottom: 12px;
        }
        .stat-card {
            padding: 10px 8px;
            border-radius: 6px;
            background: var(--vscode-input-background);
            text-align: center;
        }
        .stat-value {
            font-size: 18px;
            font-weight: 700;
            color: var(--vscode-terminal-ansiCyan);
            line-height: 1.2;
        }
        .stat-label {
            font-size: 10px;
            color: var(--vscode-descriptionForeground);
            margin-top: 2px;
            text-transform: uppercase;
            letter-spacing: 0.3px;
        }

        .section-title {
            font-size: 11px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: var(--vscode-sideBarSectionHeader-foreground);
            margin-bottom: 8px;
        }

        .quick-compress {
            margin-bottom: 12px;
        }
        .quick-compress textarea {
            width: 100%;
            min-height: 60px;
            padding: 8px;
            border: 1px solid var(--vscode-input-border);
            border-radius: 6px;
            background: var(--vscode-input-background);
            color: var(--vscode-input-foreground);
            font-family: var(--vscode-font-family);
            font-size: 12px;
            resize: vertical;
            outline: none;
        }
        .quick-compress textarea:focus {
            border-color: var(--vscode-terminal-ansiCyan);
        }
        .quick-compress textarea::placeholder {
            color: var(--vscode-input-placeholderForeground);
        }

        .btn-row {
            display: flex;
            gap: 6px;
            margin-top: 6px;
        }
        .btn {
            flex: 1;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 5px;
            padding: 7px 10px;
            border: none;
            border-radius: 6px;
            font-size: 12px;
            font-family: var(--vscode-font-family);
            cursor: pointer;
            transition: opacity 0.15s;
        }
        .btn:hover { opacity: 0.85; }
        .btn:active { opacity: 0.7; }
        .btn-primary {
            background: var(--vscode-button-background);
            color: var(--vscode-button-foreground);
        }
        .btn-secondary {
            background: var(--vscode-button-secondaryBackground);
            color: var(--vscode-button-secondaryForeground);
        }
        .btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }
        .btn .icon { font-size: 13px; }

        .result-box {
            display: none;
            padding: 10px;
            border-radius: 6px;
            background: var(--vscode-textBlockQuote-background);
            border-left: 3px solid var(--vscode-terminal-ansiCyan);
            margin-top: 8px;
            font-size: 12px;
        }
        .result-box.visible { display: block; }
        .result-box .result-label {
            font-size: 10px;
            text-transform: uppercase;
            letter-spacing: 0.3px;
            color: var(--vscode-descriptionForeground);
            margin-bottom: 4px;
        }
        .result-box pre {
            white-space: pre-wrap;
            word-break: break-word;
            font-family: var(--vscode-editor-font-family);
            font-size: 11px;
            line-height: 1.4;
        }

        .error-box {
            display: none;
            padding: 8px 10px;
            border-radius: 6px;
            background: var(--vscode-inputValidation-errorBackground);
            border: 1px solid var(--vscode-inputValidation-errorBorder);
            color: var(--vscode-errorForeground);
            font-size: 11px;
            margin-top: 8px;
        }
        .error-box.visible { display: block; }

        .hint {
            font-size: 11px;
            color: var(--vscode-descriptionForeground);
            margin-top: 12px;
            padding: 8px;
            border-radius: 6px;
            background: var(--vscode-textBlockQuote-background);
            line-height: 1.5;
        }
        kbd {
            padding: 1px 4px;
            border-radius: 3px;
            background: var(--vscode-keybindingLabel-background);
            border: 1px solid var(--vscode-keybindingLabel-border);
            font-family: var(--vscode-editor-font-family);
            font-size: 10px;
        }
    </style>
</head>
<body>
    <div class="header">
        <div class="header-logo">LG</div>
        <div class="header-text">
            <h1>LatentGate</h1>
            <span>Process Locally. Send Smart.</span>
        </div>
    </div>

    <div class="status-bar" id="statusBar">
        <div class="status-dot" id="statusDot"></div>
        <span id="statusText">Checking...</span>
    </div>

    <div class="stats-row">
        <div class="stat-card">
            <div class="stat-value" id="tokensSaved">0</div>
            <div class="stat-label">Tokens Saved</div>
        </div>
        <div class="stat-card">
            <div class="stat-value" id="queries">0</div>
            <div class="stat-label">Queries</div>
        </div>
    </div>

    <div class="quick-compress">
        <div class="section-title">Quick Compress</div>
        <textarea id="inputText" placeholder="Paste or type text to compress..." rows="3"></textarea>
        <div class="btn-row">
            <button class="btn btn-primary" id="btnCompress" type="button">
                <span>&#9889;</span> Compress
            </button>
            <button class="btn btn-secondary" id="btnImage" type="button">
                <span>&#128247;</span> Image
            </button>
        </div>
        <div class="result-box" id="resultBox">
            <div class="result-label">Compressed Output</div>
            <pre id="resultText"></pre>
        </div>
        <div class="error-box" id="errorBox"></div>
    </div>

    <div class="hint">
        Select text in editor, press <kbd>Ctrl</kbd>+<kbd>Shift</kbd>+<kbd>Alt</kbd>+<kbd>C</kbd> to compress.
    </div>

    <script nonce="${nonce}">
        const vscode = acquireVsCodeApi();
        const inputEl = document.getElementById('inputText');
        const btnCompress = document.getElementById('btnCompress');
        const btnImage = document.getElementById('btnImage');
        const resultBox = document.getElementById('resultBox');
        const resultText = document.getElementById('resultText');
        const errorBox = document.getElementById('errorBox');
        const statusDot = document.getElementById('statusDot');
        const statusText = document.getElementById('statusText');

        vscode.postMessage({ command: 'checkHealth' });
        vscode.postMessage({ command: 'getStats' });

        btnCompress.addEventListener('click', () => {
            const text = inputEl.value.trim();
            if (!text) { inputEl.focus(); return; }
            setLoading(true);
            errorBox.classList.remove('visible');
            resultBox.classList.remove('visible');
            vscode.postMessage({ command: 'quickCompress', text });
        });

        btnImage.addEventListener('click', () => {
            vscode.postMessage({ command: 'compressImage' });
        });

        inputEl.addEventListener('keydown', (e) => {
            if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
                btnCompress.click();
            }
        });

        window.addEventListener('message', event => {
            const msg = event.data;
            switch (msg.type) {
                case 'health':
                    if (msg.healthy) {
                        statusDot.classList.add('healthy');
                        statusDot.classList.remove('loading');
                        statusText.textContent = 'Ollama connected';
                    } else {
                        statusDot.classList.remove('healthy', 'loading');
                        statusText.textContent = 'Ollama offline';
                    }
                    break;
                case 'stats':
                    document.getElementById('tokensSaved').textContent = msg.stats.apiCalls * 150 || '0';
                    document.getElementById('queries').textContent = msg.stats.apiCalls || '0';
                    break;
                case 'loading':
                    setLoading(msg.loading);
                    break;
                case 'compressResult':
                    setLoading(false);
                    resultBox.classList.add('visible');
                    resultText.textContent = msg.result.compressedPrompt;
                    break;
                case 'error':
                    setLoading(false);
                    errorBox.classList.add('visible');
                    errorBox.textContent = msg.message;
                    break;
            }
        });

        function setLoading(loading) {
            btnCompress.disabled = loading;
            btnCompress.innerHTML = loading
                ? '<span class="icon">&#8987;</span> Compressing...'
                : '<span>&#9889;</span> Compress';
            if (loading) {
                statusDot.classList.add('loading');
                statusText.textContent = 'Compressing...';
            } else {
                statusDot.classList.remove('loading');
                vscode.postMessage({ command: 'checkHealth' });
            }
        }
    </script>
</body>
</html>`;
    }
}

function getNonce() {
    let text = '';
    const possible = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
    for (let i = 0; i < 32; i++) {
        text += possible.charAt(Math.floor(Math.random() * possible.length));
    }
    return text;
}
