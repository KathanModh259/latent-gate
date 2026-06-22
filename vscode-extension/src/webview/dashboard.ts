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

    public resolveWebviewView(
        webviewView: vscode.WebviewView,
    ) {
        this._view = webviewView;

        webviewView.webview.options = {
            enableScripts: true,
            localResourceRoots: [this._extensionUri],
        };

        webviewView.webview.html = this._getHtmlForWebview(webviewView.webview);

        // Handle messages from webview
        webviewView.webview.onDidReceiveMessage(async (message) => {
            switch (message.command) {
                case 'compressImage':
                    vscode.commands.executeCommand('latentGate.compressImage');
                    break;
                case 'compressText':
                    vscode.commands.executeCommand('latentGate.compressSelection');
                    break;
                case 'compressDocument':
                    vscode.commands.executeCommand('latentGate.compressDocument');
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

        // Listen for stats changes
        this.client.onDidChangeStats((stats) => {
            this._view?.webview.postMessage({ type: 'stats', stats });
        });
    }

    public show() {
        this._view?.show();
    }

    public refresh() {
        if (this._view) {
            this._view.webview.postMessage({ type: 'refresh' });
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
        :root {
            --container-padding: 20px;
            --input-padding: 6px 10px;
            --input-radius: 3px;
        }

        body {
            padding: 0;
            margin: 0;
            font-family: var(--vscode-font-family);
            font-size: var(--vscode-font-size);
            color: var(--vscode-foreground);
            background: var(--vscode-sideBar-background);
        }

        .container {
            padding: var(--container-padding);
        }

        h2 {
            margin-top: 0;
            font-size: 14px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: var(--vscode-sideBarSectionHeader-foreground);
            border-bottom: 1px solid var(--vscode-sideBarSectionHeader-border);
            padding-bottom: 8px;
        }

        .status {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 8px 12px;
            border-radius: 4px;
            margin-bottom: 12px;
            background: var(--vscode-input-background);
        }

        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: var(--vscode-errorForeground);
        }

        .status-dot.healthy {
            background: var(--vscode-terminal-ansiGreen);
        }

        .stats-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 8px;
            margin-bottom: 16px;
        }

        .stat-card {
            padding: 12px;
            border-radius: 4px;
            background: var(--vscode-input-background);
            text-align: center;
        }

        .stat-value {
            font-size: 20px;
            font-weight: bold;
            color: var(--vscode-terminal-ansiCyan);
        }

        .stat-label {
            font-size: 11px;
            color: var(--vscode-descriptionForeground);
            margin-top: 4px;
        }

        .actions {
            display: flex;
            flex-direction: column;
            gap: 8px;
        }

        .action-btn {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 10px 12px;
            border: none;
            border-radius: 4px;
            background: var(--vscode-button-background);
            color: var(--vscode-button-foreground);
            cursor: pointer;
            font-size: 13px;
            text-align: left;
            width: 100%;
        }

        .action-btn:hover {
            background: var(--vscode-button-hoverBackground);
        }

        .action-btn .icon {
            font-size: 16px;
        }

        .section {
            margin-bottom: 20px;
        }

        .hint {
            font-size: 11px;
            color: var(--vscode-descriptionForeground);
            margin-top: 16px;
            padding: 8px;
            border-radius: 4px;
            background: var(--vscode-textBlockQuote-background);
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="section">
            <h2>Status</h2>
            <div class="status" id="status">
                <div class="status-dot" id="statusDot"></div>
                <span id="statusText">Checking...</span>
            </div>
        </div>

        <div class="section">
            <h2>Session Stats</h2>
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-value" id="totalTokens">0</div>
                    <div class="stat-label">Tokens Saved</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" id="totalQueries">0</div>
                    <div class="stat-label">Queries</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" id="skipRate">0%</div>
                    <div class="stat-label">Skip Rate</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" id="reductionRatio">1.0x</div>
                    <div class="stat-label">Cost Reduction</div>
                </div>
            </div>
        </div>

        <div class="section">
            <h2>Quick Actions</h2>
            <div class="actions">
                <button class="action-btn" id="btnImage">
                    <span class="icon">$(file-binary)</span>
                    Compress Image
                </button>
                <button class="action-btn" id="btnText">
                    <span class="icon">$(compress)</span>
                    Compress Selection
                </button>
                <button class="action-btn" id="btnDocument">
                    <span class="icon">$(file-text)</span>
                    Compress Document
                </button>
            </div>
        </div>

        <div class="hint">
            💡 Select text and press <kbd>Ctrl+Shift+Alt+C</kbd> to compress quickly.
        </div>
    </div>

    <script nonce="${nonce}">
        const vscode = acquireVsCodeApi();

        // Check health
        vscode.postMessage({ command: 'checkHealth' });

        // Get stats
        vscode.postMessage({ command: 'getStats' });

        // Handle messages from extension
        window.addEventListener('message', event => {
            const message = event.data;
            switch (message.type) {
                case 'health':
                    const dot = document.getElementById('statusDot');
                    const text = document.getElementById('statusText');
                    if (message.healthy) {
                        dot.classList.add('healthy');
                        text.textContent = 'Ollama Connected';
                    } else {
                        dot.classList.remove('healthy');
                        text.textContent = 'Ollama Offline';
                    }
                    break;
                case 'stats':
                    document.getElementById('totalTokens').textContent = message.stats.apiCalls;
                    document.getElementById('totalQueries').textContent = message.stats.totalFrames;
                    document.getElementById('skipRate').textContent = message.stats.skipRate;
                    document.getElementById('reductionRatio').textContent = message.stats.reductionRatio;
                    break;
            }
        });

        // Button handlers
        document.getElementById('btnImage').addEventListener('click', () => {
            vscode.postMessage({ command: 'compressImage' });
        });
        document.getElementById('btnText').addEventListener('click', () => {
            vscode.postMessage({ command: 'compressText' });
        });
        document.getElementById('btnDocument').addEventListener('click', () => {
            vscode.postMessage({ command: 'compressDocument' });
        });
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
