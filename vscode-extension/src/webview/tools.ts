import * as vscode from 'vscode';
import { LatentGateClient } from '../client';

export class ToolsProvider implements vscode.WebviewViewProvider {
    public static readonly viewType = 'latentGate.tools';
    private _view?: vscode.WebviewView;
    private client: LatentGateClient;

    constructor(extensionUri: vscode.Uri, client: LatentGateClient) {
        this.client = client;
    }

    public resolveWebviewView(
        webviewView: vscode.WebviewView,
    ) {
        this._view = webviewView;

        webviewView.webview.options = {
            enableScripts: true,
        };

        webviewView.webview.html = this._getHtmlForWebview(webviewView.webview);

        webviewView.webview.onDidReceiveMessage(async (message) => {
            switch (message.command) {
                case 'configureMcp':
                    vscode.commands.executeCommand('latentGate.setupMcp');
                    break;
                case 'openSettings':
                    vscode.commands.executeCommand('latentGate.openSettings');
                    break;
                case 'viewDocs':
                    vscode.env.openExternal(vscode.Uri.parse('https://github.com/KathanModh259/latent-gate'));
                    break;
            }
        });
    }

    private _getHtmlForWebview(webview: vscode.Webview): string {
        const nonce = getNonce();

        return `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LatentGate Tools</title>
    <style>
        :root {
            --container-padding: 20px;
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

        .tool-card {
            padding: 12px;
            border-radius: 4px;
            background: var(--vscode-input-background);
            margin-bottom: 8px;
        }

        .tool-name {
            font-weight: 600;
            margin-bottom: 4px;
            color: var(--vscode-terminal-ansiCyan);
        }

        .tool-desc {
            font-size: 12px;
            color: var(--vscode-descriptionForeground);
        }

        .actions {
            display: flex;
            flex-direction: column;
            gap: 8px;
            margin-top: 16px;
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

        .action-btn.secondary {
            background: var(--vscode-button-secondaryBackground);
            color: var(--vscode-button-secondaryForeground);
        }

        .action-btn.secondary:hover {
            background: var(--vscode-button-secondaryHoverBackground);
        }

        .section {
            margin-bottom: 20px;
        }

        .config-code {
            font-family: var(--vscode-editor-font-family);
            font-size: 11px;
            padding: 8px;
            border-radius: 4px;
            background: var(--vscode-editor-background);
            overflow-x: auto;
            white-space: pre;
            margin-top: 8px;
        }

        .keyboard-shortcut {
            display: inline-block;
            padding: 2px 6px;
            border-radius: 3px;
            background: var(--vscode-keybindingLabel-background);
            color: var(--vscode-keybindingLabel-foreground);
            border: 1px solid var(--vscode-keybindingLabel-border);
            font-size: 11px;
            font-family: var(--vscode-editor-font-family);
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="section">
            <h2>MCP Tools</h2>
            <div class="tool-card">
                <div class="tool-name">compress_image</div>
                <div class="tool-desc">Image → ~150 tokens (saves ~86%)</div>
            </div>
            <div class="tool-card">
                <div class="tool-name">compress_text</div>
                <div class="tool-desc">Long prompts → ~100 tokens (saves ~85%)</div>
            </div>
            <div class="tool-card">
                <div class="tool-name">compress_conversation</div>
                <div class="tool-desc">Chat history → summary (saves ~86%)</div>
            </div>
            <div class="tool-card">
                <div class="tool-name">compress_documents</div>
                <div class="tool-desc">RAG docs → key facts (saves ~85%)</div>
            </div>
            <div class="tool-card">
                <div class="tool-name">get_stats</div>
                <div class="tool-desc">Session savings statistics</div>
            </div>
        </div>

        <div class="section">
            <h2>Keyboard Shortcuts</h2>
            <p style="font-size: 12px;">
                Compress Selection: <span class="keyboard-shortcut">Ctrl+Shift+Alt+C</span><br>
                Show Dashboard: <span class="keyboard-shortcut">Ctrl+Shift+Alt+D</span>
            </p>
        </div>

        <div class="section">
            <h2>MCP Configuration</h2>
            <div class="config-code">{
  "latent-gate": {
    "command": "python",
    "args": ["-m", "latent_gate.mcp_server"]
  }
}</div>
        </div>

        <div class="actions">
            <button class="action-btn" id="btnConfigure">
                <span class="icon">$(plug)</span>
                Configure MCP Server
            </button>
            <button class="action-btn secondary" id="btnSettings">
                <span class="icon">$(settings-gear)</span>
                Open Settings
            </button>
            <button class="action-btn secondary" id="btnDocs">
                <span class="icon">$(book)</span>
                View Documentation
            </button>
        </div>
    </div>

    <script nonce="${nonce}">
        const vscode = acquireVsCodeApi();

        document.getElementById('btnConfigure').addEventListener('click', () => {
            vscode.postMessage({ command: 'configureMcp' });
        });
        document.getElementById('btnSettings').addEventListener('click', () => {
            vscode.postMessage({ command: 'openSettings' });
        });
        document.getElementById('btnDocs').addEventListener('click', () => {
            vscode.postMessage({ command: 'viewDocs' });
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
