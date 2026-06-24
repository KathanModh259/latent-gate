import * as vscode from 'vscode';
import { LatentGateClient } from '../client';

export class ToolsProvider implements vscode.WebviewViewProvider {
    public static readonly viewType = 'latentGate.tools';
    private _view?: vscode.WebviewView;
    private client: LatentGateClient;

    constructor(extensionUri: vscode.Uri, client: LatentGateClient) {
        this.client = client;
    }

    public resolveWebviewView(webviewView: vscode.WebviewView) {
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
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: var(--vscode-font-family);
            font-size: var(--vscode-font-size);
            color: var(--vscode-foreground);
            background: var(--vscode-sideBar-background);
            padding: 12px;
        }

        .section-title {
            font-size: 11px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: var(--vscode-sideBarSectionHeader-foreground);
            margin-bottom: 8px;
        }

        .tool-list {
            display: flex;
            flex-direction: column;
            gap: 6px;
            margin-bottom: 16px;
        }
        .tool-item {
            padding: 10px;
            border-radius: 6px;
            background: var(--vscode-input-background);
            border-left: 3px solid var(--vscode-terminal-ansiCyan);
        }
        .tool-item .tool-name {
            font-size: 12px;
            font-weight: 600;
            font-family: var(--vscode-editor-font-family);
            color: var(--vscode-terminal-ansiCyan);
            margin-bottom: 3px;
        }
        .tool-item .tool-desc {
            font-size: 11px;
            color: var(--vscode-descriptionForeground);
            line-height: 1.4;
        }

        .shortcuts {
            margin-bottom: 16px;
        }
        .shortcut-row {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 6px 0;
            border-bottom: 1px solid var(--vscode-widget-border);
            font-size: 12px;
        }
        .shortcut-row:last-child { border-bottom: none; }
        kbd {
            padding: 2px 6px;
            border-radius: 3px;
            background: var(--vscode-keybindingLabel-background);
            color: var(--vscode-keybindingLabel-foreground);
            border: 1px solid var(--vscode-keybindingLabel-border);
            font-family: var(--vscode-editor-font-family);
            font-size: 10px;
        }

        .config-section {
            margin-bottom: 16px;
        }
        .config-code {
            font-family: var(--vscode-editor-font-family);
            font-size: 11px;
            padding: 10px;
            border-radius: 6px;
            background: var(--vscode-editor-background);
            overflow-x: auto;
            white-space: pre;
            line-height: 1.5;
        }

        .btn-list {
            display: flex;
            flex-direction: column;
            gap: 6px;
        }
        .btn {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 9px 12px;
            border: none;
            border-radius: 6px;
            font-size: 12px;
            font-family: var(--vscode-font-family);
            cursor: pointer;
            transition: opacity 0.15s;
            text-align: left;
            width: 100%;
        }
        .btn:hover { opacity: 0.85; }
        .btn-primary {
            background: var(--vscode-button-background);
            color: var(--vscode-button-foreground);
        }
        .btn-secondary {
            background: var(--vscode-button-secondaryBackground);
            color: var(--vscode-button-secondaryForeground);
        }
        .btn .icon { font-size: 14px; flex-shrink: 0; }
    </style>
</head>
<body>
    <div class="section-title">MCP Tools</div>
    <div class="tool-list">
        <div class="tool-item">
            <div class="tool-name">compress_image</div>
            <div class="tool-desc">Image ~1,200 tokens -> ~150 tokens (86% savings)</div>
        </div>
        <div class="tool-item">
            <div class="tool-name">compress_text</div>
            <div class="tool-desc">Long prompts ~500 tokens -> ~100 tokens (85% savings)</div>
        </div>
        <div class="tool-item">
            <div class="tool-name">compress_conversation</div>
            <div class="tool-desc">Chat history -> compact summary (86% savings)</div>
        </div>
        <div class="tool-item">
            <div class="tool-name">compress_documents</div>
            <div class="tool-desc">RAG docs -> key facts only (85% savings)</div>
        </div>
        <div class="tool-item">
            <div class="tool-name">get_stats</div>
            <div class="tool-desc">Session token savings statistics</div>
        </div>
    </div>

    <div class="section-title">Shortcuts</div>
    <div class="shortcuts">
        <div class="shortcut-row">
            <span>Compress Selection</span>
            <span><kbd>Ctrl</kbd>+<kbd>Shift</kbd>+<kbd>Alt</kbd>+<kbd>C</kbd></span>
        </div>
        <div class="shortcut-row">
            <span>Show Dashboard</span>
            <span><kbd>Ctrl</kbd>+<kbd>Shift</kbd>+<kbd>Alt</kbd>+<kbd>D</kbd></span>
        </div>
    </div>

    <div class="section-title">MCP Config</div>
    <div class="config-section">
        <div class="config-code">{
  "latent-gate": {
    "command": "python",
    "args": ["-m", "latent_gate.mcp_server"]
  }
}</div>
    </div>

    <div class="btn-list">
        <button class="btn btn-primary" id="btnConfigure">
            <span class="icon">&#9881;</span> Configure MCP
        </button>
        <button class="btn btn-secondary" id="btnSettings">
            <span class="icon">&#9881;</span> Settings
        </button>
        <button class="btn btn-secondary" id="btnDocs">
            <span class="icon">&#128214;</span> Documentation
        </button>
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
