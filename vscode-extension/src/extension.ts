import * as vscode from 'vscode';
import { LatentGateClient } from './client';
import { DashboardProvider } from './webview/dashboard';
import { ToolsProvider } from './webview/tools';
import { StatusBarManager } from './statusBar';
import { registerCommands } from './commands';

let client: LatentGateClient;
let statusBar: StatusBarManager;

export function activate(context: vscode.ExtensionContext) {
    console.log('LatentGate extension activating...');

    // Initialize client
    client = new LatentGateClient(context);

    // Status bar
    statusBar = new StatusBarManager(context);
    context.subscriptions.push(statusBar);

    // Webview providers
    const dashboardProvider = new DashboardProvider(context.extensionUri, client);
    const toolsProvider = new ToolsProvider(context.extensionUri, client);

    context.subscriptions.push(
        vscode.window.registerWebviewViewProvider('latentGate.dashboard', dashboardProvider),
        vscode.window.registerWebviewViewProvider('latentGate.tools', toolsProvider),
    );

    // Register commands
    registerCommands(context, client, statusBar, dashboardProvider);

    // Auto-configure MCP if first run
    const hasConfigured = context.globalState.get('latentGate.configured');
    if (!hasConfigured) {
        vscode.commands.executeCommand('latentGate.setupMcp');
        context.globalState.update('latentGate.configured', true);
    }

    // Start health check
    client.checkHealth().then(healthy => {
        statusBar.setHealthy(healthy);
        if (!healthy) {
            vscode.window.showWarningMessage(
                'LatentGate: Ollama not detected. Start with "ollama serve" or configure remote provider.',
                'Open Settings'
            ).then(action => {
                if (action === 'Open Settings') {
                    vscode.commands.executeCommand('latentGate.openSettings');
                }
            });
        }
    });

    console.log('LatentGate extension activated');
}

export function deactivate() {
    client?.dispose();
    statusBar?.dispose();
}
