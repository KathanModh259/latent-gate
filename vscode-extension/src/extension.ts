import * as vscode from 'vscode';
import { LatentGateClient } from './client';
import { DashboardProvider } from './webview/dashboard';
import { ToolsProvider } from './webview/tools';
import { StatusBarManager } from './statusBar';
import { registerCommands } from './commands';

let client: LatentGateClient;
let statusBar: StatusBarManager;

export async function activate(context: vscode.ExtensionContext) {
    console.log('LatentGate: activating...');

    client = new LatentGateClient(context);
    statusBar = new StatusBarManager(context);
    context.subscriptions.push(statusBar);

    const dashboardProvider = new DashboardProvider(context.extensionUri, client);
    const toolsProvider = new ToolsProvider(context.extensionUri, client);

    context.subscriptions.push(
        vscode.window.registerWebviewViewProvider('latentGate.dashboard', dashboardProvider),
        vscode.window.registerWebviewViewProvider('latentGate.tools', toolsProvider),
    );

    registerCommands(context, client, statusBar, dashboardProvider);

    // Auto-setup: check Ollama, pull models if needed
    statusBar.setLoading(true);
    const status = await client.setupAuto();
    statusBar.setLoading(false);
    statusBar.setHealthy(status.healthy);

    if (!status.healthy) {
        const action = await vscode.window.showWarningMessage(
            `LatentGate: ${status.message}`,
            'Open Settings',
            'Dismiss'
        );
        if (action === 'Open Settings') {
            vscode.commands.executeCommand('latentGate.openSettings');
        }
    } else {
        vscode.window.showInformationMessage('LatentGate: Ready!');
    }

    console.log('LatentGate: activated');
}

export function deactivate() {
    client?.dispose();
    statusBar?.dispose();
}
