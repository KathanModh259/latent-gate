import * as vscode from 'vscode';
import { LatentGateClient, CompressionResult } from './client';
import { StatusBarManager } from './statusBar';
import { DashboardProvider } from './webview/dashboard';

export function registerCommands(
    context: vscode.ExtensionContext,
    client: LatentGateClient,
    statusBar: StatusBarManager,
    dashboard: DashboardProvider
) {
    // Compress Image
    context.subscriptions.push(
        vscode.commands.registerCommand('latentGate.compressImage', async (uri?: vscode.Uri) => {
            let imagePath: string;

            if (uri) {
                imagePath = uri.fsPath;
            } else {
                // Open file picker
                const uris = await vscode.window.showOpenDialog({
                    canSelectMany: false,
                    filters: {
                        'Images': ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'webp'],
                    },
                    title: 'Select Image to Compress',
                });
                if (!uris || uris.length === 0) { return; }
                imagePath = uris[0].fsPath;
            }

            const question = await vscode.window.showInputBox({
                prompt: 'What do you want to know about this image?',
                value: 'Describe this image',
            });
            if (question === undefined) { return; }

            await vscode.window.withProgress({
                location: vscode.ProgressLocation.Notification,
                title: 'LatentGate',
                cancellable: false,
            }, async (progress) => {
                progress.report({ message: 'Compressing image locally...' });

                try {
                    const result = await client.compressImage(imagePath, question);
                    client.updateStats(result);

                    // Show result in new document
                    const doc = await vscode.workspace.openTextDocument({
                        content: formatResult(result, 'Image Compression'),
                        language: 'markdown',
                    });
                    await vscode.window.showTextDocument(doc);

                    statusBar.updateSavings(result.tokensSaved || 0);
                    dashboard.refresh();

                    vscode.window.showInformationMessage(
                        `LatentGate: Compressed to ~${result.tokensEstimated} tokens (${result.timing.totalMs.toFixed(0)}ms)`
                    );
                } catch (err: any) {
                    vscode.window.showErrorMessage(`LatentGate: ${err.message}`);
                }
            });
        })
    );

    // Compress Selection
    context.subscriptions.push(
        vscode.commands.registerCommand('latentGate.compressSelection', async () => {
            const editor = vscode.window.activeTextEditor;
            if (!editor) {
                vscode.window.showWarningMessage('No active editor');
                return;
            }

            const selection = editor.document.getText(editor.selection);
            if (!selection) {
                vscode.window.showWarningMessage('No text selected');
                return;
            }

            await vscode.window.withProgress({
                location: vscode.ProgressLocation.Notification,
                title: 'LatentGate',
                cancellable: false,
            }, async (progress) => {
                progress.report({ message: 'Compressing text locally...' });

                try {
                    const result = await client.compressText(selection);
                    client.updateStats(result);

                    const doc = await vscode.workspace.openTextDocument({
                        content: formatResult(result, 'Text Compression'),
                        language: 'markdown',
                    });
                    await vscode.window.showTextDocument(doc);

                    statusBar.updateSavings(result.tokensSaved || 0);
                    dashboard.refresh();

                    vscode.window.showInformationMessage(
                        `LatentGate: ${result.compressionRatio || '1.0x'} compression (${result.timing.totalMs.toFixed(0)}ms)`
                    );
                } catch (err: any) {
                    vscode.window.showErrorMessage(`LatentGate: ${err.message}`);
                }
            });
        })
    );

    // Compress Document
    context.subscriptions.push(
        vscode.commands.registerCommand('latentGate.compressDocument', async () => {
            const editor = vscode.window.activeTextEditor;
            if (!editor) {
                vscode.window.showWarningMessage('No active editor');
                return;
            }

            const text = editor.document.getText();
            if (!text) {
                vscode.window.showWarningMessage('Document is empty');
                return;
            }

            const question = await vscode.window.showInputBox({
                prompt: 'What do you want to know about this document?',
                value: 'Summarize the key points',
            });
            if (question === undefined) { return; }

            await vscode.window.withProgress({
                location: vscode.ProgressLocation.Notification,
                title: 'LatentGate',
                cancellable: false,
            }, async (progress) => {
                progress.report({ message: 'Compressing document locally...' });

                try {
                    const result = await client.compressDocument(text, question);
                    client.updateStats(result);

                    const doc = await vscode.workspace.openTextDocument({
                        content: formatResult(result, 'Document Compression'),
                        language: 'markdown',
                    });
                    await vscode.window.showTextDocument(doc);

                    statusBar.updateSavings(result.tokensSaved || 0);
                    dashboard.refresh();

                    vscode.window.showInformationMessage(
                        `LatentGate: ${result.compressionRatio || '1.0x'} compression (${result.timing.totalMs.toFixed(0)}ms)`
                    );
                } catch (err: any) {
                    vscode.window.showErrorMessage(`LatentGate: ${err.message}`);
                }
            });
        })
    );

    // Show Dashboard
    context.subscriptions.push(
        vscode.commands.registerCommand('latentGate.showDashboard', () => {
            dashboard.show();
        })
    );

    // Show Stats
    context.subscriptions.push(
        vscode.commands.registerCommand('latentGate.showStats', async () => {
            const stats = await client.getSessionStats();
            vscode.window.showInformationMessage(
                `LatentGate Stats: ${stats.apiCalls} API calls, ` +
                `${stats.skipped} skipped, ${stats.skipRate} skip rate`
            );
        })
    );

    // Open Settings
    context.subscriptions.push(
        vscode.commands.registerCommand('latentGate.openSettings', () => {
            vscode.commands.executeCommand('workbench.action.openSettings', 'latentGate');
        })
    );

    // Check Health
    context.subscriptions.push(
        vscode.commands.registerCommand('latentGate.checkHealth', async () => {
            const healthy = await client.checkHealth();
            statusBar.setHealthy(healthy);

            if (healthy) {
                vscode.window.showInformationMessage('LatentGate: Ollama is running');
            } else {
                vscode.window.showErrorMessage(
                    'LatentGate: Ollama not detected. Start with "ollama serve".',
                    'Open Settings'
                ).then(action => {
                    if (action === 'Open Settings') {
                        vscode.commands.executeCommand('latentGate.openSettings');
                    }
                });
            }
        })
    );

    // Setup MCP
    context.subscriptions.push(
        vscode.commands.registerCommand('latentGate.setupMcp', async () => {
            const config = vscode.workspace.getConfiguration('copilot');
            const mcpConfig = {
                'latent-gate': {
                    command: 'python',
                    args: ['-m', 'latent_gate.mcp_server'],
                },
            };

            // Try to configure for Copilot Chat
            try {
                const currentConfig = config.get<any>('chat.mcpServers', {});
                const updated = { ...currentConfig, ...mcpConfig };
                await config.update('chat.mcpServers', updated, vscode.ConfigurationTarget.Global);
                vscode.window.showInformationMessage(
                    'LatentGate: MCP server configured for Copilot Chat'
                );
            } catch {
                // Fallback: show manual instructions
                const configJson = JSON.stringify(mcpConfig, null, 2);
                const doc = await vscode.workspace.openTextDocument({
                    content: `# LatentGate MCP Configuration\n\nAdd this to your VSCode settings.json:\n\n\`\`\`json\n${configJson}\n\`\`\`\n\nOr add to Copilot Chat MCP settings.`,
                    language: 'markdown',
                });
                await vscode.window.showTextDocument(doc);
            }
        })
    );
}

function formatResult(result: CompressionResult, title: string): string {
    const lines = [
        `# ${title}`,
        '',
        '## Answer',
        result.answer,
        '',
        '## Statistics',
        `- **Tokens Estimated:** ~${result.tokensEstimated}`,
    ];

    if (result.tokensSaved) {
        lines.push(`- **Tokens Saved:** ${result.tokensSaved}`);
    }
    if (result.compressionRatio) {
        lines.push(`- **Compression Ratio:** ${result.compressionRatio}`);
    }

    lines.push(
        `- **Local Processing:** ${result.timing.localMs.toFixed(0)}ms`,
        `- **Remote API:** ${result.timing.remoteMs.toFixed(0)}ms`,
        `- **Total Time:** ${result.timing.totalMs.toFixed(0)}ms`,
        '',
        '## Compact Prompt',
        '```',
        result.compactPrompt,
        '```',
    );

    return lines.join('\n');
}
