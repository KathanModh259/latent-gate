import * as vscode from 'vscode';

export class StatusBarManager implements vscode.Disposable {
    private statusBarItem: vscode.StatusBarItem;
    private totalSavings: number = 0;
    private isHealthy: boolean = false;

    constructor(context: vscode.ExtensionContext) {
        this.statusBarItem = vscode.window.createStatusBarItem(
            vscode.StatusBarAlignment.Right,
            100
        );
        this.statusBarItem.command = 'latentGate.showStats';
        this.updateDisplay();
        this.statusBarItem.show();
        context.subscriptions.push(this);
    }

    setHealthy(healthy: boolean) {
        this.isHealthy = healthy;
        this.updateDisplay();
    }

    updateSavings(tokens: number) {
        this.totalSavings += tokens;
        this.updateDisplay();
    }

    resetSavings() {
        this.totalSavings = 0;
        this.updateDisplay();
    }

    private updateDisplay() {
        if (!this.isHealthy) {
            this.statusBarItem.text = '$(warning) LatentGate: Offline';
            this.statusBarItem.tooltip = 'Ollama not detected. Click to configure.';
            this.statusBarItem.backgroundColor = new vscode.ThemeColor('statusBarItem.warningBackground');
            return;
        }

        if (this.totalSavings > 0) {
            const saved = this.totalSavings >= 1000
                ? `${(this.totalSavings / 1000).toFixed(1)}k`
                : this.totalSavings.toString();
            this.statusBarItem.text = `$(zap) LatentGate: ${saved} tokens saved`;
            this.statusBarItem.tooltip = `Click to view session stats\nTotal tokens saved: ${this.totalSavings}`;
            this.statusBarItem.backgroundColor = undefined;
        } else {
            this.statusBarItem.text = '$(rocket) LatentGate';
            this.statusBarItem.tooltip = 'LatentGate active. Click for options.';
            this.statusBarItem.backgroundColor = undefined;
        }
    }

    dispose() {
        this.statusBarItem.dispose();
    }
}
