import * as vscode from 'vscode';
import { ChildProcessWithoutNullStreams, spawn } from 'child_process';
import * as path from 'path';
import * as http from 'http';

export interface CompressionResult {
    answer: string;
    compactPrompt: string;
    tokensEstimated: number;
    tokensSaved?: number;
    compressionRatio?: string;
    payload: any;
    timing: {
        localMs: number;
        remoteMs: number;
        totalMs: number;
    };
}

export interface SessionStats {
    totalFrames: number;
    apiCalls: number;
    skipped: number;
    reductionRatio: string;
    skipRate: string;
}

export class LatentGateClient implements vscode.Disposable {
    private pythonPath: string;
    private outputChannel: vscode.OutputChannel;
    private _onDidChangeStats = new vscode.EventEmitter<SessionStats>();
    public onDidChangeStats = this._onDidChangeStats.event;
    private worker?: ChildProcessWithoutNullStreams;
    private workerBuffer = '';
    private nextRequestId = 1;
    private pending = new Map<number, {
        resolve: (value: any) => void;
        reject: (reason?: any) => void;
        timer: NodeJS.Timeout;
    }>();

    private stats: SessionStats = {
        totalFrames: 0,
        apiCalls: 0,
        skipped: 0,
        reductionRatio: '1.00x',
        skipRate: '0.0%',
    };

    private ollamaReady = false;
    private modelsReady = false;

    constructor(private context: vscode.ExtensionContext) {
        this.outputChannel = vscode.window.createOutputChannel('LatentGate');
        this.pythonPath = this.getPythonPath();
    }

    private getPythonPath(): string {
        const config = vscode.workspace.getConfiguration('latentGate');
        const customPath = config.get<string>('pythonPath');
        if (customPath) { return customPath; }
        return process.platform === 'win32' ? 'python' : 'python3';
    }

    private getConfig() {
        const config = vscode.workspace.getConfiguration('latentGate');
        return {
            ollamaBaseUrl: config.get<string>('ollamaBaseUrl', 'http://localhost:11434'),
            visionModel: config.get<string>('visionModel', 'llava:7b'),
            predictorModel: config.get<string>('predictorModel', 'phi3:mini'),
            remoteProvider: config.get<string>('remoteProvider', 'ollama'),
            remoteModel: config.get<string>('remoteModel', 'phi3:mini'),
            remoteApiKey: config.get<string>('remoteApiKey', ''),
            selectiveDecoding: config.get<boolean>('selectiveDecoding', true),
            similarityThreshold: config.get<number>('similarityThreshold', 0.85),
            useEmbeddings: config.get<boolean>('useEmbeddings', true),
            enableCaching: config.get<boolean>('enableCaching', true),
            logLevel: config.get<string>('logLevel', 'INFO'),
            maxImageDimension: config.get<number>('maxImageDimension', 1280),
            maxConcurrentRequests: config.get<number>('maxConcurrentRequests', 3),
        };
    }

    async checkHealth(): Promise<boolean> {
        return new Promise((resolve) => {
            try {
                const config = this.getConfig();
                const url = new URL(`${config.ollamaBaseUrl}/api/tags`);
                const req = http.get(url.href, { timeout: 3000 }, (res) => {
                    this.ollamaReady = res.statusCode === 200;
                    resolve(res.statusCode === 200);
                    res.resume();
                });
                req.on('error', () => {
                    this.ollamaReady = false;
                    resolve(false);
                });
                req.on('timeout', () => {
                    req.destroy();
                    this.ollamaReady = false;
                    resolve(false);
                });
            } catch {
                this.ollamaReady = false;
                resolve(false);
            }
        });
    }

    async ensureModels(): Promise<boolean> {
        if (this.modelsReady) { return true; }

        try {
            const config = this.getConfig();
            const models = await this.getOllamaModels();
            if (!models) { return false; }

            const needed = [config.predictorModel, config.visionModel];
            const missing = needed.filter(m => !models.some((name: string) => name.startsWith(m)));

            if (missing.length === 0) {
                this.modelsReady = true;
                return true;
            }

            for (const model of missing) {
                this.outputChannel.appendLine(`Pulling model: ${model}...`);
                await this.pullModel(model);
            }

            this.modelsReady = true;
            return true;
        } catch {
            return false;
        }
    }

    private async getOllamaModels(): Promise<string[]> {
        return new Promise((resolve) => {
            try {
                const config = this.getConfig();
                const url = new URL(`${config.ollamaBaseUrl}/api/tags`);
                const req = http.get(url.href, { timeout: 5000 }, (res) => {
                    let data = '';
                    res.on('data', (chunk: Buffer) => { data += chunk.toString(); });
                    res.on('end', () => {
                        try {
                            const parsed = JSON.parse(data);
                            resolve((parsed.models || []).map((m: any) => m.name));
                        } catch {
                            resolve([]);
                        }
                    });
                });
                req.on('error', () => resolve([]));
                req.on('timeout', () => { req.destroy(); resolve([]); });
            } catch {
                resolve([]);
            }
        });
    }

    private async pullModel(model: string): Promise<void> {
        return new Promise((resolve, reject) => {
            const config = this.getConfig();
            const proc = spawn('ollama', ['pull', model], {
                env: process.env,
                timeout: 600000,
            });

            let stderr = '';
            proc.stderr.on('data', (data: Buffer) => {
                const line = data.toString().trim();
                if (line) {
                    this.outputChannel.appendLine(`[pull] ${line}`);
                }
                stderr += data.toString();
            });

            proc.on('close', (code) => {
                if (code === 0) {
                    this.outputChannel.appendLine(`Model ${model} pulled successfully`);
                    resolve();
                } else {
                    reject(new Error(`Failed to pull ${model}: ${stderr}`));
                }
            });

            proc.on('error', (err) => {
                reject(err);
            });
        });
    }

    async setupAuto(): Promise<{ healthy: boolean; message: string }> {
        // Step 1: Check Ollama
        const healthy = await this.checkHealth();
        if (!healthy) {
            return {
                healthy: false,
                message: 'Ollama not running. Install from ollama.com and run "ollama serve".',
            };
        }

        // Step 2: Ensure models
        try {
            await this.ensureModels();
            return {
                healthy: true,
                message: 'Ready! Models loaded.',
            };
        } catch (err: any) {
            return {
                healthy: false,
                message: `Models not ready: ${err.message}`,
            };
        }
    }

    async compressImage(imagePath: string, question: string = 'Describe this image'): Promise<CompressionResult> {
        return this.sendWorker('compress_image', { imagePath, question });
    }

    async compressText(text: string, mode: string = 'auto'): Promise<CompressionResult> {
        return this.sendWorker('compress_text', { text, mode });
    }

    async compressDocument(text: string, question: string): Promise<CompressionResult> {
        return this.sendWorker('compress_document', { text, question });
    }

    async getSessionStats(): Promise<SessionStats> {
        return this.stats;
    }

    private ensureWorker(): ChildProcessWithoutNullStreams {
        if (this.worker && !this.worker.killed) {
            return this.worker;
        }

        const extPath = this.context.extensionPath;
        this.worker = spawn(this.pythonPath, ['-m', 'latent_gate.vscode_worker'], {
            cwd: extPath,
            env: {
                ...process.env,
                PYTHONIOENCODING: 'utf-8',
                PYTHONPATH: [extPath, process.env.PYTHONPATH].filter(Boolean).join(path.delimiter),
            },
        });

        this.worker.stdout.on('data', (data: Buffer) => {
            this.workerBuffer += data.toString();
            let newline = this.workerBuffer.indexOf('\n');
            while (newline >= 0) {
                const line = this.workerBuffer.slice(0, newline).trim();
                this.workerBuffer = this.workerBuffer.slice(newline + 1);
                if (line) {
                    this.handleWorkerLine(line);
                }
                newline = this.workerBuffer.indexOf('\n');
            }
        });

        this.worker.stderr.on('data', (data: Buffer) => {
            this.outputChannel.appendLine(data.toString());
        });

        this.worker.on('close', () => {
            this.rejectPending('LatentGate worker stopped');
            this.worker = undefined;
        });

        this.worker.on('error', () => {
            this.rejectPending('Python worker failed to start. Install Python 3.10+ and add it to PATH.');
            this.worker = undefined;
        });

        return this.worker;
    }

    private handleWorkerLine(line: string) {
        let response: any;
        try {
            response = JSON.parse(line);
        } catch {
            this.outputChannel.appendLine(line);
            return;
        }

        const id = response.id;
        const pending = this.pending.get(id);
        if (!pending) { return; }

        clearTimeout(pending.timer);
        this.pending.delete(id);

        if (response.ok) {
            pending.resolve(this.toCamelCase(response.result));
        } else {
            this.outputChannel.appendLine(response.traceback || response.error || 'Worker error');
            pending.reject(new Error(response.error || 'Worker error'));
        }
    }

    private rejectPending(message: string) {
        for (const [id, pending] of this.pending) {
            clearTimeout(pending.timer);
            pending.reject(new Error(message));
            this.pending.delete(id);
        }
    }

    private async sendWorker(command: string, payload: Record<string, any>): Promise<any> {
        return new Promise((resolve, reject) => {
            const id = this.nextRequestId++;
            const timer = setTimeout(() => {
                this.pending.delete(id);
                reject(new Error('LatentGate request timed out'));
            }, 300000);

            this.pending.set(id, { resolve, reject, timer });

            try {
                const worker = this.ensureWorker();
                worker.stdin.write(JSON.stringify({
                    id,
                    command,
                    config: this.getConfig(),
                    ...payload,
                }) + '\n');
            } catch (err: any) {
                clearTimeout(timer);
                this.pending.delete(id);
                reject(err);
            }
        });
    }

    updateStats(result: CompressionResult) {
        this.stats.totalFrames++;
        this.stats.apiCalls++;
        this._onDidChangeStats.fire(this.stats);
    }

    private toCamelCase(obj: any): any {
        if (Array.isArray(obj)) {
            return obj.map(item => this.toCamelCase(item));
        }
        if (obj !== null && typeof obj === 'object' && !(obj instanceof Date)) {
            return Object.fromEntries(
                Object.entries(obj).map(([key, value]) => [
                    key.replace(/_([a-z])/g, (_, c) => c.toUpperCase()),
                    this.toCamelCase(value),
                ])
            );
        }
        return obj;
    }

    dispose() {
        if (this.worker && !this.worker.killed) {
            this.worker.stdin.write(JSON.stringify({
                id: this.nextRequestId++,
                command: 'shutdown',
                config: this.getConfig(),
            }) + '\n');
            this.worker.kill();
        }
        this.rejectPending('LatentGate client disposed');
        this.outputChannel.dispose();
        this._onDidChangeStats.dispose();
    }
}
