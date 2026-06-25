import * as vscode from 'vscode';
import { spawn } from 'child_process';
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
        const config = this.getConfig();
        const pythonScript = `
import json, sys
sys.path.insert(0, r'${path.resolve(__dirname, '../../..')}')
from latent_gate import LatentGatePipeline, PipelineConfig

config = PipelineConfig(
    vision_model="${config.visionModel}",
    predictor_model="${config.predictorModel}",
    remote_provider="${config.remoteProvider}",
    remote_model="${config.remoteModel}",
    remote_api_key="${config.remoteApiKey}",
    selective_decoding=${config.selectiveDecoding},
    similarity_threshold=${config.similarityThreshold},
    use_embeddings=${config.useEmbeddings},
    enable_caching=${config.enableCaching},
    log_level="WARNING",
)

with LatentGatePipeline(config, preload=False) as pipeline:
    result = pipeline.query(r"${imagePath.replace(/\\/g, '\\\\')}", """${question}""")
    print(json.dumps({
        "answer": result["answer"],
        "compactPrompt": result["compact_prompt"],
        "tokensEstimated": result["tokens_estimated"],
        "payload": result["payload"],
        "timing": result["timing"],
    }))
`;
        return this.runPython(pythonScript);
    }

    async compressText(text: string, mode: string = 'auto'): Promise<CompressionResult> {
        const config = this.getConfig();
        const escaped = text.replace(/\\/g, '\\\\').replace(/"/g, '\\"').replace(/\n/g, '\\n').replace(/\r/g, '');
        const pythonScript = `
import json, sys
sys.path.insert(0, r'${path.resolve(__dirname, '../../..')}')
from latent_gate.text_processor import TextProcessor
from latent_gate.config import PipelineConfig

config = PipelineConfig(
    predictor_model="${config.predictorModel}",
    ollama_base_url="${config.ollamaBaseUrl}",
    log_level="WARNING",
)

tp = TextProcessor(config)
result = tp.compress("""${escaped}""", mode="${mode}")
compact = result.to_compact_prompt()
print(json.dumps({
    "answer": compact,
    "compactPrompt": compact,
    "tokensEstimated": result.compressed_token_count,
    "originalTokens": result.original_token_count,
    "compressionRatio": f"{result.compression_ratio:.1f}x" if result.compression_ratio > 0 else "1.0x",
    "tokensSaved": result.original_token_count - result.compressed_token_count,
    "payload": {},
    "timing": {
        "localMs": result.processing_time_ms,
        "remoteMs": 0,
        "totalMs": result.processing_time_ms,
    },
}))
`;
        return this.runPython(pythonScript);
    }

    async compressDocument(text: string, question: string): Promise<CompressionResult> {
        const config = this.getConfig();
        const escaped = text.replace(/\\/g, '\\\\').replace(/"/g, '\\"').replace(/\n/g, '\\n').replace(/\r/g, '');
        const pythonScript = `
import json, sys
sys.path.insert(0, r'${path.resolve(__dirname, '../../..')}')
from latent_gate import LatentGatePipeline, PipelineConfig

config = PipelineConfig(
    predictor_model="${config.predictorModel}",
    remote_provider="${config.remoteProvider}",
    remote_model="${config.remoteModel}",
    remote_api_key="${config.remoteApiKey}",
    log_level="WARNING",
)

with LatentGatePipeline(config, preload=False) as pipeline:
    result = pipeline.query_documents(["""${escaped}"""], """${question}""")
    print(json.dumps({
        "answer": result["answer"],
        "compactPrompt": result["compact_prompt"],
        "tokensEstimated": result["tokens_estimated"],
        "originalTokens": result.get("original_tokens", 0),
        "compressionRatio": result.get("compression_ratio", "1.0x"),
        "tokensSaved": result.get("tokens_saved", 0),
        "payload": result["payload"],
        "timing": result["timing"],
    }))
`;
        return this.runPython(pythonScript);
    }

    async getSessionStats(): Promise<SessionStats> {
        return this.stats;
    }

    private async runPython(script: string): Promise<any> {
        return new Promise((resolve, reject) => {
            const proc = spawn(this.pythonPath, ['-c', script], {
                env: { ...process.env, PYTHONIOENCODING: 'utf-8' },
                timeout: 300000,
            });

            let stdout = '';
            let stderr = '';

            proc.stdout.on('data', (data: Buffer) => {
                stdout += data.toString();
            });

            proc.stderr.on('data', (data: Buffer) => {
                stderr += data.toString();
                this.outputChannel.appendLine(data.toString());
            });

            proc.on('close', (code) => {
                if (code === null) {
                    reject(new Error(stderr || 'Process timed out'));
                    return;
                }
                if (code !== 0) {
                    reject(new Error(stderr || stdout || `Exit code ${code}`));
                    return;
                }

                try {
                    const lines = stdout.trim().split('\n');
                    const jsonLine = lines.find(l => l.trim().startsWith('{'));
                    if (jsonLine) {
                        const raw = JSON.parse(jsonLine);
                        resolve(this.toCamelCase(raw));
                    } else {
                        reject(new Error(stdout.slice(0, 500) || 'No output'));
                    }
                } catch (e) {
                    reject(new Error(stdout.slice(0, 500) || 'Parse error'));
                }
            });

            proc.on('error', (err) => {
                reject(new Error(`Python not found. Install Python 3.10+ and add to PATH.`));
            });
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
        this.outputChannel.dispose();
        this._onDidChangeStats.dispose();
    }
}
