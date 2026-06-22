import * as vscode from 'vscode';
import { spawn } from 'child_process';
import * as path from 'path';

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

    constructor(private context: vscode.ExtensionContext) {
        this.outputChannel = vscode.window.createOutputChannel('LatentGate');
        this.pythonPath = this.getPythonPath();
    }

    private getPythonPath(): string {
        const config = vscode.workspace.getConfiguration('latentGate');
        const customPath = config.get<string>('pythonPath');
        if (customPath) { return customPath; }

        // Try common Python locations
        const python = process.platform === 'win32' ? 'python' : 'python3';
        return python;
    }

    private getConfig() {
        const config = vscode.workspace.getConfiguration('latentGate');
        return {
            ollamaBaseUrl: config.get<string>('ollamaBaseUrl', 'http://localhost:11434'),
            visionModel: config.get<string>('visionModel', 'llava:7b'),
            predictorModel: config.get<string>('predictorModel', 'llama3:8b'),
            remoteProvider: config.get<string>('remoteProvider', 'openai'),
            remoteModel: config.get<string>('remoteModel', 'gpt-4o-mini'),
            remoteApiKey: config.get<string>('remoteApiKey', ''),
            selectiveDecoding: config.get<boolean>('selectiveDecoding', true),
            similarityThreshold: config.get<number>('similarityThreshold', 0.85),
            useEmbeddings: config.get<boolean>('useEmbeddings', true),
            enableCaching: config.get<boolean>('enableCaching', true),
            logLevel: config.get<string>('logLevel', 'INFO'),
        };
    }

    async checkHealth(): Promise<boolean> {
        try {
            const config = this.getConfig();
            const response = await fetch(`${config.ollamaBaseUrl}/api/tags`);
            return response.ok;
        } catch {
            return false;
        }
    }

    async compressImage(imagePath: string, question: string = 'Describe this image'): Promise<CompressionResult> {
        const pythonScript = `
import json, sys
sys.path.insert(0, r'${path.resolve(__dirname, '../../..')}')
from latent_gate import LatentGatePipeline, PipelineConfig

config = PipelineConfig(
    vision_model="${this.getConfig().visionModel}",
    predictor_model="${this.getConfig().predictorModel}",
    remote_provider="${this.getConfig().remoteProvider}",
    remote_model="${this.getConfig().remoteModel}",
    selective_decoding=${this.getConfig().selectiveDecoding},
    similarity_threshold=${this.getConfig().similarityThreshold},
    use_embeddings=${this.getConfig().useEmbeddings},
    enable_caching=${this.getConfig().enableCaching},
    log_level="WARNING",
)

with LatentGatePipeline(config) as pipeline:
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
        const pythonScript = `
import json, sys
sys.path.insert(0, r'${path.resolve(__dirname, '../../..')}')
from latent_gate import LatentGatePipeline, PipelineConfig

config = PipelineConfig(
    predictor_model="${this.getConfig().predictorModel}",
    remote_provider="${this.getConfig().remoteProvider}",
    remote_model="${this.getConfig().remoteModel}",
    log_level="WARNING",
)

with LatentGatePipeline(config) as pipeline:
    result = pipeline.query_text("""${text.replace(/"/g, '\\"').replace(/\n/g, '\\n')}""", mode="${mode}")
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

    async compressDocument(text: string, question: string): Promise<CompressionResult> {
        const pythonScript = `
import json, sys
sys.path.insert(0, r'${path.resolve(__dirname, '../../..')}')
from latent_gate import LatentGatePipeline, PipelineConfig

config = PipelineConfig(
    predictor_model="${this.getConfig().predictorModel}",
    remote_provider="${this.getConfig().remoteProvider}",
    remote_model="${this.getConfig().remoteModel}",
    log_level="WARNING",
)

with LatentGatePipeline(config) as pipeline:
    result = pipeline.query_documents(["""${text.replace(/"/g, '\\"').replace(/\n/g, '\\n')}"""], """${question}""")
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
                timeout: 120000,
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
                if (code !== 0) {
                    reject(new Error(`Python process exited with code ${code}: ${stderr}`));
                    return;
                }

                try {
                    // Find JSON in output (skip any warning/log lines)
                    const lines = stdout.trim().split('\n');
                    const jsonLine = lines.find(l => l.trim().startsWith('{'));
                    if (jsonLine) {
                        resolve(JSON.parse(jsonLine));
                    } else {
                        reject(new Error(`No JSON in output: ${stdout}`));
                    }
                } catch (e) {
                    reject(new Error(`Failed to parse output: ${stdout}`));
                }
            });

            proc.on('error', (err) => {
                reject(err);
            });
        });
    }

    updateStats(result: CompressionResult) {
        this.stats.totalFrames++;
        this.stats.apiCalls++;
        this._onDidChangeStats.fire(this.stats);
    }

    dispose() {
        this.outputChannel.dispose();
        this._onDidChangeStats.dispose();
    }
}
