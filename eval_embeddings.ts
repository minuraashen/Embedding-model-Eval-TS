import * as fs from 'fs';
import { pipeline } from '@xenova/transformers';
import * as os from 'os';

console.log(`Total Logical Cores: ${os.cpus().length}`);

// -----------------------------
// CONFIG
// -----------------------------
const DATA_PATH = "dataset/flattened_filtered_dataset.jsonl";
const BATCH_SIZE = 32;
const RECALL_K_VALUES = [1, 3, 5, 10];
// const CUSTOM_MAX_SEQ_LENGTH = 512;

interface ModelConfig {
  type: string;
  path: string;
  quantized?: boolean;
}

const MODELS: Record<string, ModelConfig | string> = {
  //"minilm_base_fp32": "sentence-transformers/all-MiniLM-L6-v2",
  //"code_search_fp32": "isuruwijesiri/all-MiniLM-L6-v2-code-search-512",
  "code_search_quantized": {
    type: "onnx",
    path: "isuruwijesiri/all-MiniLM-L6-v2-code-search-512",
    quantized: true
  }
};

interface DataItem {
  reply: string;
  description: string;
}

interface EvaluationResults {
  model: string;
  xml_embedding_time_sec: number;
  query_embedding_time_sec: number;
  avg_xml_latency_ms: number;
  avg_query_latency_ms: number;
  xml_peak_memory_mb?: number;
  query_peak_memory_mb?: number;
  xml_cpu_percent?: number;
  query_cpu_percent?: number;
  [key: string]: number | string | undefined;
}

// -----------------------------
// LOAD DATASET
// -----------------------------
function loadDataset(path: string): [string[], string[]] {
  const xmlTexts: string[] = [];
  const descriptions: string[] = [];

  const fileContent = fs.readFileSync(path, 'utf-8');
  const lines = fileContent.split('\n').filter(line => line.trim());

  for (const line of lines) {
    const item: DataItem = JSON.parse(line);
    xmlTexts.push(item.reply);
    descriptions.push(item.description);
  }

  return [xmlTexts, descriptions];
}

// -----------------------------
// METRICS
// -----------------------------
function recallAtK(similarityMatrix: number[][], groundTruth: number[], k: number): number {
  let hits = 0;
  for (let i = 0; i < similarityMatrix.length; i++) {
    const ranked = similarityMatrix[i]
      .map((score, idx) => ({ score, idx }))
      .sort((a, b) => b.score - a.score)
      .slice(0, k)
      .map(item => item.idx);
    
    if (ranked.includes(groundTruth[i])) {
      hits++;
    }
  }
  return hits / similarityMatrix.length;
}

function meanReciprocalRank(similarityMatrix: number[][], groundTruth: number[]): number {
  let rrSum = 0.0;
  for (let i = 0; i < similarityMatrix.length; i++) {
    const ranked = similarityMatrix[i]
      .map((score, idx) => ({ score, idx }))
      .sort((a, b) => b.score - a.score)
      .map(item => item.idx);
    
    const rank = ranked.indexOf(groundTruth[i]) + 1;
    rrSum += 1.0 / rank;
  }
  return rrSum / similarityMatrix.length;
}

// -----------------------------
// EVALUATION PIPELINE
// -----------------------------
interface EncodingResult {
  embeddings: number[][];
  peakMemoryMb: number;
  wallTimeMs: number;
  cpuPercent: number;
}

async function encodeTexts(
  extractor: any,
  texts: string[],
  batchSize: number,
  showProgress: boolean = true
): Promise<EncodingResult> {
  const allEmbeddings: number[][] = [];
  let peakRss = 0;
  
  const startTime = process.hrtime.bigint();
  const startCpu = process.cpuUsage();
  
  for (let i = 0; i < texts.length; i += batchSize) {
    const batch = texts.slice(i, Math.min(i + batchSize, texts.length));
    
    if (showProgress) {
      console.log(`Encoding batch ${Math.floor(i / batchSize) + 1}/${Math.ceil(texts.length / batchSize)}`);
    }

    for (const text of batch) {
      const output = await extractor(text, {
        pooling: 'mean',
        normalize: true
      });
      const embedding = Array.from(output.data) as number[];
      allEmbeddings.push(embedding);
    }
    
    // Track memory after each batch
    const mem = process.memoryUsage();
    peakRss = Math.max(peakRss, mem.rss);
  }
  
  const endTime = process.hrtime.bigint();
  const endCpu = process.cpuUsage(startCpu);
  
  const wallTimeMs = Number(endTime - startTime) / 1e6;
  const cpuTimeMs = (endCpu.user + endCpu.system) / 1000;
  const cpuPercent = (cpuTimeMs / wallTimeMs) * 100;
  const peakMemoryMb = peakRss / (1024 * 1024);
  
  if (showProgress) {
    console.log(`  Peak memory: ${peakMemoryMb.toFixed(2)} MB, CPU usage: ${cpuPercent.toFixed(1)}%`);
  }
  
  return {
    embeddings: allEmbeddings,
    peakMemoryMb,
    wallTimeMs,
    cpuPercent
  };
}

function matmul(a: number[][], b: number[][]): number[][] {
  const result: number[][] = [];
  for (let i = 0; i < a.length; i++) {
    result[i] = [];
    for (let j = 0; j < b[0].length; j++) {
      let sum = 0;
      for (let k = 0; k < a[0].length; k++) {
        sum += a[i][k] * b[k][j];
      }
      result[i][j] = sum;
    }
  }
  return result;
}

function transpose(matrix: number[][]): number[][] {
  if (matrix.length === 0) return [];
  const result: number[][] = [];
  for (let j = 0; j < matrix[0].length; j++) {
    result[j] = [];
    for (let i = 0; i < matrix.length; i++) {
      result[j][i] = matrix[i][j];
    }
  }
  return result;
}

async function evaluateModel(
  modelName: string,
  modelConfig: ModelConfig | string,
  xmlTexts: string[],
  descriptions: string[]
): Promise<EvaluationResults> {
  console.log(`\nEvaluating model: ${modelName}`);
  
  let modelPath: string;
  let quantized: boolean = false;
  
  if (typeof modelConfig === 'string') {
    modelPath = modelConfig;
  } else {
    modelPath = modelConfig.path;
    quantized = modelConfig.quantized ?? false;
  }
  
  console.log(`Loading model: ${modelPath}`);
  console.log(`Quantized: ${quantized}`);
  
  // Load the model using @xenova/transformers
  const extractor = await pipeline('feature-extraction', modelPath, {
    quantized: quantized
  });
  
  console.log('Model loaded successfully!');
  
  // Encode XML corpus
  console.log('\nEncoding XML corpus...');
  const xmlResult = await encodeTexts(extractor, xmlTexts, BATCH_SIZE);
  const xmlTime = xmlResult.wallTimeMs / 1000;
  
  // Encode queries
  console.log('\nEncoding queries...');
  const queryResult = await encodeTexts(extractor, descriptions, BATCH_SIZE);
  const queryTime = queryResult.wallTimeMs / 1000;
  
  // Similarity matrix
  const similarityMatrix = matmul(queryResult.embeddings, transpose(xmlResult.embeddings));
  
  // Ground truth: ith description ↔ ith XML
  const groundTruth = Array.from({ length: descriptions.length }, (_, i) => i);
  
  // Metrics
  const results: EvaluationResults = {
    model: modelName,
    xml_embedding_time_sec: xmlTime,
    query_embedding_time_sec: queryTime,
    avg_xml_latency_ms: (xmlTime / xmlTexts.length) * 1000,
    avg_query_latency_ms: (queryTime / descriptions.length) * 1000,
    xml_peak_memory_mb: xmlResult.peakMemoryMb,
    query_peak_memory_mb: queryResult.peakMemoryMb,
    xml_cpu_percent: xmlResult.cpuPercent,
    query_cpu_percent: queryResult.cpuPercent,
  };
  
  for (const k of RECALL_K_VALUES) {
    results[`Recall@${k}`] = recallAtK(similarityMatrix, groundTruth, k);
  }
  
  results['MRR'] = meanReciprocalRank(similarityMatrix, groundTruth);
  
  return results;
}

// -----------------------------
// MAIN
// -----------------------------
async function main() {
  const [xmlTexts, descriptions] = loadDataset(DATA_PATH);
  
  console.log(`Loaded ${xmlTexts.length} samples`);
  
  const allResults: EvaluationResults[] = [];
  
  for (const [modelName, modelConfig] of Object.entries(MODELS)) {
    const results = await evaluateModel(modelName, modelConfig, xmlTexts, descriptions);
    allResults.push(results);
  }
  
  // Print Results
  console.log('\n===== FINAL RESULTS =====');
  for (const res of allResults) {
    console.log('\nModel:', res.model);
    console.log(`MRR: ${res['MRR'].toFixed(4)}`);
    for (const k of RECALL_K_VALUES) {
      console.log(`Recall@${k}: ${res[`Recall@${k}`].toFixed(4)}`);
    }
    console.log(`Avg XML latency (ms): ${res.avg_xml_latency_ms.toFixed(2)}`);
    console.log(`Avg Query latency (ms): ${res.avg_query_latency_ms.toFixed(2)}`);
    console.log(`XML Peak Memory: ${res.xml_peak_memory_mb?.toFixed(2)} MB`);
    console.log(`Query Peak Memory: ${res.query_peak_memory_mb?.toFixed(2)} MB`);
    console.log(`XML CPU Usage: ${res.xml_cpu_percent?.toFixed(1)}%`);
    console.log(`Query CPU Usage: ${res.query_cpu_percent?.toFixed(1)}%`);
  }
}

main().catch(console.error);
