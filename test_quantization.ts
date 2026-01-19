import { pipeline } from '@xenova/transformers';

async function testQuantization() {
  console.log('Testing FP32 (quantized: false)...');
  const extractorFP32 = await pipeline('feature-extraction', 'isuruwijesiri/all-MiniLM-L6-v2-code-search-512', {
    quantized: false
  });
  
  const code = "def add(a, b): return a + b";
  const outputFP32 = await extractorFP32(code, { pooling: 'mean', normalize: true });
  const embeddingFP32 = Array.from(outputFP32.data).slice(0, 5);
  
  console.log('FP32 first 5 values:', embeddingFP32);
  console.log('FP32 model size info:', extractorFP32.model);
  
  console.log('\n---\n');
  
  console.log('Testing Quantized (quantized: true)...');
  const extractorQuant = await pipeline('feature-extraction', 'isuruwijesiri/all-MiniLM-L6-v2-code-search-512', {
    quantized: true
  });
  
  const outputQuant = await extractorQuant(code, { pooling: 'mean', normalize: true });
  const embeddingQuant = Array.from(outputQuant.data).slice(0, 5);
  
  console.log('Quantized first 5 values:', embeddingQuant);
  console.log('Quantized model size info:', extractorQuant.model);
  
  console.log('\n---\n');
  console.log('Are embeddings identical?', JSON.stringify(embeddingFP32) === JSON.stringify(embeddingQuant));
  console.log('Max difference:', Math.max(...embeddingFP32.map((v, i) => Math.abs(v - embeddingQuant[i]))));
}

testQuantization().catch(console.error);
