import * as fs from 'fs';
import { AutoTokenizer } from '@xenova/transformers';

interface DataItem {
  reply: string;
  description: string;
  [key: string]: any;
}

async function filterData(
  inputFile: string = 'flattened_data.jsonl',
  outputFile: string = 'flattened_filtered_data.jsonl',
  modelName: string = 'Xenova/all-MiniLM-L6-v2',
  maxTokens: number = 512
): Promise<void> {
  console.log('Loading tokenizer...');
  const tokenizer = await AutoTokenizer.from_pretrained(modelName);
  
  console.log(`Reading input file: ${inputFile}`);
  const fileContent = fs.readFileSync(inputFile, 'utf-8');
  const lines = fileContent.split('\n').filter(line => line.trim());
  
  console.log(`Total records: ${lines.length}`);
  
  const filteredData: DataItem[] = [];
  let totalTokenCount = 0;
  let maxTokensFound = 0;
  let minTokensFound = Infinity;
  
  for (let i = 0; i < lines.length; i++) {
    const item: DataItem = JSON.parse(lines[i]);
    const reply = item.reply || '';
    
    // Tokenize the reply field
    const encoded = await tokenizer(reply);
    const tokenCount = encoded.input_ids.size;
    
    totalTokenCount += tokenCount;
    maxTokensFound = Math.max(maxTokensFound, tokenCount);
    minTokensFound = Math.min(minTokensFound, tokenCount);
    
    // Keep only items with token count below the threshold
    if (tokenCount < maxTokens) {
      filteredData.push(item);
    }
    
    // Progress indicator
    if ((i + 1) % 50 === 0) {
      console.log(`Processed ${i + 1}/${lines.length} records...`);
    }
  }
  
  console.log('\n========================================');
  console.log('FILTERING SUMMARY');
  console.log('========================================');
  console.log(`Total records processed: ${lines.length}`);
  console.log(`Records kept (< ${maxTokens} tokens): ${filteredData.length}`);
  console.log(`Records filtered out: ${lines.length - filteredData.length}`);
  console.log(`Average tokens per reply: ${(totalTokenCount / lines.length).toFixed(2)}`);
  console.log(`Min tokens found: ${minTokensFound}`);
  console.log(`Max tokens found: ${maxTokensFound}`);
  console.log('========================================\n');
  
  // Write filtered data to output file
  console.log(`Writing filtered data to: ${outputFile}`);
  const outputLines = filteredData.map(item => JSON.stringify(item)).join('\n');
  fs.writeFileSync(outputFile, outputLines + '\n', 'utf-8');
  
  console.log(`✅ Done! Filtered dataset saved to ${outputFile}`);
}

// Main execution
if (import.meta.url === `file://${process.argv[1]}`) {
  (async () => {
    const inputFile = process.argv[2] || 'merged_data.jsonl';
    const outputFile = process.argv[3] || 'filtered_data.jsonl';
    
    await filterData(inputFile, outputFile);
  })().catch(console.error);
}

export { filterData };
