import * as fs from 'fs';
import { AutoTokenizer } from '@xenova/transformers';

interface TokenizationResult {
  tokens: string[];
  token_ids: number[];
  token_count: number;
  unique_token_count: number;
  model_max_length: number;
}

async function tokenizeXml(
  xmlFile: string,
  modelName: string = 'sentence-transformers/all-MiniLM-L6-v2'
): Promise<TokenizationResult> {
  /**
   * Tokenizes an XML file using a HuggingFace sentence-transformers model tokenizer.
   * Shows detailed information about tokens, splits, and token count.
   */
  
  // Load the tokenizer
  console.log(`Loading tokenizer for: ${modelName}`);
  const tokenizer = await AutoTokenizer.from_pretrained(modelName);
  
  // Read the XML file
  const xmlContent = fs.readFileSync(xmlFile, 'utf-8');
  
  console.log('='.repeat(80));
  console.log(`MODEL: ${modelName}`);
  console.log('='.repeat(80));
  
  // Tokenize the content
  const encoded = await tokenizer(xmlContent);
  const tokenIds = Array.from(encoded.input_ids.data) as number[];
  
  // Decode individual tokens
  const tokens: string[] = [];
  for (const tokenId of tokenIds) {
    const token = tokenizer.decode([tokenId], { skip_special_tokens: false });
    tokens.push(token);
  }
  
  // Get the model's max length
  const modelMaxLength = tokenizer.model_max_length || 512;
  
  console.log('\n' + '='.repeat(80));
  console.log('TOKENIZATION DETAILS');
  console.log('='.repeat(80));
  
  console.log(`\n📊 TOTAL TOKEN COUNT: ${tokens.length}`);
  console.log(`📊 Token IDs count: ${tokenIds.length}`);
  console.log(`📊 Model max length: ${modelMaxLength}`);
  
  // Check if truncation would occur
  if (tokens.length > modelMaxLength) {
    console.log(`\n⚠️  WARNING: Content exceeds model max length!`);
    console.log(`   Tokens will be truncated from ${tokens.length} to ${modelMaxLength}`);
  }
  
  console.log('\n' + '-'.repeat(80));
  console.log('TOKEN LIST (showing how text is split into subwords)');
  console.log('-'.repeat(80));
  
  // Display tokens with their IDs in a formatted way
  console.log(`\n${'Index'.padEnd(8)} ${'Token ID'.padEnd(12)} ${'Token'.padEnd(30)}`);
  console.log('-'.repeat(50));
  
  const specialTokens = ['[CLS]', '[SEP]', '[PAD]', '[UNK]', '<s>', '</s>', '<pad>'];
  
  for (let i = 0; i < tokenIds.length; i++) {
    const tokenId = tokenIds[i];
    const token = tokens[i];
    
    // Highlight special tokens
    if (specialTokens.includes(token)) {
      console.log(`${String(i).padEnd(8)} ${String(tokenId).padEnd(12)} ${token.padEnd(30)} [SPECIAL]`);
    } else {
      console.log(`${String(i).padEnd(8)} ${String(tokenId).padEnd(12)} ${token.padEnd(30)}`);
    }
  }
  
  console.log('\n' + '='.repeat(80));
  console.log('TOKEN ANALYSIS');
  console.log('='.repeat(80));
  
  // Analyze subword splitting
  const subwordTokens = tokens.filter(t => t.startsWith('##') || t.startsWith('▁'));
  const regularTokens = tokens.filter(
    t => !t.startsWith('##') && !t.startsWith('▁') && !specialTokens.includes(t)
  );
  const specialTokensList = tokens.filter(t => specialTokens.includes(t));
  
  console.log(`\n📈 Token Statistics:`);
  console.log(`   - Total tokens: ${tokens.length}`);
  console.log(`   - Regular tokens: ${regularTokens.length}`);
  console.log(`   - Subword tokens (## or ▁ prefix): ${subwordTokens.length}`);
  console.log(`   - Special tokens: ${specialTokensList.length}`);
  
  // Show unique tokens
  const uniqueTokens = new Set(tokens);
  console.log(`   - Unique tokens: ${uniqueTokens.size}`);
  
  // Decode back to text to show reconstruction
  console.log('\n' + '-'.repeat(80));
  console.log('DECODED TEXT (reconstructed from tokens)');
  console.log('-'.repeat(80));
  
  const decodedText = tokenizer.decode(tokenIds, { skip_special_tokens: true });
  console.log(`\n${decodedText}...`);
  
  return {
    tokens,
    token_ids: tokenIds,
    token_count: tokens.length,
    unique_token_count: uniqueTokens.size,
    model_max_length: modelMaxLength
  };
}

// Main execution - check if this file is being run directly
const isMainModule = import.meta.url === `file://${process.argv[1]}`;

if (isMainModule) {
  (async () => {
    const result = await tokenizeXml('example_xml.txt');
    console.log('\n' + '='.repeat(80));
    console.log('SUMMARY');
    console.log('='.repeat(80));
    console.log(`Total tokens: ${result.token_count}`);
    console.log(`Unique tokens: ${result.unique_token_count}`);
    console.log(`Model max length: ${result.model_max_length}`);
  })().catch(console.error);
}

export { tokenizeXml, TokenizationResult };
