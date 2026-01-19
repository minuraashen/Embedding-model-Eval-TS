from transformers import AutoTokenizer

def tokenize_xml(xml_file, model_name='sentence-transformers/all-MiniLM-L6-v2'):
    """
    Tokenizes an XML file using a HuggingFace sentence-transformers model tokenizer.
    Shows detailed information about tokens, splits, and token count.

    Args:
        xml_file (str): Path to the XML file to be tokenized.
        model_name (str): Name of the sentence-transformers model to use.
    """
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Read the XML file
    with open(xml_file, 'r', encoding='utf-8') as file:
        xml_content = file.read()
    
    print("=" * 80)
    print(f"MODEL: {model_name}")
    print("=" * 80)
    
    # Tokenize without truncation to see full token count
    tokens_full = tokenizer(xml_content, truncation=False, add_special_tokens=True)
    
    # Get token IDs
    token_ids = tokens_full['input_ids']
    
    # Convert token IDs back to tokens to see how text is split
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    
    # Get the original text for each token
    print("\n" + "=" * 80)
    print("TOKENIZATION DETAILS")
    print("=" * 80)
    
    print(f"\n📊 TOTAL TOKEN COUNT: {len(tokens)}")
    print(f"📊 Token IDs count: {len(token_ids)}")
    print(f"📊 Model max length: {tokenizer.model_max_length}")
    
    # Check if truncation would occur
    if len(tokens) > tokenizer.model_max_length:
        print(f"\n⚠️  WARNING: Content exceeds model max length!")
        print(f"   Tokens will be truncated from {len(tokens)} to {tokenizer.model_max_length}")
    
    print("\n" + "-" * 80)
    print("TOKEN LIST (showing how text is split into subwords)")
    print("-" * 80)
    
    # Display tokens with their IDs in a formatted way
    print(f"\n{'Index':<8} {'Token ID':<12} {'Token':<30}")
    print("-" * 50)
    
    for i, (token_id, token) in enumerate(zip(token_ids, tokens)):
        # Highlight special tokens
        if token in ['[CLS]', '[SEP]', '[PAD]', '[UNK]', '<s>', '</s>', '<pad>']:
            print(f"{i:<8} {token_id:<12} {token:<30} [SPECIAL]")
        else:
            print(f"{i:<8} {token_id:<12} {token:<30}")
    
    print("\n" + "=" * 80)
    print("TOKEN ANALYSIS")
    print("=" * 80)
    
    # Analyze subword splitting
    subword_tokens = [t for t in tokens if t.startswith('##') or t.startswith('▁')]
    regular_tokens = [t for t in tokens if not t.startswith('##') and not t.startswith('▁') 
                      and t not in ['[CLS]', '[SEP]', '[PAD]', '[UNK]', '<s>', '</s>', '<pad>']]
    special_tokens = [t for t in tokens if t in ['[CLS]', '[SEP]', '[PAD]', '[UNK]', '<s>', '</s>', '<pad>']]
    
    print(f"\n📈 Token Statistics:")
    print(f"   - Total tokens: {len(tokens)}")
    print(f"   - Regular tokens: {len(regular_tokens)}")
    print(f"   - Subword tokens (## prefix): {len(subword_tokens)}")
    print(f"   - Special tokens: {len(special_tokens)}")
    
    # Show unique tokens
    unique_tokens = set(tokens)
    print(f"   - Unique tokens: {len(unique_tokens)}")
    
    # Decode back to text to show reconstruction
    print("\n" + "-" * 80)
    print("DECODED TEXT (reconstructed from tokens)")
    print("-" * 80)
    decoded_text = tokenizer.decode(token_ids, skip_special_tokens=True)
    print(f"\n{decoded_text[:]}...")  # Show chars
    
    return {
        'tokens': tokens,
        'token_ids': token_ids,
        'token_count': len(tokens),
        'unique_token_count': len(unique_tokens),
        'model_max_length': tokenizer.model_max_length
    }


if __name__ == "__main__":
    result = tokenize_xml('example_xml.txt')
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total tokens: {result['token_count']}")
    print(f"Unique tokens: {result['unique_token_count']}")
    print(f"Model max length: {result['model_max_length']}")