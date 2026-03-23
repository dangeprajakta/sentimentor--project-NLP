import argparse
from transformers import pipeline

# --- FUNCTION 1: Setup the CLI ---

def setup_cli():
    parser = argparse.ArgumentParser(description="Sentimentator - CLI Sentiment Analysis")
    parser.add_argument("--text", type=str, nargs='+', help="Text(s) to analyze (can pass multiple)")
    parser.add_argument("--model", type=str, 
                        default="distilbert-base-uncased-finetuned-sst-2-english", 
                        help="Hugging Face model name")
    return parser.parse_args()

# --- FUNCTION 2: The Logic ---

def analyze_sentiment(texts, model_name):
    try:
        # 'pipeline' loads the model and the tokenizer automatically[cite: 109, 114].
        classifier = pipeline("sentiment-analysis", model=model_name)
        results = classifier(texts)
        return results
    except Exception as e:
        print(f"Error: Could not load model. {e}")
        return None

# --- FUNCTION 3: The Output Formatter ---

def display_result(texts, results):
    if results:
        for text, result in zip(texts, results):
            print(f"\nInput: {text}")
            print(f"Sentiment: {result['label']} (Confidence: {result['score']:.4f})")

# --- THE MAIN ENTRY POINT ---
# This follows the 'Main Flow Pseudocode' in your LLD[cite: 44, 45].
def main():
    args = setup_cli()
    
    if args.text:
        # Step-by-step execution
        results = analyze_sentiment(args.text, args.model)
        display_result(args.text, results)
    else:
        # Guidance if the user forgets the --text flag
        print("Usage: python sentimentator.py --text 'Your message here' (can pass multiple texts)")

if __name__ == "__main__":
    main()