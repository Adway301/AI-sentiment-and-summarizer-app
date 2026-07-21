from transformers import pipeline


sentiment_analyzer = pipeline("sentiment-analysis")
summarizer = pipeline("text-generation", model="facebook/bart-large-cnn")

print("AI sentiment and summarizer app...")

raw_data = input("Enter your text: ")

def clean_data(raw_data):
    cleaned= raw_data.strip()
    
    if len(cleaned) < 30:
        return None
    
    return cleaned[:2000]

processed_data = clean_data(raw_data)
if processed_data is None:
    print("Your data is too short!")
else:
    # sentiment
    result = sentiment_analyzer(processed_data)
    label = result[0]['label']
    score = result[0]['score']

    print(f"sentiment: {label}")
    print(f"confidence: {round(score*100, 2)}%")

    # summarization
    summary_result = summarizer(processed_data, max_length=50, min_length=15, do_sample=False)
    summary= summary_result[0]['summary_text']
    print(f"\nSummary: {summary}")