from transformers import pipeline


sentiment_analyzer = pipeline("sentiment-analysis")


print("AI sentiment and summarizer app...")

raw_data = input("Enter your text: ")

def clean_data(raw_data):
    cleaned= raw_data.strip()
    
    if len(cleaned) < 3:
        return None
    
    return cleaned[:500]

processed_data = clean_data(raw_data)
if processed_data is None:
    print("Your data is too short!")
else:
    result = sentiment_analyzer(processed_data)
    label = result[0]['label']
    score = result[0]['score']

    print(f"sentiment: {label}")
    print(f"confidence: {round(score*100, 2)}%")