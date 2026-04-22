from transformers import pipeline
from transformers import logging as tf_logging
import re


# ====================================================
# Emotion color palette (Plutchik Wheel)
# ====================================================
EMOTION_COLORS = {
    'anger':    '#FF0000',  # Red - aggression
    'disgust':  '#8B4513',  # Brown - repulsion
    'fear':     '#4B0082',  # Purple - anxiety
    'joy':      '#FFD700',  # Gold/Yellow - happiness
    'neutral':  '#808080',  # Grey - neutrality
    'sadness':  '#0000FF',  # Blue - sorrow
    'surprise': '#FFA500',  # Orange - astonishment
}

MIN_TEXT_LENGTH = 20
PROB_THRESHOLD = 0.08

# Load model once
classifier = pipeline(
    'text-classification',
    model='./model',
    top_k=None,  # return all 7 emotions
)

def is_meaningful(text, tokenizer):
    """Check if text contains enough recognizable words."""
    
    MIN_WORD_RATIO = 0.5
    
    words = re.findall(r'[a-zA-Z]{2,}', text)
    if len(words) == 0:
        return False

    known = sum(1 for w in words if w.lower() in tokenizer.get_vocab())
    return (known / len(words)) >= MIN_WORD_RATIO
    
def analyze_emotions(text=None):
    """Analyze text emotions and return filtered list with colors.

    Parameters
    ----------
    text : str or None
        Input text. If None, reads from relevant_text.txt.

    Returns
    -------
    list[dict]
        Each dict: {'emotion': str, 'probability': float, 'color': str}
        Sorted by probability descending, filtered by PROB_THRESHOLD.
    """
    
    if text is None:
        with open('relevant_text.txt', 'r', encoding='utf-8') as f:
            text = f.read()

    if len(text.strip()) < MIN_TEXT_LENGTH:
        raise ValueError(
            f'Text is too short ({len(text.strip())} chars).'
            f'Minimum required: {MIN_TEXT_LENGTH}.'
        )
    
    Raise_or_neutral = 0
    if(Raise_or_neutral):
        if not is_meaningful(text, classifier.tokenizer):
            raise ValueError("It's not an English text!")        
    else:
        if not is_meaningful(text, classifier.tokenizer):
            return [{"emotion": "neutral", "probability": 1.0, "color": EMOTION_COLORS["neutral"]}]
    
    # top_k=None returns all labels sorted by score
    raw_results = classifier(text, truncation=True, max_length=512)[0]

    results = []
    for item in raw_results:
        if item['score'] >= PROB_THRESHOLD:
            results.append({
                'emotion':     item['label'],
                'probability': round(item['score'], 3),
                'color':       EMOTION_COLORS[item['label']],
            })

    results.sort(key=lambda x: x['probability'], reverse=True)
    return results

##################################################################

if __name__ == "__main__":

    # From file
    emotions = analyze_emotions()

    #emotions = analyze_emotions('sd,ghjukjwdfghbjkqasvbj ilqwejhmzfi zqwedhfim')
    
    for e in emotions:
        print(f"  {e['emotion']:<10s}  {e['probability']:.3f}  {e['color']}")
    
    print(f'\n Script finished')
    