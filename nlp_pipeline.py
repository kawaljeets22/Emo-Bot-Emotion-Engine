# nlp_pipeline.py
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import text2emotion as t2e
import nltk

# ensure punkt etc loaded for text2emotion if needed
try:
    nltk.data.find('tokenizers/punkt')
except Exception:
    nltk.download('punkt')

class EmotionSentimentPipeline:
    def __init__(self, model_name="j-hartmann/emotion-english-distilroberta-base"):
        self.model_name = model_name
        self._load_models()

    def _load_models(self):
        # Emotion classifier (transformer) - may be slow on first load
        try:
            self.emotion_pipe = pipeline("text-classification", model=self.model_name, return_all_scores=True)
            self.sentiment_pipe = pipeline("sentiment-analysis")  # generic sentiment
            self.use_transformer = True
        except Exception as e:
            print("Transformer pipelines failed to load; falling back to VADER/Text2Emotion.", e)
            self.use_transformer = False
            self.vader = SentimentIntensityAnalyzer()

    def predict(self, text):
        """
        Return: (emotion_prediction, sentiment_prediction)
        Each is a dict: {label: score, ...}
        """
        text = text or ""
        if self.use_transformer and text.strip():
            try:
                emo_scores = self.emotion_pipe(text)[0]  # list of dicts
                emo = {item['label']: float(item['score']) for item in emo_scores}
            except Exception as e:
                print("Emotion transform failed:", e)
                emo = t2e.get_emotion(text)
            try:
                sent_scores = self.sentiment_pipe(text)
                # pipeline returns list of dicts with label and score
                sent = {sent_scores[0]['label']: float(sent_scores[0]['score'])}
            except Exception as e:
                print("Sentiment transform failed:", e)
                vs = self.vader.polarity_scores(text)
                # convert compound to label
                compound = vs['compound']
                lab = "POSITIVE" if compound >= 0.05 else ("NEGATIVE" if compound <= -0.05 else "NEUTRAL")
                sent = {lab: float(abs(compound))}
            return emo, sent
        else:
            # fallback
            emo = t2e.get_emotion(text)
            vs = self.vader.polarity_scores(text)
            compound = vs['compound']
            lab = "POSITIVE" if compound >= 0.05 else ("NEGATIVE" if compound <= -0.05 else "NEUTRAL")
            sent = {lab: float(abs(compound))}
            return emo, sent
