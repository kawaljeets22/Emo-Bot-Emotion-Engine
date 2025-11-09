# fsm_engine.py
from transitions import Machine
from graphviz import Digraph
import math

class EmotionFSM:
    states = ["Neutral", "Happy", "Sad", "Angry", "Surprised", "Fearful", "Curious"]

    transitions_def = [
        # generic transitions (can be improved with intensity thresholds)
        {"trigger": "to_happy", "source": "*", "dest": "Happy"},
        {"trigger": "to_sad", "source": "*", "dest": "Sad"},
        {"trigger": "to_angry", "source": "*", "dest": "Angry"},
        {"trigger": "to_neutral", "source": "*", "dest": "Neutral"},
        {"trigger": "to_surprised", "source": "*", "dest": "Surprised"},
        {"trigger": "to_fearful", "source": "*", "dest": "Fearful"},
        {"trigger": "to_curious", "source": "*", "dest": "Curious"},
    ]

    def __init__(self):
        self.machine = Machine(model=self, states=EmotionFSM.states, transitions=EmotionFSM.transitions_def, initial="Neutral")
        # internal memory / momentum
        self.mood_score = {s: 0.0 for s in EmotionFSM.states}

    @property
    def state(self):
        return self.machine.state

    def get_graphviz_source(self):
        dot = Digraph(format='png')
        dot.attr(rankdir='LR')
        for s in EmotionFSM.states:
            if s == self.state:
                dot.node(s, style="filled", color="lightblue")
            else:
                dot.node(s)
        # show possible transitions (complete graph here)
        for s in EmotionFSM.states:
            for t in EmotionFSM.states:
                dot.edge(s, t, arrowhead='vee')
        return dot.source

    def update_from_nlp(self, emotion_scores: dict, sentiment_scores: dict):
        """
        Decide transition based on top emotion and sentiment.
        emotion_scores: dict label->score (e.g., {'joy':0.7, 'sadness':0.1,...})
        sentiment_scores: dict label->score (e.g., {'POSITIVE':0.8})
        """
        # Normalize and pick top emotion
        if not emotion_scores:
            top_emo = None
        else:
            # pick label with max score
            top_emo = max(emotion_scores.items(), key=lambda x: x[1])[0]

        # likewise for sentiment
        if sentiment_scores:
            top_sent = max(sentiment_scores.items(), key=lambda x: x[1])[0]
        else:
            top_sent = None

        # map emotion label variants to FSM states
        mapping = {
            "joy": "Happy", "happy": "Happy", "happiness": "Happy",
            "sadness": "Sad", "sad": "Sad",
            "anger": "Angry", "angry": "Angry",
            "surprise": "Surprised", "surprised": "Surprised",
            "fear": "Fearful", "fearful": "Fearful",
            "neutral": "Neutral",
            "disgust": "Angry", "trust": "Curious", "anticipation": "Curious",
            "curiosity": "Curious", "curious": "Curious"
        }

        chosen_state = None

        if top_emo:
            t = top_emo.lower()
            chosen_state = mapping.get(t, None)

        if not chosen_state and top_sent:
            if top_sent.upper() == "POSITIVE":
                chosen_state = "Happy"
            elif top_sent.upper() == "NEGATIVE":
                chosen_state = "Sad"
            else:
                chosen_state = "Neutral"

        # update mood momentum
        if chosen_state:
            self._apply_transition(chosen_state)
        else:
            # default to neutral
            self.to_neutral()
        return self.state

    def _apply_transition(self, target_state):
        # call corresponding trigger
        trigger = {
            "Happy": self.to_happy,
            "Sad": self.to_sad,
            "Angry": self.to_angry,
            "Neutral": self.to_neutral,
            "Surprised": self.to_surprised,
            "Fearful": self.to_fearful,
            "Curious": self.to_curious
        }.get(target_state, self.to_neutral)
        trigger()
