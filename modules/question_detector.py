import re
import time

INTERROGATIVE = (
    "what", "why", "how", "when", "where", "who",
    "which", "whose", "whom", "can", "could",
    "would", "will", "should", "is", "are",
    "do", "does", "did", "have", "has",
    "explain", "describe", "tell me"
)

class QuestionTracker:

    def __init__(self, pause_seconds=1.2):
        self.pause_seconds = pause_seconds
        self.reset()

    def reset(self):
        self.text = ""
        self.last_update = 0
        self.last_question = ""

    def update(self, text):
        if not text:
            return

        text = re.sub(r"\s+", " ", text).strip()

        if text != self.text:
            self.text = text
            self.last_update = time.monotonic()

    def poll(self):
        if not self.text or not self.last_update:
            return None

        if time.monotonic() - self.last_update < self.pause_seconds:
            return None

        text = self.text.strip()

        if len(text.split()) < 2:
            return None

        starters = (
            "what", "why", "how", "when", "where", "who",
            "which", "whose", "whom", "can", "could",
            "would", "will", "should", "is", "are",
            "do", "does", "did", "have", "has",
            "explain", "describe", "tell me"
        )

        lowered = text.lower()

        positions = [
            match.start()
            for starter in starters
            for match in re.finditer(
                r"\b" + re.escape(starter) + r"\b",
                lowered
            )
        ]

        if not positions:
            return None

        question = text[min(positions):].strip(" ,.!?")

        if len(question.split()) < 3:
            return None

        if question.lower() == self.last_question.lower():
            return None

        last_word = re.sub(r"[^\w]", "", question.split()[-1]).lower()

        incomplete = {
            "a", "an", "the", "of", "to", "for", "with",
            "and", "or", "but", "in", "on", "at", "by",
            "from", "about", "into", "what", "how", "which",
            "who", "when", "where", "can", "could"
        }

        if last_word in incomplete:
            return None

        self.last_question = question
        return question