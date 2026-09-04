"""
Trains a simple ML classifier that predicts the "wisdom category"
of a transcript (Marriage, Career, Money, Relationships, General).
This demonstrates real ML training + inference, not just an LLM call.
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import joblib

# Training data: (text, label)
# In a real system this would grow as users add more interviews.
training_data = [
    ("When I married your grandfather everyone said we were too young but a good marriage is about listening and forgiving each other", "Marriage"),
    ("You should marry someone who respects you not someone with money", "Marriage"),
    ("Choose a partner who listens to you and stays through hard times", "Marriage"),
    ("When I lost my job at 45 everyone panicked more than me start over quietly", "Career"),
    ("Work hard but do not let your job define who you are as a person", "Career"),
    ("Take the next opportunity even if it looks smaller than before", "Career"),
    ("Never spend more than you earn and always save a little every month", "Money"),
    ("Money is not everything but you should still plan your finances wisely", "Money"),
    ("Do not lend money you cannot afford to lose to anyone", "Money"),
    ("Family always comes first even when you disagree with them", "Relationships"),
    ("Forgive your siblings quickly because life is too short for grudges", "Relationships"),
    ("A true friend stays with you when things are difficult not just when things are easy", "Relationships"),
    ("Always be honest and work hard no matter what you choose in life", "General"),
    ("Wake up early and be disciplined because habits shape your future", "General"),
]

texts = [t for t, label in training_data]
labels = [label for t, label in training_data]

model = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english")),
    ("clf", MultinomialNB()),
])

model.fit(texts, labels)
joblib.dump(model, "wisdom_classifier.pkl")
print("Model trained and saved as wisdom_classifier.pkl")

# Quick self-test
test_texts = [
    "Choose a spouse who forgives you and listens",
    "Save money every month for your future",
]
for t in test_texts:
    pred = model.predict([t])[0]
    print(f"'{t[:40]}...' -> {pred}")