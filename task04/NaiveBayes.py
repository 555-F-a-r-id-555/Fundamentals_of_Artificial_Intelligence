# алгоритм naive bayes для определения спама
from typing import Set, NamedTuple, List, Tuple, Dict, Iterable
import re
import math
from collections import defaultdict


def tokenize(text: str) -> Set[str]:
    text = text.lower()
    all_words = re.findall("[a-z0-9]+", text)
    return set(all_words)


class Message(NamedTuple):
    text: str
    is_spam: bool


class NaiveBayesClassifier:
    def __init__(self, k: float = 0.5) -> None:
        self.k = k
        self.tokens: Set[str] = set()
        self.token_spam_counts: Dict[str, int] = defaultdict(int)
        self.token_ham_counts: Dict[str, int] = defaultdict(int)
        self.spam_messages = self.ham_messages = 0

    def train(self, messages: Iterable[Message]) -> None:
        for message in messages:
            if message.is_spam:
                self.spam_messages += 1
            else:
                self.ham_messages += 1

            for token in tokenize(message.text):
                self.tokens.add(token)
                if message.is_spam:
                    self.token_spam_counts[token] += 1
                else:
                    self.token_ham_counts[token] += 1

    def _probabilities(self, token: str) -> Tuple[float, float]:
        spam = self.token_spam_counts[token]  # Количество токена в спам-сообщениях
        ham = self.token_ham_counts[token]  # Количество токена в хэм-сообщениях

        p_token_spam = (spam + self.k) / (self.spam_messages + 2 * self.k)  # Вероятность токена при спаме
        p_token_ham = (ham + self.k) / (self.ham_messages + 2 * self.k)  # Вероятность токена при хэм

        return p_token_spam, p_token_ham

    def predict(self, text: str) -> float:
        text_tokens = tokenize(text)
        log_prob_if_spam = log_prob_if_ham = 0

        for token in self.tokens:
            p_token_spam, p_token_ham = self._probabilities(token)

            # Умножаем на вероятность для каждого токена в тексте
            if token in text_tokens:
                log_prob_if_spam += math.log(p_token_spam)
                log_prob_if_ham += math.log(p_token_ham)
            else:
                log_prob_if_spam += math.log(1 - p_token_spam)
                log_prob_if_ham += math.log(1 - p_token_ham)

        # Применяем обратный логарифм для окончательных вероятностей
        prob_if_spam = math.exp(log_prob_if_spam) * (self.spam_messages / (self.spam_messages + self.ham_messages))
        prob_if_ham = math.exp(log_prob_if_ham) * (self.ham_messages / (self.spam_messages + self.ham_messages))

        return prob_if_spam / (prob_if_spam + prob_if_ham)

    def predict_verbose(self, text: str) -> float:
        text_tokens = tokenize(text)
        log_prob_if_spam = log_prob_if_ham = 0

        print(f"\nАнализ текста: '{text}'")
        print(f"{'Token':<15}{'P(token|spam)':<20}{'P(token|ham)':<20}")

        for token in self.tokens:
            p_token_spam, p_token_ham = self._probabilities(token)

            if token in text_tokens:
                log_prob_if_spam += math.log(p_token_spam)
                log_prob_if_ham += math.log(p_token_ham)
                mark = "+"
            else:
                log_prob_if_spam += math.log(1 - p_token_spam)
                log_prob_if_ham += math.log(1 - p_token_ham)
                mark = "-"

            print(f"{mark} {token:<13}{p_token_spam:<20.5f}{p_token_ham:<20.5f}")

        prob_if_spam = math.exp(log_prob_if_spam) * (self.spam_messages / (self.spam_messages + self.ham_messages))
        prob_if_ham = math.exp(log_prob_if_ham) * (self.ham_messages / (self.spam_messages + self.ham_messages))

        final = prob_if_spam / (prob_if_spam + prob_if_ham)
        print(f"\nFinal Spam Probability: {final:.4f}")
        return final


# -----------------------------------------------------------------v1------------------------------------------------------------------
# Пример обучения
messages = [
    Message("spam messages", is_spam=True),
    Message("ham messages", is_spam=False),
    Message("Hello ham", is_spam=False),
]

model = NaiveBayesClassifier(k=0.5)
model.train(messages)

# Проверяем состояние модели
print("model.tokens = ", model.tokens)
print("model.spam_messages = ", model.spam_messages)
print("model.ham_messages = ", model.ham_messages)
print("model.token_spam_counts = ", model.token_spam_counts.items())
print("model.token_ham_counts = ", model.token_ham_counts.items())
#
# Предсказание
text = "hello spam"
print("model.predict(text) = ", model.predict(text))

# Рассчитываем вероятности для спама и не-спама
# model.spam_messages =  1
probs_if_spam = [
    (1 + 0.5) / (1 + 2 * 0.5),  # spam 1+
    1 - (0 + 0.5) / (1 + 2 * 0.5),  # ham 0-
    1 - (1 + 0.5) / (1 + 2 * 0.5),  # messages 1-
    (0 + 0.5) / (1 + 2 * 0.5)  # hello 0+
]

# model.ham_messages =  2
probs_if_ham = [
    (0 + 0.5) / (2 + 2 + 0.5),  # spam 0+
    1 - (2 + 0.5) / (2 + 2 * 0.5),  # ham 2-
    1 - (1 + 0.5) / (2 + 2 * 0.5),  # messages 1-
    (1 + 0.5) / (2 + 2 * 0.5)  # hello 1+
]

# Вычисляем вероятности
p_if_spam = math.exp(sum(math.log(p) for p in probs_if_spam))
p_if_ham = math.exp(sum(math.log(p) for p in probs_if_ham))

# Итоговая вероятность для спама
res = p_if_spam / (p_if_ham + p_if_spam)
print(f"Итоговая вероятность для спама: {res:.4f}")

print(model.predict(text) == res)  # :(

# -----------------------------------------------------------------v2------------------------------------------------------------------

# Пример обучения
messages = [
    Message("Free money now!", is_spam=True),  # Спам
    Message("Hello, how are you?", is_spam=False),  # Не спам
    Message("Get a discount on products!", is_spam=True),  # Спам
    Message("Meeting at 10 AM", is_spam=False),  # Не спам
    Message("Congratulations, you won a lottery!", is_spam=True),  # Спам
    Message("Are we still on for the meeting?", is_spam=False),  # Не спам
]

# Обучаем модель
model = NaiveBayesClassifier(k=0.5)
model.train(messages)

# -------------------------------------------------------------------------
test_texts = [
    "Free lottery tickets available",  # Ожидаем, что это будет спам
    "Let's grab lunch tomorrow",  # Ожидаем, что это не спам
    "Huge sale on electronics!",  # Ожидаем, что это будет спам
    "Project meeting rescheduled"  # Ожидаем, что это не спам
]

# Применяем модель для предсказания
for text in test_texts:
    prediction = model.predict(text)
    print(f"Text: '{text}'")
    print(f"Prediction (Spam Probability): {prediction:.4f}")
    print(f"Prediction: {'Spam' if prediction > 0.5 else 'Ham'}\n")

print()
print(model.predict_verbose("Let's grab lunch tomorrow"))

# ------------------------------------------------------------------v3----------------------------------------------------------------------------

training_messages = [
    # Спам
    Message("Win a free iPhone now!", is_spam=True),
    Message("Congratulations! You won a lottery.", is_spam=True),
    Message("Limited time offer: huge discount on products", is_spam=True),
    Message("Free money available for lucky winners", is_spam=True),
    Message("Get your prize now for free", is_spam=True),
    Message("Click here to claim your $1000 gift card", is_spam=True),

    # Хэм
    Message("Let's grab lunch tomorrow", is_spam=False),
    Message("Are you coming to the meeting?", is_spam=False),
    Message("Project deadline is next Monday", is_spam=False),
    Message("How are you doing today?", is_spam=False),
    Message("Can we reschedule our appointment?", is_spam=False),
    Message("I'll send you the report by evening", is_spam=False),
    Message("Reminder: team meeting at 10 AM", is_spam=False),
    Message("Dinner at my place?", is_spam=False),
]

# Обучаем модель
model = NaiveBayesClassifier(k=0.5)
model.train(training_messages)

# -------------------------------------------------------------------------
test_texts = [
    "Free lottery tickets available",  # Ожидаем, что это будет спам
    "Let's grab lunch tomorrow",  # Ожидаем, что это не спам
    "Huge sale on electronics!",  # Ожидаем, что это будет спам
    "Project meeting rescheduled"  # Ожидаем, что это не спам
]

# Применяем модель для предсказания
for text in test_texts:
    prediction = model.predict(text)
    print(f"Text: '{text}'")
    print(f"Prediction (Spam Probability): {prediction:.4f}")
    print(f"Prediction: {'Spam' if prediction > 0.5 else 'Ham'}\n")

print()
print(model.predict_verbose("Let's grab lunch tomorrow"))
