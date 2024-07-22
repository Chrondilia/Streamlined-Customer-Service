# main.py

import openai

# Set the OpenAI API key directly in the code
openai.api_key = "sk-proj-30EoPOxtdY1wnmOv6QMtT3BlbkFJiumXoZFSoYVAADoCtO2v"

def chatbot_response(user_message):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # Ensure this is the correct model name
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_message},
        ],
        max_tokens=150,
        temperature=0.7,
    )
    return response['choices'][0]['message']['content'].strip()

# Example of usage (optional for testing)
if __name__ == "__main__":
    user_message = "Hello, how are you?"
    print(chatbot_response(user_message))




import random
import re
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline

nltk.download('vader_lexicon')

class SpecialistBot:
    def return_queries(self, sentiment):
        responses = {
            "Positive emotion": (
                "Returns are processed within 7-10 business days.",
                "To initiate a return, please visit our website and fill out the return form.",
                "Returns are accepted within 30 days of purchase with proof of receipt."
            ),
            "Negative emotion": (
                "We apologize for any inconvenience caused by return delays. Please contact us for assistance.",
                "If your return is delayed, please contact our customer service team for updates."
            ),
            "Neutral emotion": (
                "For return inquiries, please check our return policy on our website.",
                "You can track the status of your return by logging into your account."
            )
        }
        return random.choice(responses[sentiment])

    def generic_question_1(self):
        responses = (
            "Could you please clarify your question?",
            "I'm sorry, I didn't quite understand. Could you elaborate?",
            "Is there something specific you would like to know?"
        )
        return random.choice(responses)

    def generic_question_2(self):
        responses = (
            "I'm sorry, I'm not able to assist with that.",
            "That's outside my area of expertise. Can I help you with something else?",
            "I recommend reaching out to our support team for assistance."
        )
        return random.choice(responses)

class RuleBot:
    negative_responses = (
        "sorry", "no", "not", "never", "don't", "nothing", "none", "nobody", 
        "neither", "nowhere", "hardly", "barely", "scarcely", "against", "lacking", 
        "without", "fail", "refuse", "deny", "decline", "avoid", "reject", "can't", 
        "won't", "didn't", "isn't", "wasn't", "shouldn't", "couldn't", "wouldn't", 
        "haven't", "hadn't", "doesn't"
    )

    exit_commands = (
        "quit", "exit", "end", "bye", "goodbye", "see you", "later", "farewell", 
        "close", "stop", "disconnect", "terminate", "adios", "ciao", "au revoir", 
        "take care", "catch you later", "peace out", "I'm out", "sign off", "log off", 
        "finish", "over and out", "so long", "talk to you later", "gotta go", "check out", 
        "time to go", "end chat", "end conversation", "wrap up", "good night", "good day", 
        "until next time"
    )

    random_question_prompts = (
        "Can you tell me more about this topic?", 
        "What is the purpose of this feature?", 
        "How does this system work?", 
        "Why is this method used?", 
        "Who is responsible for this project?", 
        "Where can I find more information?", 
        "When did this event take place?", 
        "Is it true that this approach is effective?", 
        "Do you know any alternatives to this solution?", 
        "Could you explain how this algorithm functions?", 
        "What are the benefits of using this tool?", 
        "Can you describe the process step-by-step?", 
        "What happens if this step is skipped?", 
        "How do I resolve this issue?", 
        "Why should I consider this option?", 
        "What does it mean to optimize this parameter?", 
        "Can you help me understand this concept?", 
        "What is the difference between these two methods?", 
        "Is there a way to automate this task?", 
        "Can you show me an example?", 
        "What are the causes of this problem?", 
        "How can I improve the performance?", 
        "What are some examples of best practices?", 
        "Can you list the key features?", 
        "How do I prevent this error?", 
        "What is the process for getting started?", 
        "What steps should I take to complete this task?", 
        "Why does this error occur?", 
        "What are the consequences of ignoring this issue?", 
        "Can you provide information on the latest update?"
    )

    def __init__(self):
        self.patterns = {
            "order_queries": r'\b(order|price|stock)\b',
            "technical_support": r'\b(technical|support|issue|problem|bug|error|troubleshoot)\b',
            "billing_status": r'\b(billing|invoice|payment|charge)\b',
            "service_availability": r'\b(service|availability|access|downtime|maintenance|outage)\b',
            "return_queries": r'\b(return|refund|exchange|back|send\sback)\b'
        }
        self.specialist_bot = SpecialistBot()
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.vectorizer = TfidfVectorizer()
        self.classifier = Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('clf', LinearSVC())
        ])
        self.train_intent_classifier()

    def train_intent_classifier(self):
        training_data = [
            ("I want to return an item", "return_queries"),
            ("How can I get a refund?", "return_queries"),
            ("My order has an issue", "order_queries"),
            ("What is the price of this product?", "order_queries"),
            ("I need technical support", "technical_support"),
            ("There is a bug in the system", "technical_support"),
            ("What is my billing status?", "billing_status"),
            ("I have a payment issue", "billing_status"),
            ("Is the service down?", "service_availability"),
            ("Is there any downtime scheduled?", "service_availability"),
            ("Where is my order?", "order_queries"),
            ("Can I cancel my order?", "order_queries"),
            ("I need help with my order", "order_queries"),
            ("I want to change my order", "order_queries"),
            ("I have a technical issue", "technical_support"),
            ("I can't log in", "technical_support"),
            ("My account is locked", "technical_support"),
            ("I need help with billing", "billing_status"),
            ("I have a billing problem", "billing_status"),
            ("Is the service available?", "service_availability"),
            ("When will the service be back?", "service_availability")
        ]
        X_train, y_train = zip(*training_data)
        self.classifier.fit(X_train, y_train)

    def greet(self):
        self.name = input("What is your name? ")
        will_help = input(f"Hello {self.name}, how may I help you today? ")
        if any(neg in will_help.lower() for neg in self.negative_responses):
            print("It was great talking to you. Have a great time ahead!")
            return
        self.chat()

    def make_exit(self, reply):
        for command in self.exit_commands:
            if command in reply:
                print("It was great talking to you. Have a great time ahead!")
                return True
        return False

    def chat(self):
        reply = input(random.choice(self.random_question_prompts) + " ").lower()
        while not self.make_exit(reply):
            reply = input(self.match_reply(reply))

    def match_reply(self, reply):
        sentiment = self.analyze_sentiment(reply)
        intent = self.analyze_intent(reply)
        
        if intent:
            if intent == 'order_queries':
                return self.order_queries(sentiment)
            elif intent == 'technical_support':
                return self.technical_support(sentiment)
            elif intent == 'billing_status':
                return self.billing_status(sentiment)
            elif intent == 'service_availability':
                return self.service_availability(sentiment)
            elif intent == 'return_queries':
                return self.specialist_bot.return_queries(sentiment)
        return self.no_match(sentiment)

    def order_queries(self, sentiment):
        responses = {
            "Positive emotion": (
                "Your order was successfully placed and is being processed. Can I help with anything else?",
                "The price depends on the items selected. Please specify the product for accurate pricing.",
                "We are processing your order. Can you provide the order number for more details?",
                "Your payment was successful, and your order is confirmed. Thank you for shopping with us!"
            ),
            "Negative emotion": (
                "We apologize for the delay in your order. Our team is working hard to resolve this issue.",
                "I understand your frustration. Let me check the status of your order for you.",
                "Please accept our apologies for the delay. We're investigating the cause and will update you soon."
            ),
            "Neutral emotion": (
                "For specific order details, please check your email confirmation or account dashboard.",
                "Availability and pricing of items are listed on our website. Please visit for the latest information.",
                "Your order is being processed. For updates, please check your order history on our website."
            )
        }
        return random.choice(responses[sentiment])

    def technical_support(self, sentiment):
        responses = {
            "Positive emotion": (
                "Our technical support team is here to help you resolve any issues.",
                "Thank you for reaching out. Let's work together to fix the problem.",
                "It seems like you're experiencing some issues. Let's see how we can fix it."
            ),
            "Negative emotion": (
                "We understand this is frustrating. Let's see how we can fix the issue you're facing.",
                "It seems frustrating. Let's see how we can fix the issue you're facing.",
                "I understand this is challenging. Can you provide more details so we can assist you better?"
            ),
            "Neutral emotion": (
                "Please try restarting your device. Often, this resolves many technical issues.",
                "Check if your software is up-to-date. Updates can fix bugs and improve performance.",
                "For technical support, please visit our support page or contact our helpdesk."
            )
        }
        return random.choice(responses[sentiment])

    def billing_status(self, sentiment):
        responses = {
            "Positive emotion": (
                "Your billing status is currently up to date.",
                "Please log in to your account to view your billing status.",
                "You can check your billing history by logging into your account."
            ),
            "Negative emotion": (
                "There appears to be an issue with your billing status. Please contact our support team for assistance.",
                "You have an outstanding balance on your account. Please make a payment to update your billing status.",
                "Your billing status may be temporarily unavailable due to system maintenance."
            ),
            "Neutral emotion": (
                "For billing inquiries, please contact our customer service team.",
                "You are currently subscribed to our premium plan. Your billing is handled automatically.",
                "Please provide your account ID so we can check your billing status."
            )
        }
        return random.choice(responses[sentiment])

    def service_availability(self, sentiment):
        responses = {
            "Positive emotion": (
                "Our services are available 24/7 for your convenience.",
                "We strive to provide reliable service across all platforms and devices."
            ),
            "Negative emotion": (
                "Service availability may be affected due to high demand. Please check our website for updates.",
                "We apologize for any inconvenience caused by service disruptions. Our team is working to resolve this.",
                "If you're experiencing service issues, please contact our support team for immediate assistance."
            ),
            "Neutral emotion": (
                "Please ensure your device is connected to the internet to access our services.",
                "Service status updates are available on our website. Please check there for real-time information."
            )
        }
        return random.choice(responses[sentiment])

    def no_match(self, sentiment):
        responses = {
            "Positive emotion": (
                "I appreciate your question. Let's explore this further.",
                "That's an interesting topic. Let's discuss it in more detail.",
                "It seems like a unique question. I'm here to help you find the answer."
            ),
            "Negative emotion": (
                "I'm sorry, I don't have information on that topic right now.",
                "It looks like I can't assist with that specific query. Is there something else I can help with?",
                "Unfortunately, I'm not able to answer that question at the moment."
            ),
            "Neutral emotion": (
                "Let's focus on another topic. How else can I assist you today?",
                "I recommend looking up that information online or consulting an expert.",
                "I'm here to help with any other questions you may have."
            )
        }
        return random.choice(responses[sentiment])

    def analyze_sentiment(self, text):
        score = self.sentiment_analyzer.polarity_scores(text)
        compound_score = score['compound']

        if compound_score >= 0.05:
            return "Positive emotion"
        elif compound_score <= -0.05:
            return "Negative emotion"
        else:
            return "Neutral emotion"

    def analyze_intent(self, text):
        return self.classifier.predict([text])[0]

def main():
    rule_bot = RuleBot()
    rule_bot.greet()

if __name__ == "__main__":
    main()
