import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from tkinter import *
import pickle
import traceback
import emails
import traceback


# -------------------------------
# STEP 1: Load dataset (initial training from local file)
# -------------------------------
try:
    df = pd.read_csv("mail_data.csv", encoding='latin1')
    df = df.where(pd.notnull(df), '')
    df['label'] = df['Category'].map({'ham': 0, 'spam': 1})
    X = df['Message']
    y = df['label']

    vectorizer = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
    X_vec = vectorizer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)
    model = MultinomialNB()
    model.fit(X_train, y_train)

    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"✅ Model trained from local file. Accuracy: {acc:.2f}")
except Exception as e:
    print("❌ Error loading initial dataset:")
    traceback.print_exc()

# -------------------------------
# STEP 2: GUI Functions
# -------------------------------
def check_spam():
    msg = text_input.get("1.0", END).strip()
    if not msg:
        label_result.config(text="Please enter a message.", fg="orange")
        return
    msg_vector = vectorizer.transform([msg])
    prediction = model.predict(msg_vector)[0]
    confidence = model.predict_proba(msg_vector)[0].max() * 100
    if prediction == 1:
        label_result.config(text=f"Spam ({confidence:.2f}%)", fg="red")
    else:
        label_result.config(text=f"Not Spam ({confidence:.2f}%)", fg="green")

def clear_text():
    text_input.delete("1.0", END)
    label_result.config(text="")

def refresh_dataset():
    global model, vectorizer
    try:
        df = pd.read_csv("mail_data.csv", encoding='latin1')


        if 'Category' not in df.columns or 'Message' not in df.columns:
            raise ValueError("Dataset must have 'Category' and 'Message' columns.")

        df = df.where(pd.notnull(df), '')
        df['label'] = df['Category'].map({'ham': 0, 'spam': 1})
        X = df['Message']
        y = df['label']

        vectorizer = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
        X_vec = vectorizer.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(max_iter=1000)

        model.fit(X_train, y_train)

        acc = accuracy_score(y_test, model.predict(X_test))
        label_result.config(text=f"✅ Refreshed. Accuracy: {acc:.2f}", fg="blue")
        print(f"✅ Dataset refreshed successfully. Accuracy: {acc:.2f}")
    except Exception as e:
        traceback.print_exc()
        try:
            label_result.config(text=f"❌ Failed to refresh. {type(e).__name__}: {e}", fg="red")
        except:
            pass

# -------------------------------
# STEP 3: GUI Setup
# -------------------------------
window = Tk()
window.title("Email Spam Detector")
window.geometry("550x500")
window.configure(bg="#f9f9f9")

Label(window, text="Email Spam Detection", font=("Helvetica", 18, "bold"), bg="#f9f9f9").pack(pady=15)
Label(window, text="Enter your email content:", bg="#f9f9f9", font=("Arial", 12)).pack()

text_input = Text(window, height=10, width=60, font=("Arial", 12))
text_input.pack(pady=10)

Button(window, text="Check", command=check_spam, bg="green", fg="white", font=("Arial", 12)).pack(pady=5)
Button(window, text="Clear", command=clear_text, bg="red", fg="white", font=("Arial", 12)).pack(pady=5)
Button(window, text="Refresh Dataset", command=refresh_dataset, bg="#2196F3", fg="white", font=("Arial", 12)).pack(pady=5)

label_result = Label(window, text="", font=("Arial", 14), bg="#f9f9f9")
label_result.pack(pady=20)

window.mainloop()
