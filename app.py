
from flask import Flask, render_template, request, redirect, url_for
import joblib
import sqlite3
from datetime import datetime

app = Flask(__name__)

model = joblib.load("model/spam_model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")


# Create the history database
def create_database():
    connection = sqlite3.connect("history.db")

    connection.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT,
            result TEXT,
            confidence REAL,
            date TEXT
        )
    """)

    connection.commit()
    connection.close()


@app.route("/", methods=["GET", "POST"])
def home():

    result = None
    email_text = ""
    confidence = None

    if request.method == "POST":

        email_text = request.form["email"]

        if email_text.strip():

            email_vector = vectorizer.transform([email_text])

            prediction = model.predict(email_vector)[0]

            probabilities = model.predict_proba(email_vector)[0]

            class_index = list(model.classes_).index(prediction)

            confidence = round(
                probabilities[class_index] * 100, 2
            )

            if prediction == "spam":
                result = "SPAM"
            else:
                result = "NORMAL"

            # Save prediction to database
            connection = sqlite3.connect("history.db")

            connection.execute(
                """INSERT INTO history
                (email, result, confidence, date)
                VALUES (?, ?, ?, ?)""",
                (
                    email_text,
                    result,
                    confidence,
                    datetime.now().strftime("%d-%m-%Y %H:%M")
                )
            )

            connection.commit()
            connection.close()

    # Retrieve previous predictions
    connection = sqlite3.connect("history.db")

    history = connection.execute(
        "SELECT * FROM history ORDER BY id DESC"
    ).fetchall()

    connection.close()

    return render_template(
        "index.html",
        result=result,
        email_text=email_text,
        confidence=confidence,
        history=history
    )


@app.route("/clear-history", methods=["POST"])
def clear_history():

    connection = sqlite3.connect("history.db")

    connection.execute("DELETE FROM history")

    connection.commit()
    connection.close()

    return redirect(url_for("home"))

if __name__ == "__main__":

    create_database()

    app.run(debug=True)
