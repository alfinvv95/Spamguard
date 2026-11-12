
import os
import pandas as pd
from email import policy
from email.parser import BytesParser

emails = []

folders = {
    "dataset/extracted/easy_ham_2": "ham",
    "dataset/extracted/spam_2": "spam"
}

for folder, label in folders.items():

    print("Reading:", folder)

    for filename in os.listdir(folder):

        file_path = os.path.join(folder, filename)

        if not os.path.isfile(file_path):
            continue

        try:
            with open(file_path, "rb") as file:
                message = BytesParser(
                    policy=policy.default
                ).parse(file)

            subject = str(message.get("subject", ""))

            body = ""

            if message.is_multipart():
                part = message.get_body(
                    preferencelist=("plain",)
                )

                if part:
                    body = part.get_content()
            else:
                body = message.get_content()

            text = subject + " " + str(body)

            emails.append({
                "text": text,
                "label": label
            })

        except Exception as error:
            print("Skipped:", filename)

df = pd.DataFrame(emails)

os.makedirs("dataset", exist_ok=True)

df.to_csv(
    "dataset/emails.csv",
    index=False
)

print("\nDataset created successfully!")
print("\nTotal emails:", len(df))
print(df["label"].value_counts())
