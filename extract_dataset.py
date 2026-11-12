
import tarfile
import os

dataset_folder = "dataset"
extract_folder = "dataset/extracted"

os.makedirs(extract_folder, exist_ok=True)

files = [
    "20030228_easy_ham_2.tar.bz2",
    "20030228_spam_2.tar.bz2"
]

for filename in files:
    file_path = os.path.join(dataset_folder, filename)

    if os.path.exists(file_path):
        with tarfile.open(file_path, "r:bz2") as archive:
            archive.extractall(
                path=extract_folder,
                filter="data"
            )

        print(filename, "extracted successfully!")
    else:
        print(filename, "not found!")

print("Dataset extraction completed!")
