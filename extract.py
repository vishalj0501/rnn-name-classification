import os
import sys
import zipfile

# if not os.path.exists("Data"):
#     os.makedirs("Data", exist_ok=True)


def extract(path):
    if os.path.exists(path):
        if zipfile.is_zipfile(path):
            with zipfile.ZipFile(path) as zf:
                zf.extractall()
        else:
            print("Nah!")

extract("data.zip")