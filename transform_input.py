import pandas as pd
import nltk
from nltk.corpus import stopwords as nltk_stopwords
import datetime
import re
import os
from glob import glob
import xml.etree.ElementTree as ET

def normalize_and_transform():
    nltk.download('stopwords')
    stopwords = nltk_stopwords.words('german')
    
    start = datetime.datetime.now().replace(microsecond=0)
    
    # Merge XML files
    input_folder = "../mensaArchiv"
    output_file = "merged_speiseplan.csv"
    
    all_titles = []

    filenames = glob(os.path.join(input_folder, "*.xml"))
    for filename in filenames:
        try:
            with open(filename, 'r', encoding='utf-8') as file:
                tree = ET.parse(file)
            root = tree.getroot()
            for title in root.findall(".//title"):
                title_text = title.text or ""
                cleaned_title = clean_title(title_text)
                if cleaned_title:
                    all_titles.append(cleaned_title)
                else:
                    print(f"Invalid characters found: {title_text} in {filename}")
        except ET.ParseError:
            print(f"Fehler beim Parsen von {filename}, Datei wird übersprungen.")

    # Save merged titles
    with open(output_file, "w", encoding="utf-8") as outfile:
        for title in all_titles:
            outfile.write(f"{title}\n")

    gerichte = pd.read_csv(
        output_file,
        header=None,
        sep=";",
        names=["Gericht"]
    )

    # Remove stopwords
    gerichte['Gericht'] = gerichte['Gericht'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stopwords)]))

    # Normalize
    anzahl_gerichte = (
        gerichte.groupby("Gericht")
        .size()
        .reset_index(name="Anzahl")
        .sort_values(by="Anzahl", ascending=False)
    )

    # Reorder columns
    anzahl_gerichte = anzahl_gerichte[["Anzahl", "Gericht"]]

    # Save result
    anzahl_gerichte.to_csv(
        "gerichte_anzahl-py-normalized.csv",
        index=False,
        sep=";",
        quoting=3
    )

    end = datetime.datetime.now().replace(microsecond=0)

    print(f"Transform and normalize done. Time: {end-start}")
    return end-start

def clean_title(title_text):
    bracket_pattern = re.compile(r"\s*\([^)]*\)")
    valid_char_pattern = re.compile(r"^[a-zA-Z0-9äöüÄÖÜß ,.!?()´`\"\'\-:€+-/\\\n]+$")
    normalized_pattern = re.compile(r"[^a-zäöüß\s]")

    if not valid_char_pattern.match(title_text):
        return None

    cleaned_title = bracket_pattern.sub("", title_text).strip().lower()
    cleaned_title = normalized_pattern.sub("", cleaned_title)
    cleaned_title = re.sub(r"\s+", " ", cleaned_title)
    return cleaned_title.strip()