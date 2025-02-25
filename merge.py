import os
from glob import glob
import datetime
import xml.etree.ElementTree as ET

input_folder = "../mensaArchiv"
output_file = "merged_speiseplan-test.csv"

def merge_files():
    start = datetime.datetime.now().replace(microsecond=0)

    all_titles = []

    filenames = glob(os.path.join(input_folder, "*.xml"))
    for filename in filenames:
        try:
            with open(filename, 'r', encoding='utf-8') as file:
                tree = ET.parse(file)
            root = tree.getroot()
            for title in root.findall(".//title"):
                title_text = title.text or ""
                all_titles.append(title_text)
        except ET.ParseError:
            print(f"Fehler beim Parsen von {filename}, Datei wird übersprungen.")

    # Titel speichern
    with open(output_file, "w", encoding="utf-8") as outfile:
        for title in all_titles:
            outfile.write(f"{title}\n")

    end = datetime.datetime.now().replace(microsecond=0)

    print(f"Merging done. Time: {end-start}")
    return end-start