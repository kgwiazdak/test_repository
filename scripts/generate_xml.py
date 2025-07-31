"""Generate FA(3) XML and validate."""

import csv
import json
from pathlib import Path

from lxml import etree

from ksef_vision.mapping.fa3_mapper import map_to_fa3
from ksef_vision.mapping.validator import validate_xml

PREDICTIONS_PATH = Path("results/predictions.json")
OUTPUT_CSV = Path("results/xml_validity.csv")
XML_DIR = Path("results/xml")


def create_xml(record: dict, out_path: Path) -> None:
    root = etree.Element("Invoice")
    for k, v in map_to_fa3(record).items():
        child = etree.SubElement(root, k)
        child.text = str(v)
    tree = etree.ElementTree(root)
    tree.write(str(out_path), encoding="utf-8", xml_declaration=True)


def main():
    XML_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    with PREDICTIONS_PATH.open() as f:
        data = json.load(f)
    for idx, rec in enumerate(data):
        xml_path = XML_DIR / f"{idx}.xml"
        create_xml(rec, xml_path)
        is_valid = validate_xml(xml_path)
        results.append({"file": xml_path.name, "valid": is_valid})
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "valid"])
        writer.writeheader()
        writer.writerows(results)


if __name__ == "__main__":
    main()