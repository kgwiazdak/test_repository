"""FA(3) XML validation utilities."""

from pathlib import Path
from urllib.request import urlretrieve

from lxml import etree

SCHEMA_URL = "https://example.com/fa3_schema.xsd"
SCHEMA_DIR = Path("data/fa3_schema")
SCHEMA_PATH = SCHEMA_DIR / "fa3_schema.xsd"


def ensure_schema():
    """Download FA(3) XSD schema if missing."""
    SCHEMA_DIR.mkdir(parents=True, exist_ok=True)
    if not SCHEMA_PATH.exists():
        urlretrieve(SCHEMA_URL, SCHEMA_PATH)


def validate_xml(xml_path: Path) -> bool:
    """Validate XML file against FA(3) schema."""
    ensure_schema()
    schema_doc = etree.parse(str(SCHEMA_PATH))
    schema = etree.XMLSchema(schema_doc)
    doc = etree.parse(str(xml_path))
    return schema.validate(doc)