"""Mapping between fields and FA(3) XPath tags."""

FIELD_TO_FA3 = {
    "invoice_number": "P_2",
    "issue_date": "P_6",
    "issuer_name": "P_15",
    "buyer_name": "P_3",
    "net_total": "P_13_1",
    "vat_total": "P_14_1",
    "currency": "P_17",
}


def map_to_fa3(predictions):
    """Map prediction dict to FA(3) fields."""
    return {FIELD_TO_FA3.get(k, k): v for k, v in predictions.items()}