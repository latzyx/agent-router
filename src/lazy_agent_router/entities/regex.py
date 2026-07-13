import re
from typing import Any

from .base import BaseEntityExtractor


class RegexEntityExtractor(BaseEntityExtractor):
    """Conservative extractors for common enterprise routing parameters."""

    _amount = re.compile(
        r"(?:(?:人民币|¥|￥|RMB)\s*(?P<currency_value>\d+(?:\.\d+)?)(?P<currency_unit>万|千|元)?|"
        r"(?P<unit_value>\d+(?:\.\d+)?)\s*(?P<unit>万|千|元))",
        re.I,
    )
    _date = re.compile(r"\b(20\d{2}[-/.年]\d{1,2}(?:[-/.月]\d{1,2}日?)?)\b")
    _employee_id = re.compile(r"(?:员工|工号|employee\s*id)\s*[:：#]?\s*([A-Za-z0-9_-]{3,})", re.I)

    def extract(self, text: str) -> dict[str, Any]:
        entities: dict[str, Any] = {}
        if match := self._employee_id.search(text):
            entities["employee_id"] = match.group(1)
        if match := self._date.search(text):
            entities["date"] = match.group(1)
        if match := self._amount.search(text):
            value = match.group("currency_value") or match.group("unit_value")
            unit = match.group("currency_unit") or match.group("unit") or "元"
            entities["amount"] = {"value": float(value), "unit": unit}
        return entities
