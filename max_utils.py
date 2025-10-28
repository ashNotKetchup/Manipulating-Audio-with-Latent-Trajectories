import json
from typing import Union, List, Any

class MaxUtils:
    @staticmethod
    def sanitize_fragments(parts: List[Any]) -> str:
        cleaned = []
        for x in parts:
            if isinstance(x, tuple) and len(x) == 1 and isinstance(x[0], (int, float)):
                cleaned.append(str(x[0]))
                cleaned.append(",")
            else:
                cleaned.append(str(x))
        s = "".join(cleaned)
        s = s.replace("][", "],[")
        return s

    @staticmethod
    def json_cleaner(json_in: Union[str, List[Any]]) -> Any:
        """Cleans JSON-like input from Max and returns parsed structure."""
        # Only sanitize if input is a fragmented list
        if isinstance(json_in, list):
            json_in = MaxUtils.sanitize_fragments(json_in)

        if isinstance(json_in, str):
            json_in = json_in.strip()
            try:
                # handle double-encoded JSON from Max
                json_in = json.loads(json_in)
            except Exception:
                raise

        return json_in
