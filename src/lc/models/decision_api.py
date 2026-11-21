"""A decision model behind OpenRouter's Decisions API: the provider runs it, in its own best setup.

The request is the local decision model's (``decision.request``), laid out as the Decision Index kit sends
text tasks: an empty state, the text and the question in the instructions, the options keyed A, B, C...
with their descriptions. The provider applies the model's own prompt template and readout, and answers
with a probability per option, used as is.

A request is retried 3 times on rate limits, server errors and network errors. A refusal, or an error left
after the retries, raises ``RuntimeError``: the runner records the run as failed and moves on. A text the
provider rejects for quoting reserved tokens (``<|im_end|>``, ``</think>``: 1 test and at most 3 training
papers of 11,000) is resent once with a space after each ``<``. Each answer
appends one line (model, tokens, cost, latency, answer type) to ``usage_log`` when one is given.
"""

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from lc.models.decision import LABELS_FILE, request

URL = "https://openrouter.ai/api/alpha/decisions"
RETRIES = 3
TIMEOUT_S = 120
QUESTION = "category"


@dataclass(frozen=True)
class DecisionAPI:
    model: str  # an OpenRouter slug, e.g. baseten/mammoth-1
    usage_log: str | None = None  # not part of the identity: where to write, not what is measured

    def identity(self) -> dict:
        return {"approach": "decision-api", "model": self.model, "layout": "decision-index-text-v1"}

    def fit(self, task: str, train: pd.DataFrame, labels: list[str], seed: int, out_dir: Path) -> int:
        (out_dir / LABELS_FILE).write_text(json.dumps({"task": task, "labels": labels}))
        return 0

    def load(self, out_dir: Path) -> "DecisionAPIPredictor":
        spec = json.loads((out_dir / LABELS_FILE).read_text())
        key = os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise RuntimeError("OPENROUTER_API_KEY is not set")
        return DecisionAPIPredictor(self.model, key, spec["task"], spec["labels"], self.usage_log)

    def disk_mb(self, out_dir: Path) -> float:
        return 0.0


def body(model: str, task: str, text: str, labels: list[str]) -> tuple[dict, list[str]]:
    """The request body, and our label for each option key in order."""
    instructions, options, order = request(task, text, labels)
    question = {"type": "choice", "instructions": instructions, "criteria": dict(options)}
    return {"model": model, "state": "{}", "questions": {QUESTION: question}}, order


class DecisionAPIPredictor:
    def __init__(self, model: str, key: str, task: str, labels: list[str], usage_log: str | None):
        self.model, self.key, self.task, self.labels, self.usage_log = model, key, task, labels, usage_log

    def predict(self, text: str) -> np.ndarray:
        payload, order = body(self.model, self.task, text, self.labels)
        keys = list(payload["questions"][QUESTION]["criteria"])
        tick = time.perf_counter()
        try:
            response = self._post(payload)
        except RuntimeError as error:
            # A few abstracts quote tokens such as <|im_end|> or </think>, which some providers reject:
            # resend those, and only those, with a space after each "<".
            if "reserved" not in str(error):
                raise
            payload, _ = body(self.model, self.task, text.replace("<", "< "), self.labels)
            response = self._post(payload)
        answer = response.get("answers", {}).get(QUESTION, {})
        self._log(response, answer, (time.perf_counter() - tick) * 1000)
        if answer.get("type") != "choice" or set(answer.get("probabilities", {})) != set(keys):
            raise RuntimeError(f"{self.model} gave no usable answer: {json.dumps(answer)[:300]}")
        by_label = {label: answer["probabilities"][key] for key, label in zip(keys, order, strict=True)}
        probs = np.array([by_label[label] for label in self.labels], dtype=float)
        return probs / probs.sum()  # providers round to 4 decimals: the sum can be off by a few 1e-4

    def _post(self, payload: dict) -> dict:
        data = json.dumps(payload).encode()
        headers = {"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"}
        for attempt in range(RETRIES + 1):
            try:
                req = urllib.request.Request(URL, data, headers)
                with urllib.request.urlopen(req, timeout=TIMEOUT_S) as r:
                    return json.loads(r.read())
            except urllib.error.HTTPError as error:
                detail = error.read().decode(errors="replace")[:300]
                if (error.code != 429 and error.code < 500) or attempt == RETRIES:
                    raise RuntimeError(f"{self.model}: HTTP {error.code}: {detail}") from error
            except (urllib.error.URLError, TimeoutError) as error:
                if attempt == RETRIES:
                    raise RuntimeError(f"{self.model}: {error}") from error
            time.sleep(2 ** (attempt + 1))
        raise AssertionError("unreachable")

    def _log(self, response: dict, answer: dict, latency_ms: float) -> None:
        if not self.usage_log:
            return
        usage = response.get("usage", {})
        line = {
            "model": self.model,
            "provider": response.get("provider"),
            "type": answer.get("type"),
            "input_tokens": usage.get("input_tokens"),
            "cost": usage.get("cost"),
            "latency_ms": round(latency_ms, 1),
        }
        with Path(self.usage_log).open("a") as f:
            f.write(json.dumps(line) + "\n")
