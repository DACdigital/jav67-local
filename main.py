"""Embed a phrase and a set of options with a local Ollama model, then rank options by cosine distance."""

import argparse
import math
import time

from agno.knowledge.embedder.ollama import OllamaEmbedder
from pydantic import BaseModel, Field

DEFAULT_MODEL = "qwen3-embedding:0.6b"
DEFAULT_DIMENSIONS = 1024  # qwen3-embedding:0.6b output size
DEFAULT_HOST = "http://localhost:11434"

# Qwen3 embedding models expect the query side wrapped as "Instruct: <task>\nQuery: <text>".
# The option side is embedded bare.
DEFAULT_INSTRUCTION = "Identify the emotion expressed in the given text"

# Softmax sharpness (inverse temperature) applied to cosine similarities when computing confidence.
# 0 gives a uniform distribution; larger values concentrate confidence on the best option.
DEFAULT_SHARPNESS = 10.0


class Option(BaseModel):
    label: str = Field(min_length=1)
    description: str | None = None

    @classmethod
    def parse(cls, text: str) -> "Option":
        """Parse 'Label' or 'Label: description' as given on the command line."""
        label, sep, description = text.partition(":")
        return cls(label=label.strip(), description=description.strip() if sep else None)

    @property
    def embedding_text(self) -> str:
        return f"{self.label}: {self.description}" if self.description else self.label


DEFAULT_OPTIONS = [
    Option(label="Anger", description="feeling furious, frustrated or fed up"),
    Option(label="Fear", description="feeling scared, anxious or threatened"),
    Option(label="Joy", description="feeling happy, delighted or pleased"),
]


class MatchRequest(BaseModel):
    phrase: str = Field(min_length=1)
    options: list[Option] = Field(min_length=2)
    instruction: str | None = DEFAULT_INSTRUCTION  # None disables the prefix
    sharpness: float = Field(default=DEFAULT_SHARPNESS, ge=0)

    @property
    def query_text(self) -> str:
        if self.instruction:
            return f"Instruct: {self.instruction}\nQuery: {self.phrase}"
        return self.phrase


class OptionScore(BaseModel):
    option: str
    cosine_similarity: float
    cosine_distance: float
    scaled: float  # min-max scaled similarity across options: best = 1, worst = 0
    confidence: float  # softmax(sharpness * cosine_similarity); sums to 1 across options
    embed_ms: float  # time to embed this option's text


class Timings(BaseModel):
    phrase_embed_ms: float
    options_embed_ms: float  # sum over all options
    scoring_ms: float  # cosine similarity, scaling, confidence and sorting
    total_ms: float


class MatchResult(BaseModel):
    phrase: str
    model: str
    scores: list[OptionScore]  # sorted by ascending distance, best match first
    sharpness: float
    timings: Timings

    @property
    def best(self) -> OptionScore:
        return self.scores[0]

    @property
    def margin(self) -> float:
        """Similarity gap between the best and second-best option."""
        return self.scores[0].cosine_similarity - self.scores[1].cosine_similarity


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        raise ValueError("Cannot compute cosine similarity of a zero vector")
    return dot / (norm_a * norm_b)


def min_max_scale(values: list[float]) -> list[float]:
    """Scale to [0, 1] with the maximum at 1 and the minimum at 0. All-equal input scales to 1."""
    lo, hi = min(values), max(values)
    if hi == lo:
        return [1.0] * len(values)
    return [(v - lo) / (hi - lo) for v in values]


def softmax(values: list[float], sharpness: float) -> list[float]:
    """Numerically stable softmax of sharpness * values."""
    scaled = [sharpness * v for v in values]
    peak = max(scaled)
    exps = [math.exp(v - peak) for v in scaled]
    total = sum(exps)
    return [e / total for e in exps]


def rescore(result: "MatchResult", sharpness: float) -> "MatchResult":
    """Recompute confidence for a different sharpness without re-embedding."""
    sims = [s.cosine_similarity for s in result.scores]
    confidences = softmax(sims, sharpness)
    scores = [s.model_copy(update={"confidence": c}) for s, c in zip(result.scores, confidences, strict=True)]
    return result.model_copy(update={"scores": scores, "sharpness": sharpness})


def make_embedder(model: str, dimensions: int, host: str) -> OllamaEmbedder:
    return OllamaEmbedder(id=model, dimensions=dimensions, host=host)


def embed(embedder: OllamaEmbedder, text: str) -> list[float]:
    vector = embedder.get_embedding(text)
    if not vector:
        raise RuntimeError(
            f"No embedding returned for {text!r}. Check that '{embedder.id}' is pulled "
            f"and that dimensions={embedder.dimensions} matches its output size."
        )
    return vector


def _ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000


def match(request: MatchRequest, embedder: OllamaEmbedder) -> MatchResult:
    t_total = time.perf_counter()

    t = time.perf_counter()
    phrase_vec = embed(embedder, request.query_text)
    phrase_embed_ms = _ms(t)

    option_vecs: list[tuple[Option, list[float], float]] = []
    for option in request.options:
        t = time.perf_counter()
        vec = embed(embedder, option.embedding_text)
        option_vecs.append((option, vec, _ms(t)))
    options_embed_ms = sum(ms for _, _, ms in option_vecs)

    t = time.perf_counter()
    sims = [cosine_similarity(phrase_vec, vec) for _, vec, _ in option_vecs]
    scaled = min_max_scale(sims)
    confidences = softmax(sims, request.sharpness)
    scores = [
        OptionScore(
            option=option.label,
            cosine_similarity=sim,
            cosine_distance=1.0 - sim,
            scaled=sc,
            confidence=conf,
            embed_ms=embed_ms,
        )
        for (option, _, embed_ms), sim, sc, conf in zip(option_vecs, sims, scaled, confidences, strict=True)
    ]
    scores.sort(key=lambda s: s.cosine_distance)
    scoring_ms = _ms(t)

    timings = Timings(
        phrase_embed_ms=phrase_embed_ms,
        options_embed_ms=options_embed_ms,
        scoring_ms=scoring_ms,
        total_ms=_ms(t_total),
    )
    return MatchResult(
        phrase=request.phrase, model=embedder.id, scores=scores, sharpness=request.sharpness, timings=timings
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank options by cosine distance to a phrase using a local Ollama model.")
    parser.add_argument("phrase", help="Text to compare against the options")
    parser.add_argument(
        "--options",
        nargs="+",
        default=None,
        help="Candidate labels, each as 'Label' or 'Label: description' (default: Anger, Fear, Joy with descriptions)",
    )
    parser.add_argument(
        "--instruction",
        default=DEFAULT_INSTRUCTION,
        help=f"Task instruction prepended to the phrase (default: {DEFAULT_INSTRUCTION!r})",
    )
    parser.add_argument("--no-instruction", action="store_true", help="Embed the bare phrase without an instruction")
    parser.add_argument(
        "--sharpness",
        type=float,
        default=DEFAULT_SHARPNESS,
        help=f"Softmax sharpness for confidence; higher favours the best option (default: {DEFAULT_SHARPNESS})",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Ollama model id (default: {DEFAULT_MODEL})")
    parser.add_argument(
        "--dimensions",
        type=int,
        default=DEFAULT_DIMENSIONS,
        help=f"Expected embedding size; must match the model (default: {DEFAULT_DIMENSIONS})",
    )
    parser.add_argument("--host", default=DEFAULT_HOST, help=f"Ollama base URL (default: {DEFAULT_HOST})")
    parser.add_argument("--json", action="store_true", help="Print the result as JSON instead of a table")
    args = parser.parse_args()

    options = [Option.parse(o) for o in args.options] if args.options else DEFAULT_OPTIONS
    request = MatchRequest(
        phrase=args.phrase,
        options=options,
        instruction=None if args.no_instruction else args.instruction,
        sharpness=args.sharpness,
    )
    result = match(request, make_embedder(args.model, args.dimensions, args.host))

    if args.json:
        print(result.model_dump_json(indent=2))
        return

    print(f"model:   {result.model}")
    print(f"phrase:  {result.phrase!r}")
    print(f"query:   {request.query_text!r}")
    print()
    width = max(len(s.option) for s in result.scores)
    print(f"{'option':<{width}}  {'similarity':>10}  {'distance':>10}  {'scaled':>7}  {'confidence':>10}  {'embed ms':>9}")
    for s in result.scores:
        print(
            f"{s.option:<{width}}  {s.cosine_similarity:>10.4f}  {s.cosine_distance:>10.4f}  "
            f"{s.scaled:>7.3f}  {s.confidence:>10.1%}  {s.embed_ms:>9.1f}"
        )
    print()
    print(f"best:       {result.best.option}")
    print(f"margin:     {result.margin:.4f}")
    print(f"confidence: {result.best.confidence:.1%}  (sharpness {result.sharpness:g})")
    print()
    t = result.timings
    n = len(result.scores)
    print("timings")
    print(f"  phrase embedding:   {t.phrase_embed_ms:>8.1f} ms")
    print(f"  options embedding:  {t.options_embed_ms:>8.1f} ms  ({n} options, {t.options_embed_ms / n:.1f} ms avg)")
    print(f"  scoring:            {t.scoring_ms:>8.2f} ms")
    print(f"  total:              {t.total_ms:>8.1f} ms")


if __name__ == "__main__":
    main()
