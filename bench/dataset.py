"""Dataset loading and default corpus/queries.

If ``data/corpus.txt`` and ``data/queries.txt`` don't exist, the built-in
Wikipedia excerpt and default questions are used.
"""
from __future__ import annotations

import os
from pathlib import Path

_DEFAULT_CORPUS = """
Marie Curie was a pioneering physicist and chemist who conducted groundbreaking research on
radioactivity. She was the first woman to win a Nobel Prize and the only person to win Nobel
Prizes in two different sciences. Pierre Curie, her husband, collaborated with her on this
research. They discovered two elements: polonium and radium. The Curies worked at the
University of Paris.

Albert Einstein developed the theory of relativity, which fundamentally changed our
understanding of space, time, and energy. His famous equation E=mc2 describes the
equivalence of mass and energy. Einstein was awarded the Nobel Prize in Physics in 1921 for
his discovery of the law of the photoelectric effect. He worked at the Institute for Advanced
Study in Princeton.

Isaac Newton formulated the laws of motion and universal gravitation. His work laid the
foundation for classical mechanics. Newton also made significant contributions to optics and
developed calculus independently. He was a fellow of the Royal Society and later became its
president. Newton spent most of his career at Cambridge University.

Nikola Tesla was an inventor and electrical engineer who is best known for his contributions
to the design of the modern alternating current (AC) electricity supply system. He worked
briefly for Thomas Edison before striking out on his own. Tesla's patents and theoretical work
formed the basis of modern AC electric power systems, including the polyphase power distribution
systems and the AC motor. He collaborated with George Westinghouse to develop AC power.

Charles Darwin developed the theory of evolution by natural selection, published in his book
On the Origin of Species in 1859. He observed that organisms better adapted to their
environment tend to survive and produce more offspring. Darwin conducted extensive research
during his voyage on HMS Beagle. He later corresponded with Alfred Russel Wallace who
independently developed a similar theory.
"""

_DEFAULT_QUERIES = [
    "Who discovered polonium and radium?",
    "What is Einstein's most famous equation?",
    "Where did Newton work?",
    "Who invented the AC electricity supply system?",
    "What book did Darwin publish in 1859?",
    "Which scientists won Nobel Prizes?",
    "Who collaborated with Marie Curie?",
    "What did Einstein win the Nobel Prize for?",
    "What theory did Darwin develop?",
    "Where did Tesla work before going independent?",
]


def load_corpus(path: str | None = None) -> str:
    if path and os.path.exists(path):
        return Path(path).read_text(encoding="utf-8")
    return _DEFAULT_CORPUS.strip()


def load_queries(path: str | None = None) -> list[str]:
    if path and os.path.exists(path):
        lines = Path(path).read_text(encoding="utf-8").splitlines()
        return [ln.strip() for ln in lines if ln.strip()]
    return _DEFAULT_QUERIES
