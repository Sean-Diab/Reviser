"""Realistic editing data (Task 21): IteraTeR / CoEdIT loading, overlap audit, GPT-2 tokenization,
and derivation + replay validation of synthetic-order edit scripts for human before/after pairs.

Modules:
  * ``sources``     raw-file loaders -> normalized text records (official splits kept).
  * ``dedup``       text normalization, duplicate detection and eval-overlap filtering.
  * ``editscript``  (before_ids, after_ids) -> validated action script + edit statistics.
  * ``build``       end-to-end pipeline writing ``data/real/*.jsonl`` and ``manifest.json``.
"""
