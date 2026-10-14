"""Shared helpers for the compression-intelligence experiments.

Modules: `pairs` (model registry), `text` (tokenization and windows), `score` (bits from a model),
`stats` (bootstrap and position buckets), `io` (corpora, CSV rows, saved bits, HF cache).
Everything is pure except the effect functions named in each module's docstring.
"""
