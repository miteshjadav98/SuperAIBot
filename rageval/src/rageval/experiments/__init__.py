"""Reproducible, keyless experiments that dogfood the harness on real questions.

These aren't part of the harness's contract — they're demonstrations that use its own
metric layer to *measure* something. ``retrieval_lift`` is the headline one: it proves the
harness can quantify a retrieval improvement (RRF hybrid vs single-signal BM25).
"""
