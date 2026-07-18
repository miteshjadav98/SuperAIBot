"""Harness core — the adapter contract, result types, manifest, and store.

Zero target-specific imports live here. Adapters depend on this package; it never
depends on them. That inversion is what keeps the harness portable.
"""
