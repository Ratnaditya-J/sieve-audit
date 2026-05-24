"""SIEVE - Safety Indicator Evidence Validation Engine.

Validity checks for AI safety signals: does a signal survive controls, or is it
merely decodable? See DESIGN.md for the v0.1 specification.
"""
from sieve_audit.verdict import AuditCard, Verdict

__version__ = "0.0.1"
__all__ = ["AuditCard", "Verdict", "__version__"]
