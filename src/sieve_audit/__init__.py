"""SIEVE - Safety Indicator Evidence Validation Engine.

Validity checks for AI safety signals: does a signal survive controls, or is it
merely decodable? See DESIGN.md for the v0.1 specification.
"""
from sieve_audit.bundle import (
    DecodabilityEvidence,
    EfficacyRecord,
    EvidenceBundle,
    SteeringRecord,
)
from sieve_audit.config import AuditConfig
from sieve_audit.engine import AuditResult, run_audit
from sieve_audit.prereg import PreRegistration, build_prereg, verify_prereg
from sieve_audit.verdict import INSUFFICIENT_PROTOCOL, AuditCard, Verdict

__version__ = "0.1.0"
__all__ = [
    "AuditCard",
    "AuditConfig",
    "AuditResult",
    "DecodabilityEvidence",
    "EfficacyRecord",
    "EvidenceBundle",
    "INSUFFICIENT_PROTOCOL",
    "PreRegistration",
    "SteeringRecord",
    "Verdict",
    "build_prereg",
    "run_audit",
    "verify_prereg",
    "__version__",
]
