"""
CWE Taxonomy Mapping Engine.

Maps vulnerability data to MITRE Common Weakness Enumeration (CWE)
identifiers and generates structured security metadata for training samples.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class Severity(Enum):
    """CVSS-derived severity levels."""
    NONE = "NONE"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass(frozen=True)
class CWEEntry:
    """A single CWE taxonomy entry."""
    cwe_id: str            # e.g., "CWE-79"
    name: str              # e.g., "Improper Neutralization of Input During Web Page Generation"
    description: str
    category: str          # High-level category
    related_cwes: tuple[str, ...] = ()
    common_languages: tuple[str, ...] = ()
    mitre_url: str = ""

    @property
    def numeric_id(self) -> int:
        """Extract numeric ID from CWE string."""
        match = re.search(r"\d+", self.cwe_id)
        return int(match.group()) if match else 0


# Top 25 most dangerous CWEs (2024 MITRE list) — core taxonomy
CWE_TAXONOMY: dict[str, CWEEntry] = {
    "CWE-787": CWEEntry(
        cwe_id="CWE-787", name="Out-of-bounds Write",
        description="Software writes data past the end of a buffer.",
        category="Memory Safety",
        common_languages=("C", "C++"),
        mitre_url="https://cwe.mitre.org/data/definitions/787.html",
    ),
    "CWE-79": CWEEntry(
        cwe_id="CWE-79", name="Cross-site Scripting (XSS)",
        description="Improper neutralization of input during web page generation.",
        category="Injection",
        common_languages=("JavaScript", "PHP", "Python", "Java"),
        mitre_url="https://cwe.mitre.org/data/definitions/79.html",
    ),
    "CWE-89": CWEEntry(
        cwe_id="CWE-89", name="SQL Injection",
        description="Improper neutralization of special elements in SQL commands.",
        category="Injection",
        common_languages=("PHP", "Python", "Java", "JavaScript"),
        mitre_url="https://cwe.mitre.org/data/definitions/89.html",
    ),
    "CWE-416": CWEEntry(
        cwe_id="CWE-416", name="Use After Free",
        description="Referencing memory after it has been freed.",
        category="Memory Safety",
        common_languages=("C", "C++"),
        mitre_url="https://cwe.mitre.org/data/definitions/416.html",
    ),
    "CWE-78": CWEEntry(
        cwe_id="CWE-78", name="OS Command Injection",
        description="Improper neutralization of special elements in OS commands.",
        category="Injection",
        common_languages=("Python", "PHP", "Bash", "Java"),
        mitre_url="https://cwe.mitre.org/data/definitions/78.html",
    ),
    "CWE-20": CWEEntry(
        cwe_id="CWE-20", name="Improper Input Validation",
        description="Software does not validate or incorrectly validates input.",
        category="Input Validation",
        common_languages=("Python", "Java", "C", "JavaScript"),
        mitre_url="https://cwe.mitre.org/data/definitions/20.html",
    ),
    "CWE-125": CWEEntry(
        cwe_id="CWE-125", name="Out-of-bounds Read",
        description="Software reads data past the end of a buffer.",
        category="Memory Safety",
        common_languages=("C", "C++"),
        mitre_url="https://cwe.mitre.org/data/definitions/125.html",
    ),
    "CWE-22": CWEEntry(
        cwe_id="CWE-22", name="Path Traversal",
        description="Improper limitation of a pathname to a restricted directory.",
        category="File System",
        common_languages=("Python", "PHP", "Java", "JavaScript"),
        mitre_url="https://cwe.mitre.org/data/definitions/22.html",
    ),
    "CWE-352": CWEEntry(
        cwe_id="CWE-352", name="Cross-Site Request Forgery (CSRF)",
        description="Web app does not verify request was intentionally sent.",
        category="Authentication",
        common_languages=("PHP", "Python", "Java", "JavaScript"),
        mitre_url="https://cwe.mitre.org/data/definitions/352.html",
    ),
    "CWE-434": CWEEntry(
        cwe_id="CWE-434", name="Unrestricted Upload of File with Dangerous Type",
        description="Software allows upload of dangerous file types.",
        category="File System",
        common_languages=("PHP", "Python", "Java"),
        mitre_url="https://cwe.mitre.org/data/definitions/434.html",
    ),
    "CWE-862": CWEEntry(
        cwe_id="CWE-862", name="Missing Authorization",
        description="Software does not perform authorization check.",
        category="Access Control",
        common_languages=("Python", "Java", "PHP", "JavaScript"),
        mitre_url="https://cwe.mitre.org/data/definitions/862.html",
    ),
    "CWE-476": CWEEntry(
        cwe_id="CWE-476", name="NULL Pointer Dereference",
        description="Software dereferences a pointer that is NULL.",
        category="Memory Safety",
        common_languages=("C", "C++", "Java"),
        mitre_url="https://cwe.mitre.org/data/definitions/476.html",
    ),
    "CWE-502": CWEEntry(
        cwe_id="CWE-502", name="Deserialization of Untrusted Data",
        description="Software deserializes untrusted data without verification.",
        category="Input Validation",
        common_languages=("Java", "Python", "PHP", "Ruby"),
        mitre_url="https://cwe.mitre.org/data/definitions/502.html",
    ),
    "CWE-190": CWEEntry(
        cwe_id="CWE-190", name="Integer Overflow or Wraparound",
        description="Calculation produces an integer overflow.",
        category="Numeric Errors",
        common_languages=("C", "C++", "Java"),
        mitre_url="https://cwe.mitre.org/data/definitions/190.html",
    ),
    "CWE-918": CWEEntry(
        cwe_id="CWE-918", name="Server-Side Request Forgery (SSRF)",
        description="Server-side request to attacker-controlled URL.",
        category="Injection",
        common_languages=("Python", "Java", "JavaScript", "PHP"),
        mitre_url="https://cwe.mitre.org/data/definitions/918.html",
    ),
}


class CWEMapper:
    """
    Maps vulnerability data to CWE taxonomy entries.

    Supports exact matching, heuristic inference from code patterns,
    and NVD-based CWE resolution.
    """

    # Common code patterns → likely CWE mappings
    PATTERN_HEURISTICS: list[tuple[str, str]] = [
        (r"execute\s*\(.*f['\"]", "CWE-89"),             # f-string in SQL execute
        (r"os\.system\s*\(", "CWE-78"),                   # os.system calls
        (r"subprocess\.\w+\s*\(.*shell\s*=\s*True", "CWE-78"),  # shell=True
        (r"innerHTML\s*=", "CWE-79"),                     # innerHTML assignment
        (r"dangerouslySetInnerHTML", "CWE-79"),           # React XSS
        (r"pickle\.loads?\s*\(", "CWE-502"),              # Pickle deserialization
        (r"yaml\.load\s*\((?!.*Loader)", "CWE-502"),     # Unsafe YAML load
        (r"os\.path\.join\s*\(.*request", "CWE-22"),      # Path traversal
        (r"open\s*\(.*request", "CWE-22"),                # File open with user input
        (r"eval\s*\(", "CWE-94"),                         # Code injection via eval
        (r"requests\.get\s*\(.*request\.", "CWE-918"),    # SSRF pattern
        (r"malloc\s*\(.*\*", "CWE-190"),                  # Integer overflow in malloc
        (r"free\s*\((\w+)\).*\n.*\1", "CWE-416"),          # Potential use-after-free
        (r"strcpy\s*\(", "CWE-787"),                      # Unsafe strcpy
        (r"sprintf\s*\(", "CWE-787"),                     # Unsafe sprintf
    ]

    def __init__(self) -> None:
        self.taxonomy = CWE_TAXONOMY
        self._compiled_patterns = [
            (re.compile(pattern, re.IGNORECASE | re.MULTILINE), cwe_id)
            for pattern, cwe_id in self.PATTERN_HEURISTICS
        ]

    def lookup(self, cwe_id: str) -> Optional[CWEEntry]:
        """Look up a CWE entry by ID."""
        # Normalize format
        if not cwe_id.startswith("CWE-"):
            cwe_id = f"CWE-{cwe_id}"
        return self.taxonomy.get(cwe_id)

    def infer_from_code(self, vulnerable_code: str) -> list[str]:
        """
        Infer likely CWE IDs from code patterns.

        This is a heuristic approach used to augment missing CWE
        mappings in training data.
        """
        matches: list[str] = []
        for pattern, cwe_id in self._compiled_patterns:
            if pattern.search(vulnerable_code):
                if cwe_id not in matches:
                    matches.append(cwe_id)
        return matches

    def severity_from_cvss(self, cvss_score: float) -> Severity:
        """Convert CVSS score to severity level."""
        if cvss_score == 0.0:
            return Severity.NONE
        elif cvss_score < 4.0:
            return Severity.LOW
        elif cvss_score < 7.0:
            return Severity.MEDIUM
        elif cvss_score < 9.0:
            return Severity.HIGH
        else:
            return Severity.CRITICAL

    def generate_structured_finding(
        self,
        cwe_id: str,
        location: str,
        vulnerable_code: str,
        secure_alternative: str,
        explanation: str,
        cvss_score: float = 0.0,
        cvss_vector: str = "",
    ) -> dict[str, str | list[str]]:
        """
        Generate a PurpCode-style structured finding.

        Every model output must ground its findings in CWE taxonomy
        with CVSS 4.0 scoring, as specified in Section 4.3 of the spec.
        """
        entry = self.lookup(cwe_id)
        name = entry.name if entry else "Unknown"
        mitre_url = entry.mitre_url if entry else ""

        return {
            "title": f"{name} ({cwe_id})",
            "cwe": cwe_id,
            "cwe_name": name,
            "severity": self.severity_from_cvss(cvss_score).value,
            "cvss_score": str(cvss_score),
            "cvss_vector": cvss_vector,
            "location": location,
            "vulnerable_code": vulnerable_code,
            "secure_alternative": secure_alternative,
            "explanation": explanation,
            "references": [mitre_url] if mitre_url else [],
        }
