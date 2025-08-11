# Copyright (c) 2025 Jascha Wanger / Tarnover, LLC
# SPDX-License-Identifier: MIT
#
# This file is part of the VectorSmuggle project.
# You may obtain a copy of the license at https://opensource.org/licenses/MIT

"""Detection avoidance techniques for evading DLP, content analysis, and anomaly detection.

Note: This module uses the standard random module for content obfuscation and evasion techniques.
This is intentional and not for cryptographic purposes, hence the nosec comments.
"""

import hashlib
import logging
import random
import re
from typing import Any

import numpy as np


class DetectionAvoidance:
    """Implements techniques to avoid detection by DLP systems and content analysis."""

    def __init__(
        self,
        dlp_keywords: list[str] | None = None,
        transformation_strength: float = 0.3,
        statistical_noise_level: float = 0.1,
        content_obfuscation_ratio: float = 0.2,
        random_state: np.random.RandomState | None = None
    ):
        """
        Initialize detection avoidance system.

        Args:
            dlp_keywords: List of DLP keywords to avoid
            transformation_strength: Strength of content transformations (0.0-1.0)
            statistical_noise_level: Level of statistical noise to inject
            content_obfuscation_ratio: Ratio of content to obfuscate
            random_state: Seeded RandomState for deterministic operations
        """
        if dlp_keywords is None:
            dlp_keywords = self._get_default_dlp_keywords()
        self.dlp_keywords = dlp_keywords
        self.transformation_strength = transformation_strength
        self.statistical_noise_level = statistical_noise_level
        self.content_obfuscation_ratio = content_obfuscation_ratio
        self.random_state = random_state or np.random.RandomState()

        self.detection_patterns = self._compile_detection_patterns()
        self.transformation_cache: dict[str, str] = {}
        self.logger = logging.getLogger(__name__)

    def _get_default_dlp_keywords(self) -> list[str]:
        """Get default DLP keywords to avoid."""
        return [
            # Financial
            "credit card", "ssn", "social security", "bank account", "routing number",
            "financial", "salary", "revenue", "profit", "budget", "invoice",
            "payment", "transaction", "account number", "tax id", "ein",
            "wire transfer", "ach", "swift", "iban", "cryptocurrency", "bitcoin",

            # Personal Information
            "personal", "confidential", "private", "sensitive", "classified",
            "employee", "customer", "client", "patient", "medical",
            "pii", "phi", "hipaa", "gdpr", "personal data", "biometric",
            "driver license", "passport", "birth date", "maiden name",

            # Technical
            "password", "api key", "secret", "token", "credential", "authentication",
            "database", "server", "network", "security", "vulnerability",
            "encryption key", "private key", "certificate", "oauth", "jwt",
            "access token", "session id", "hash", "salt", "cipher",

            # Business
            "proprietary", "trade secret", "intellectual property", "merger",
            "acquisition", "strategic", "competitive", "internal", "restricted",
            "board meeting", "executive", "c-level", "ceo", "cfo", "cto",
            "insider", "material information", "earnings", "forecast",

            # Legal/Compliance
            "legal", "compliance", "regulation", "audit", "investigation",
            "lawsuit", "contract", "agreement", "license", "copyright",
            "subpoena", "litigation", "settlement", "nda", "non-disclosure",
            "attorney", "counsel", "privileged", "discovery"
        ]

    def _compile_detection_patterns(self) -> list[re.Pattern]:
        """Compile regex patterns for common detection signatures."""
        patterns = []

        # Common data patterns
        pattern_strings = [
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
            r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',  # Credit card
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
            r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b',  # IP address
            r'\b[A-F0-9]{32}\b',  # MD5 hash
            r'\b[A-F0-9]{40}\b',  # SHA1 hash
            r'\b[A-F0-9]{64}\b',  # SHA256 hash
            r'\b\d{9}\b',  # Tax ID/EIN
            r'\b[A-Z]{2}\d{2}[A-Z0-9]{4}\d{7}([A-Z0-9]?){0,16}\b',  # IBAN
            r'\b[A-Z]{6}[A-Z0-9]{2}([A-Z0-9]{3})?\b',  # SWIFT/BIC
            r'\b\d{10,12}\b',  # Phone numbers
            r'\b[A-Z]\d{8}\b',  # Passport pattern
            r'\b[A-Z]\d{7}\b',  # Driver license pattern
            r'\bmysql://[^\s]+\b',  # Database connection strings
            r'\bpostgres://[^\s]+\b',
            r'\bmongodb://[^\s]+\b',
            r'\bBearer\s+[A-Za-z0-9\-._~+/]+=*\b',  # Bearer tokens
            r'\bBasic\s+[A-Za-z0-9+/]+=*\b',  # Basic auth
            r'\bsk-[A-Za-z0-9]{48}\b',  # OpenAI API keys
            r'\bAKIA[0-9A-Z]{16}\b',  # AWS access keys
        ]

        for pattern_str in pattern_strings:
            try:
                patterns.append(re.compile(pattern_str, re.IGNORECASE))
            except re.error as e:
                self.logger.warning(f"Invalid regex pattern {pattern_str}: {e}")

        return patterns

    def avoid_dlp_keywords(self, text: str) -> str:
        """
        Transform text to avoid DLP keyword detection.

        Args:
            text: Input text to transform

        Returns:
            Transformed text with DLP keywords avoided
        """
        if not text:
            return text

        # Check cache first
        text_hash = hashlib.md5(text.encode(), usedforsecurity=False).hexdigest()
        if text_hash in self.transformation_cache:
            return self.transformation_cache[text_hash]

        transformed = text

        for keyword in self.dlp_keywords:
            if keyword.lower() in transformed.lower():
                # Apply various transformation techniques
                transformed = self._transform_keyword(transformed, keyword)

        # Cache the result
        self.transformation_cache[text_hash] = transformed

        return transformed

    def _transform_keyword(self, text: str, keyword: str) -> str:
        """Apply transformations to avoid specific keyword detection."""
        transformations = [
            self._character_substitution,
            self._word_splitting,
            self._synonym_replacement,
            self._context_obfuscation,
            self._unicode_normalization
        ]

        # Apply random transformation based on strength
        if random.random() < self.transformation_strength:  # nosec B311
            transformation = random.choice(transformations)  # nosec B311
            text = transformation(text, keyword)

        return text

    def _character_substitution(self, text: str, keyword: str) -> str:
        """Replace characters with similar-looking alternatives."""
        substitutions = {
            'a': ['@', 'α', 'а', 'ɑ', 'ａ', 'ä', 'à', 'á', 'â', 'ã'],  # Cyrillic 'а', Greek alpha, etc.
            'e': ['3', 'е', 'ε', 'ｅ', 'é', 'è', 'ê', 'ë', '€', 'ē'],  # Cyrillic 'е', Greek epsilon
            'i': ['1', '!', 'і', 'ι', 'ｉ', 'í', 'ì', 'î', 'ï', 'ī', '|'],  # Cyrillic 'і', Greek iota
            'o': ['0', 'о', 'ο', 'ｏ', 'ó', 'ò', 'ô', 'õ', 'ö', 'ø', '°'],  # Cyrillic 'о', Greek omicron
            's': ['$', '5', 'ѕ', 'ｓ', 'š', 'ś', 'ş', 'ș', '§'],  # Cyrillic 'ѕ'
            't': ['7', '+', 'т', 'τ', 'ｔ', 'ť', 'ţ', 'ț', '†'],  # Cyrillic 'т', Greek tau
            'c': ['с', 'ç', 'ć', 'č', 'ĉ', 'ċ', 'ｃ', '©'],  # Cyrillic 'с'
            'p': ['р', 'ρ', 'ｐ', 'þ'],  # Cyrillic 'р', Greek rho
            'x': ['х', 'χ', 'ｘ', '×'],  # Cyrillic 'х', Greek chi
            'y': ['у', 'γ', 'ｙ', 'ý', 'ÿ', 'ŷ'],  # Cyrillic 'у', Greek gamma
            'n': ['ñ', 'ń', 'ň', 'ņ', 'ｎ'],
            'u': ['ü', 'ú', 'ù', 'û', 'ū', 'ｕ'],
            'l': ['ł', 'ľ', 'ļ', 'ｌ', '|', '1'],
            'r': ['ř', 'ŕ', 'ｒ'],
            'd': ['ď', 'đ', 'ｄ'],
            'g': ['ğ', 'ģ', 'ｇ'],
            'h': ['ħ', 'ｈ'],
            'j': ['ĵ', 'ｊ'],
            'k': ['ķ', 'ｋ'],
            'm': ['ｍ'],
            'v': ['ｖ'],
            'w': ['ｗ'],
            'z': ['ž', 'ź', 'ż', 'ｚ']
        }

        # Find keyword occurrences (case insensitive)
        pattern = re.compile(re.escape(keyword), re.IGNORECASE)

        def replace_func(match):
            matched_text = match.group()
            result = ""

            for char in matched_text:
                if char.lower() in substitutions and random.random() < 0.3:
                    # 30% chance to substitute each character
                    replacement = random.choice(substitutions[char.lower()])
                    result += replacement if char.islower() else replacement.upper()
                else:
                    result += char

            return result

        return pattern.sub(replace_func, text)

    def _word_splitting(self, text: str, keyword: str) -> str:
        """Split words with invisible characters or spaces."""
        # Find keyword occurrences
        pattern = re.compile(re.escape(keyword), re.IGNORECASE)

        def replace_func(match):
            matched_text = match.group()

            if len(matched_text) > 2:
                # Enhanced zero-width character insertion
                split_chars = [
                    '\u200B',  # Zero-width space
                    '\u200C',  # Zero-width non-joiner
                    '\u200D',  # Zero-width joiner
                    '\u2009',  # Thin space
                    '\u00A0',  # Non-breaking space
                    '\u2060',  # Word joiner
                    '\uFEFF',  # Zero-width no-break space
                ]

                # Multiple insertion strategy for better obfuscation
                result = matched_text
                if random.random() < 0.7:  # 70% chance to apply splitting
                    # Insert 1-3 zero-width characters at different positions
                    num_insertions = random.randint(1, min(3, len(matched_text) - 1))
                    positions = sorted(random.sample(range(1, len(matched_text)), num_insertions))

                    # Insert from right to left to maintain position indices
                    for pos in reversed(positions):
                        split_char = random.choice(split_chars)
                        result = result[:pos] + split_char + result[pos:]

            return result

        return pattern.sub(replace_func, text)

    def _synonym_replacement(self, text: str, keyword: str) -> str:
        """Replace keywords with synonyms or related terms."""
        synonyms = {
            'confidential': ['private', 'restricted', 'internal', 'sensitive'],
            'secret': ['private', 'hidden', 'classified', 'covert'],
            'password': ['passcode', 'credentials', 'access code', 'login'],
            'financial': ['monetary', 'fiscal', 'economic', 'pecuniary'],
            'customer': ['client', 'patron', 'consumer', 'user'],
            'employee': ['worker', 'staff', 'personnel', 'team member'],
            'database': ['data store', 'repository', 'data warehouse', 'records'],
            'security': ['protection', 'safety', 'defense', 'safeguard']
        }

        keyword_lower = keyword.lower()
        if keyword_lower in synonyms:
            synonym = random.choice(synonyms[keyword_lower])
            pattern = re.compile(re.escape(keyword), re.IGNORECASE)

            def replace_func(match):
                matched_text = match.group()
                if matched_text.isupper():
                    return synonym.upper()
                elif matched_text.istitle():
                    return synonym.title()
                else:
                    return synonym

            return pattern.sub(replace_func, text)

        return text

    def _context_obfuscation(self, text: str, keyword: str) -> str:
        """Add context that changes the meaning of sensitive terms."""
        obfuscation_prefixes = [
            "example of ", "sample ", "test ", "demo ", "mock ", "fake ",
            "hypothetical ", "simulated ", "training ", "educational "
        ]

        pattern = re.compile(r'\b' + re.escape(keyword) + r'\b', re.IGNORECASE)

        def replace_func(match):
            if random.random() < 0.3:  # 30% chance to add prefix
                prefix = random.choice(obfuscation_prefixes)
                return prefix + match.group()
            return match.group()

        return pattern.sub(replace_func, text)

    def _unicode_normalization(self, text: str, keyword: str) -> str:
        """Use Unicode normalization to create visually similar text."""
        # This is a simplified version - in practice, you'd use more sophisticated
        # Unicode normalization techniques

        unicode_alternatives = {
            'a': 'а',  # Cyrillic
            'e': 'е',  # Cyrillic
            'o': 'о',  # Cyrillic
            'p': 'р',  # Cyrillic
            'c': 'с',  # Cyrillic
            'x': 'х',  # Cyrillic
        }

        pattern = re.compile(re.escape(keyword), re.IGNORECASE)

        def replace_func(match):
            matched_text = match.group()
            result = ""

            for char in matched_text:
                if char.lower() in unicode_alternatives and random.random() < 0.2:
                    result += unicode_alternatives[char.lower()]
                else:
                    result += char

            return result

        return pattern.sub(replace_func, text)

    def transform_content_signatures(self, content: str) -> str:
        """Transform content to avoid signature-based detection."""
        if not content:
            return content

        transformed = content

        # Apply pattern-based transformations
        for pattern in self.detection_patterns:
            matches = pattern.findall(transformed)
            for match in matches:
                if random.random() < self.content_obfuscation_ratio:
                    # Obfuscate detected patterns
                    obfuscated = self._obfuscate_pattern(match)
                    transformed = transformed.replace(match, obfuscated, 1)

        return transformed

    def _obfuscate_pattern(self, pattern_match: str) -> str:
        """Obfuscate a detected pattern."""
        # Simple obfuscation - replace some characters with similar ones
        obfuscated = ""
        for char in pattern_match:
            if char.isdigit() and random.random() < 0.3:
                # Replace some digits with letters
                obfuscated += random.choice(['O', 'l', 'I', 'S', 'G'])
            elif char == '@' and random.random() < 0.5:
                obfuscated += '[at]'
            elif char == '.' and random.random() < 0.3:
                obfuscated += '[dot]'
            else:
                obfuscated += char

        return obfuscated

    def inject_statistical_noise(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Inject statistical noise to avoid anomaly detection.

        Args:
            embeddings: Input embeddings array

        Returns:
            Embeddings with statistical noise injected
        """
        if embeddings.size == 0:
            return embeddings

        # Calculate statistics of original embeddings
        std = np.std(embeddings, axis=0)

        # Generate noise that maintains statistical properties
        noise_scale = self.statistical_noise_level

        # Add correlated noise to maintain embedding structure
        noise = np.random.normal(0, noise_scale * std, embeddings.shape)

        # Apply noise selectively to avoid detection
        noise_mask = np.random.random(embeddings.shape) < self.content_obfuscation_ratio
        noise = noise * noise_mask

        noisy_embeddings = embeddings + noise

        self.logger.debug(f"Injected statistical noise to {embeddings.shape[0]} embeddings")
        return noisy_embeddings

    def create_decoy_patterns(self, num_decoys: int = 10) -> list[str]:
        """
        Create decoy patterns that look suspicious but are harmless.

        Args:
            num_decoys: Number of decoy patterns to create

        Returns:
            List of decoy pattern strings
        """
        decoy_templates = [
            "Test data for {purpose} validation",
            "Sample {data_type} for training purposes",
            "Mock {item} used in development",
            "Dummy {content} for testing only",
            "Placeholder {element} - not real data",
            "Example {object} for documentation",
            "Simulated {thing} for demo purposes",
            "Training {material} - educational use only"
        ]

        purposes = ["security", "compliance", "audit", "validation", "verification"]
        data_types = ["records", "entries", "information", "content", "data"]
        items = ["credentials", "accounts", "profiles", "documents", "files"]

        decoys = []
        for _ in range(num_decoys):
            template = random.choice(decoy_templates)

            # Fill in template variables
            if "{purpose}" in template:
                decoy = template.format(purpose=random.choice(purposes))
            elif "{data_type}" in template:
                decoy = template.format(data_type=random.choice(data_types))
            elif "{item}" in template:
                decoy = template.format(item=random.choice(items))
            elif "{content}" in template:
                decoy = template.format(content=random.choice(data_types))
            elif "{element}" in template:
                decoy = template.format(element=random.choice(items))
            elif "{object}" in template:
                decoy = template.format(object=random.choice(data_types))
            elif "{thing}" in template:
                decoy = template.format(thing=random.choice(items))
            elif "{material}" in template:
                decoy = template.format(material=random.choice(data_types))
            else:
                decoy = template

            decoys.append(decoy)

        return decoys

    def evade_anomaly_detection(self, data_points: list[Any]) -> list[Any]:
        """
        Modify data points to evade statistical anomaly detection.

        Args:
            data_points: List of data points to modify

        Returns:
            Modified data points that appear more normal
        """
        if len(data_points) < 2:
            return data_points

        # Convert to numpy array for easier manipulation
        if isinstance(data_points[0], int | float):
            data_array = np.array(data_points)

            # Calculate statistics
            mean = np.mean(data_array)
            std = np.std(data_array)

            # Identify potential outliers (beyond 2 standard deviations)
            outlier_threshold = 2 * std
            outliers = np.abs(data_array - mean) > outlier_threshold

            # Adjust outliers to be within normal range
            adjusted_data = data_array.copy()
            for i, is_outlier in enumerate(outliers):
                if is_outlier and random.random() < 0.7:  # 70% chance to adjust
                    # Move outlier closer to mean
                    if data_array[i] > mean:
                        adjusted_data[i] = mean + outlier_threshold * 0.8
                    else:
                        adjusted_data[i] = mean - outlier_threshold * 0.8

            return adjusted_data.tolist()

        # For non-numeric data, return as-is
        return data_points

    def assess_detection_risk(self, content: str) -> dict[str, Any]:
        """
        Assess the detection risk of given content.

        Args:
            content: Content to assess

        Returns:
            Risk assessment dictionary
        """
        risk_score = 0.0
        detected_patterns = []
        detected_keywords = []

        # Check for DLP keywords
        for keyword in self.dlp_keywords:
            if keyword.lower() in content.lower():
                detected_keywords.append(keyword)
                risk_score += 0.1

        # Check for detection patterns
        for pattern in self.detection_patterns:
            matches = pattern.findall(content)
            if matches:
                detected_patterns.extend(matches)
                risk_score += 0.2 * len(matches)

        # Normalize risk score
        risk_score = min(1.0, risk_score)

        # Categorize risk level
        if risk_score < 0.2:
            risk_level = "low"
        elif risk_score < 0.5:
            risk_level = "medium"
        elif risk_score < 0.8:
            risk_level = "high"
        else:
            risk_level = "critical"

        assessment = {
            "risk_score": risk_score,
            "risk_level": risk_level,
            "detected_keywords": detected_keywords,
            "detected_patterns": detected_patterns,
            "content_length": len(content),
            "keyword_density": len(detected_keywords) / len(content.split()) if content.split() else 0
        }

        return assessment

    def get_detection_statistics(self) -> dict[str, Any]:
        """Get statistics about detection avoidance activities."""
        stats = {
            "dlp_keywords_count": len(self.dlp_keywords),
            "detection_patterns_count": len(self.detection_patterns),
            "transformation_cache_size": len(self.transformation_cache),
            "transformation_strength": self.transformation_strength,
            "statistical_noise_level": self.statistical_noise_level,
            "content_obfuscation_ratio": self.content_obfuscation_ratio
        }

        return stats

    def clear_transformation_cache(self) -> None:
        """Clear the transformation cache."""
        self.transformation_cache.clear()
        self.logger.info("Cleared transformation cache")
