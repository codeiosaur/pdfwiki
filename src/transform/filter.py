import re
from typing import List

# Countries, regions, and geographic entities to reject
COUNTRIES_REGIONS = {
    "Afghanistan", "Albania", "Algeria", "Andorra", "Angola", "Argentina", "Armenia",
    "Austria", "Azerbaijan", "Bahamas", "Bahrain", "Bangladesh", "Barbados", "Belarus",
    "Belgium", "Belize", "Benin", "Bhutan", "Bolivia", "Bosnia", "Botswana", "Brazil",
    "Brunei", "Bulgaria", "Burkina", "Burma", "Burundi", "Cambodia", "Cameroon", "Canada",
    "Cape", "Central African", "Chad", "Chile", "China", "Colombia", "Comoros", "Congo",
    "Costa Rica", "Croatia", "Cuba", "Cyprus", "Czech", "Denmark", "Djibouti", "Dominica",
    "Dominican", "Ecuador", "Egypt", "El Salvador", "Equatorial Guinea", "Eritrea", "Estonia",
    "Eswatini", "Ethiopia", "Fiji", "Finland", "France", "Gabon", "Gambia", "Georgia",
    "Germany", "Ghana", "Greece", "Grenada", "Guatemala", "Guinea", "Guinea-Bissau", "Guyana",
    "Haiti", "Honduras", "Hungary", "Iceland", "India", "Indonesia", "Iran", "Iraq", "Ireland",
    "Israel", "Italy", "Jamaica", "Japan", "Jordan", "Kazakhstan", "Kenya", "Kiribati",
    "Kosovo", "Kuwait", "Kyrgyzstan", "Laos", "Latvia", "Lebanon", "Lesotho", "Liberia",
    "Libya", "Liechtenstein", "Lithuania", "Luxembourg", "Madagascar", "Malawi", "Malaysia",
    "Maldives", "Mali", "Malta", "Marshall", "Mauritania", "Mauritius", "Mexico", "Micronesia",
    "Moldova", "Monaco", "Mongolia", "Montenegro", "Morocco", "Mozambique", "Namibia", "Nauru",
    "Nepal", "Netherlands", "New Zealand", "Nicaragua", "Niger", "Nigeria", "North", "Norway",
    "Oman", "Pakistan", "Palau", "Palestine", "Panama", "Papua", "Paraguay", "Peru",
    "Philippines", "Poland", "Portugal", "Qatar", "Republic", "Romania", "Russia", "Rwanda",
    "Saint", "Samoa", "San Marino", "Sao Tome", "Saudi Arabia", "Senegal", "Serbia",
    "Seychelles", "Sierra Leone", "Singapore", "Slovakia", "Slovenia", "Solomon", "Somalia",
    "South Africa", "South Sudan", "Spain", "Sri Lanka", "Sudan", "Suriname", "Sweden",
    "Switzerland", "Syria", "Taiwan", "Tajikistan", "Tanzania", "Thailand", "Timor", 
    "Togo", "Tonga", "Trinidad", "Tunisia", "Turkey", "Turkmenistan", "Tuvalu", "Uganda",
    "Ukraine", "United Arab", "United Kingdom", "United States", "Uruguay", "Uzbekistan",
    "Vanuatu", "Vatican", "Venezuela", "Vietnam", "Yemen", "Zambia", "Zimbabwe"
}

# Common nouns that end in 'ing' (exceptions to verb rejection)
# Be conservative: only include words that are primarily used as nouns, not verb forms
COMMON_ING_NOUNS = {
    "string", "king", "ring", "thing", "wing", "spring",
    "morning", "evening", "meeting", "greeting", "warning",
    "meaning", "feeling", "building", "training", "learning", "testing",
    "boxing", "mining", "dining", "timing", "bearing", "hearing",
    "feeling", "being", "beginning", "ending"
}

# Vague descriptors that typically indicate non-concepts
VAGUE_DESCRIPTORS = {
    "Example", "Case", "Scenario", "Impact", "Effect", "Overview", "Summary",
    "Introduction", "Conclusion", "Discussion", "Analysis", "Review", "History"
}

IMPERATIVE_STARTERS = {
    "make", "record", "create", "do", "prepare", "write", "provide",
    "give", "list", "show", "describe", "conduct", "submit", "search",
    "locate", "analyze", "identify", "determine", "find", "calculate",
    "compute", "apply", "solve", "examine", "check", "verify", "state",
}

# Adjectives that signal a predicate phrase rather than a real concept name.
# e.g. "Higher Profit", "True Ownership", "Better Method" are fragments of
# sentences, not concepts a student would look up.
ADJECTIVE_FIRST_STARTERS = {
    "higher", "lower", "high", "low", "greater", "lesser",
    "true", "false", "real", "proper", "correct", "incorrect",
    "good", "bad", "better", "worse", "best", "worst",
    "large", "small", "larger", "smaller", "additional", "extra",
    "old", "new", "same", "different", "full", "partial",
    "similar", "common", "general", "specific", "typical",
    "significant", "important", "accurate", "inaccurate",
}

# Resource-type suffixes that indicate a website/platform reference rather than a concept.
_RESOURCE_SUFFIXES = {"website", "portal", "page", "site", "platform", "tool"}

# Internal pipeline/process labels that should never surface as user-facing concepts.
_INTERNAL_CONCEPT_EXACT = {
    "canonicalize concept names",
    "source chunk id",
    "chunk id",
    "json array",
    "existing facts",
}


def is_valid_concept(name: str) -> bool:
    """
    Determine if a concept name is likely a reusable knowledge concept.
    
    Rejects:
    - Concepts containing years (e.g., 2014, 2021)
    - Long phrases (> 6 words)
    - Possessive proper nouns (e.g., "France's", "U S")
    - Country/region names
    - Verbs (words ending in 'ing' unless common nouns)
    - Vague descriptors (Example, Case, Scenario, etc.)
    
    Keeps:
    - Short noun phrases (1-4 words)
    - Generalizable ideas
    
    Args:
        name: Concept name to validate
        
    Returns:
        True if concept is valid, False otherwise
    """
    if not name or not isinstance(name, str):
        return False
    
    name = name.strip()
    
    # Rule 1: Reject if contains year (1900-2099)
    if re.search(r'\b(19|20)\d{2}\b', name):
        return False
    
    # Rule 2: Reject if too long (> 6 words)
    words = name.split()
    if len(words) > 6:
        return False
    
    # Rule 3: Reject possessive proper nouns (e.g., "France's", "U S")
    if re.search(r"'s\b", name):  # Possessive 's
        return False
    if re.search(r"\b[A-Z]\s+[A-Z]\b", name):  # Pattern like "U S", "U K"
        return False
    
    # Rule 4: Reject country/region names
    name_lower = name.lower()
    for country in COUNTRIES_REGIONS:
        if country.lower() in name_lower:
            return False
    
    # Rule 5: Reject verbs (words ending in 'ing' unless common nouns)
    for word in words:
        word_lower = word.rstrip('.,!?;:').lower()  # Remove trailing punctuation
        if word_lower.endswith('ing') and word_lower not in COMMON_ING_NOUNS:
            return False
    
    # Rule 6: Reject vague descriptors at start
    for descriptor in VAGUE_DESCRIPTORS:
        if name.startswith(descriptor):
            return False

    # Rule 7: Reject instruction-derived imperative phrases.
    first_word = words[0].rstrip('.,!?;:').lower() if words else ""
    if first_word in IMPERATIVE_STARTERS:
        return False
    
    # Rule 8: Reject single-word generic/junk concepts that lack specificity.
    GENERIC_JUNK_WORDS = {
        "error", "cost", "value", "other", "information", "data",
        "result", "output", "input", "process", "item", "thing",
        "avg", "note", "summary", "description"
    }
    if len(words) == 1 and name.lower() in GENERIC_JUNK_WORDS:
        return False

    # Rule 9: Reject adjective-first phrases — these are sentence fragments,
    # not concept names (e.g. "Higher Profit", "True Ownership").
    if first_word in ADJECTIVE_FIRST_STARTERS:
        return False

    # Rule 10: Reject concepts that end with a resource-type suffix
    # (e.g. "SEC Website", "Company Portal") — these are navigation references.
    last_word = words[-1].rstrip(".,!?;:").lower()
    if last_word in _RESOURCE_SUFFIXES:
        return False

    # Rule 11: Reject internal pipeline/process terms.
    normalized = re.sub(r"\s+", " ", name.lower()).strip()
    if normalized in _INTERNAL_CONCEPT_EXACT:
        return False
    if "canonicalize" in normalized and "concept" in normalized:
        return False

    return True


def filter_concepts(facts: List) -> List:
    """
    Filter facts by concept validity.
    
    Args:
        facts: List of Fact objects with .concept attribute
        
    Returns:
        Filtered list of facts with only valid concepts
    """
    return [fact for fact in facts if is_valid_concept(fact.concept)]
