"""Unit conversion utility using pint."""

from typing import Optional
import pint
import re


def parse_latex_unit(latex_unit: str) -> str:
    """Parse LaTeX unit string to pint-compatible format.

    Examples:
        "$\mathrm{J} \mathrm{K}^{-1} \mathrm{~mol}^{-1}$" -> "J / K / mol"
        "$^\circ$" -> "degree"
        "$ \mathrm{~m} / \mathrm{s}$" -> "m / s"
        "$10^{34} \mathrm{~m}^{-3}$" -> This is tricky - extracts just the unit part
    """
    if not latex_unit:
        return ""

    unit = latex_unit.strip()

    # Remove outer $ signs
    if unit.startswith('$') and unit.endswith('$'):
        unit = unit[1:-1]

    # Remove leading/trailing spaces and tildes
    unit = unit.strip()
    unit = unit.replace('~', '')

    # Replace \mathrm{X} with X
    unit = re.sub(r'\\mathrm\{([^}]+)\}', r'\1', unit)

    # Replace \circ with degree (for °)
    unit = unit.replace(r'\circ', 'degree')
    unit = unit.replace('°', 'degree')

    # Handle scientific notation 10^{...} - this is likely a value multiplier, not part of unit
    # Remove it if it appears before the actual unit
    unit = re.sub(r'^10\^{\d+}\s*', '', unit)
    unit = re.sub(r'^10\^{-\d+}\s*', '', unit)
    unit = re.sub(r'^10\^\d+\s*', '', unit)
    unit = re.sub(r'^10\^-\d+\s*', '', unit)

    # Handle ^ notation for exponents
    # ^{-3} -> **(-3) for pint
    unit = re.sub(r'\^{(-?\d+)}', r'**(\1)', unit)
    # ^-3 -> **(-3)
    unit = re.sub(r'\^(-\d+)', r'**(\1)', unit)
    # ^3 -> **3
    unit = re.sub(r'\^(\d+)', r'**\1', unit)

    # Replace / with proper division (pint expects spaces around /)
    # Handle cases like "m/s", "m / s", etc.
    unit = re.sub(r'\s*/\s*', ' / ', unit)
    unit = re.sub(r'([a-zA-Z0-9)])\s*/\s*([a-zA-Z0-9(])', r'\1 / \2', unit)

    # Clean up whitespace
    unit = ' '.join(unit.split())

    # Remove any remaining backslashes
    unit = unit.replace('\\', '')

    return unit


class UnitConverter:
    """Convert and normalize units for answer comparison."""

    def __init__(self):
        """Initialize with pint unit registry."""
        self.ureg = pint.UnitRegistry()

    def normalize(self, value: float, from_unit: str, to_unit: str) -> Optional[float]:
        """Convert value from one unit to another.

        Args:
            value: Numeric value
            from_unit: Source unit string (may contain LaTeX formatting)
            to_unit: Target unit string (may contain LaTeX formatting)

        Returns:
            Converted value, or None if conversion failed
        """
        if not from_unit or not to_unit:
            return value

        # Parse LaTeX formatting from units
        from_unit_clean = parse_latex_unit(from_unit)
        to_unit_clean = parse_latex_unit(to_unit)

        if not from_unit_clean or not to_unit_clean:
            return value

        if from_unit_clean == to_unit_clean:
            return value

        try:
            # Create quantities with units
            from_quantity = value * self.ureg(from_unit_clean)
            # Convert to target unit
            converted = from_quantity.to(to_unit_clean)
            return float(converted.magnitude)
        except Exception as e:
            # Conversion failed (incompatible units, unknown unit, etc.)
            return None

    def are_compatible(self, unit1: str, unit2: str) -> bool:
        """Check if two units are convertible (have same dimensionality).

        Args:
            unit1: First unit string (may contain LaTeX formatting)
            unit2: Second unit string (may contain LaTeX formatting)

        Returns:
            True if units are compatible, False otherwise
        """
        if not unit1 or not unit2:
            return True

        # Parse LaTeX formatting from units
        unit1_clean = parse_latex_unit(unit1)
        unit2_clean = parse_latex_unit(unit2)

        if unit1_clean == unit2_clean:
            return True

        try:
            quantity1 = 1 * self.ureg(unit1_clean)
            quantity2 = 1 * self.ureg(unit2_clean)
            # Check if they have the same dimensionality
            return quantity1.dimensionality == quantity2.dimensionality
        except Exception:
            return False

    def standardize_unit(self, unit: str) -> str:
        """Standardize unit string to a canonical form.

        Args:
            unit: Unit string to standardize

        Returns:
            Standardized unit string
        """
        if not unit:
            return ""

        try:
            # Parse and convert back to string to standardize
            quantity = 1 * self.ureg(unit)
            return str(quantity.units)
        except Exception:
            # Return original if standardization fails
            return unit

    def parse_numeric_with_unit(self, text: str) -> tuple[Optional[float], Optional[str]]:
        """Parse a string containing a numeric value and unit.

        Args:
            text: Text like "1.5 m/s" or "50 atm"

        Returns:
            Tuple of (numeric_value, unit_string), or (None, None) if parsing failed
        """
        if not text:
            return None, None

        text = text.strip()
        parts = text.split()

        if len(parts) < 1:
            return None, None

        # Try to parse first part as number
        try:
            value = float(parts[0])
        except ValueError:
            return None, None

        # Remaining parts are the unit
        unit = " ".join(parts[1:]) if len(parts) > 1 else ""

        return value, unit
