"""Unit conversion utility using pint."""

from typing import Optional
import pint


class UnitConverter:
    """Convert and normalize units for answer comparison."""

    def __init__(self):
        """Initialize with pint unit registry."""
        self.ureg = pint.UnitRegistry()

    def normalize(self, value: float, from_unit: str, to_unit: str) -> Optional[float]:
        """Convert value from one unit to another.

        Args:
            value: Numeric value
            from_unit: Source unit string
            to_unit: Target unit string

        Returns:
            Converted value, or None if conversion failed
        """
        if not from_unit or not to_unit:
            return value

        if from_unit == to_unit:
            return value

        try:
            # Create quantities with units
            from_quantity = value * self.ureg(from_unit)
            # Convert to target unit
            converted = from_quantity.to(to_unit)
            return float(converted.magnitude)
        except Exception as e:
            # Conversion failed (incompatible units, unknown unit, etc.)
            return None

    def are_compatible(self, unit1: str, unit2: str) -> bool:
        """Check if two units are convertible (have same dimensionality).

        Args:
            unit1: First unit string
            unit2: Second unit string

        Returns:
            True if units are compatible, False otherwise
        """
        if not unit1 or not unit2:
            return True

        if unit1 == unit2:
            return True

        try:
            quantity1 = 1 * self.ureg(unit1)
            quantity2 = 1 * self.ureg(unit2)
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
