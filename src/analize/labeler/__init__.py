"""
Outcome labeling module for Analize.

Labels signal records with:
- Time-windowed outcomes (1m, 5m, 15m, 60m, 240m)
- Maximum Favorable Excursion (MFE)
- Maximum Adverse Excursion (MAE)
- TP/SL hit times
- Exit reasons
"""

from analize.labeler.outcome_labeler import OutcomeLabeler

__all__ = ["OutcomeLabeler"]
