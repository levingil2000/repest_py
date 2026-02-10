"""
RepEst - Replicate Estimation for Survey Data

A Python package for analyzing international large-scale assessment data
(PISA, PIAAC, TALIS, etc.) with replicate weights and plausible values.
"""

from .core import RepEst, SurveyParameters, SURVEY_CONFIGS
from .estimation import EstimationFunctions
from .cleaner import DataCleaner

__version__ = "1.0.0"
__all__ = [
    "RepEst",
    "SurveyParameters", 
    "SURVEY_CONFIGS",
    "EstimationFunctions",
    "DataCleaner",
]