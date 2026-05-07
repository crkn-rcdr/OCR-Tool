"""
OCR engine package.

Exposes:
    PaddleEngine: OCR engine class
    engine: default initialized engine instance
"""
from .paddle_engine import PaddleEngine

engine = PaddleEngine()

__all__ = ["engine", "PaddleEngine"]