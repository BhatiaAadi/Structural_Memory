"""
evaluator — DSA Code Evaluation Pipeline
=========================================
Analyzes student code via tree-sitter AST + LLM rubric evaluation
and scores against the 22-node Skeleton Graph.

Usage:
    from evaluator import evaluate_code
    result = evaluate_code("diana", "q_dijkstra_basic", code_string)
"""

from .evaluator import evaluate_code

__all__ = ["evaluate_code"]
