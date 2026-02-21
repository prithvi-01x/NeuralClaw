"""
agent/ — NeuralClaw Agent Core

Public API:
    from agent import Orchestrator, Session, AgentResponse

Component overview:
    Session             Per-user state (memory, plan, trust, cancellation)
    ContextBuilder      Assembles LLM prompt from memory + history + system prompt
    Planner             Decomposes goals into step sequences (autonomous mode)
    Reasoner            Chain-of-thought + self-critique for high-risk actions
    ResponseSynthesizer Formats agent output for Telegram / CLI
    Orchestrator        Central loop: observe → think → act → reflect
"""

from agent.context_builder import ContextBuilder
from agent.orchestrator import Orchestrator
from agent.planner import Planner, PlanResult, RecoveryResult
from agent.reasoner import EvalVerdict, Reasoner
from agent.response_synthesizer import AgentResponse, ResponseKind, ResponseSynthesizer
from agent.session import ActivePlan, PlanStep, Session

__all__ = [
    "Orchestrator",
    "Session",
    "AgentResponse",
    "ContextBuilder",
    "Planner",
    "Reasoner",
    "ResponseSynthesizer",
    "ActivePlan",
    "PlanStep",
    "PlanResult",
    "RecoveryResult",
    "EvalVerdict",
    "ResponseKind",
]