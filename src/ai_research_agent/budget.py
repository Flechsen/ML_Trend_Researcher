from dataclasses import dataclass, field

PRICING: dict[str, dict[str, float]] = {
    # USD per token
    "claude-sonnet-4-6":      {"in": 3.00e-6, "out": 15.00e-6},
    "claude-haiku-4-5":       {"in": 1.00e-6, "out":  5.00e-6},
    "text-embedding-3-small": {"in": 0.02e-6, "out":  0.0     },
}


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class Budget:
    cap_usd: float
    spent: float = 0.0
    calls: list[tuple[str, str, int, int, float]] = field(default_factory=list)

    def charge(self, stage: str, model: str, in_tok: int, out_tok: int) -> None:
        p = PRICING[model]  # KeyError on unknown model
        cost = in_tok * p["in"] + out_tok * p["out"]
        self.spent += cost
        self.calls.append((stage, model, in_tok, out_tok, cost))
        if self.spent > self.cap_usd:
            raise BudgetExceeded(
                f"Budget exceeded: spent ${self.spent:.4f} > cap ${self.cap_usd:.4f}"
            )

    def report(self) -> str:
        lines = [f"Total: ${self.spent:.4f} ({len(self.calls)} calls)"]
        for stage, model, in_t, out_t, cost in self.calls:
            lines.append(f"  {stage:12s} {model:24s} in={in_t} out={out_t} ${cost:.4f}")
        return "\n".join(lines)
