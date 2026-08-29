"""Turn bookkeeping on top of the raw detections.

Detection gives you "T20". A game needs to know whose turn it is, that three
darts make a visit, and - for an X01 game - whether a visit busts.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .detector import Dart

DARTS_PER_TURN = 3


@dataclass
class Turn:
    player: int
    darts: list[Dart] = field(default_factory=list)
    busted: bool = False

    @property
    def points(self) -> int:
        return 0 if self.busted else sum(d.points for d in self.darts)

    @property
    def labels(self) -> list[str]:
        return [d.label for d in self.darts]


@dataclass
class Player:
    name: str
    remaining: int
    history: list[Turn] = field(default_factory=list)

    @property
    def darts_thrown(self) -> int:
        return sum(len(t.darts) for t in self.history)


class Session:
    """Free-scoring by default; pass ``start_score`` for a game of X01."""

    def __init__(
        self,
        players: list[str] | None = None,
        start_score: int = 0,
        double_out: bool = True,
    ) -> None:
        names = players or ["Player 1"]
        self.start_score = start_score
        self.double_out = double_out
        self.players = [Player(n, start_score) for n in names]
        self.current = 0
        self.turn = Turn(player=0)
        self.messages: list[str] = []

    # ------------------------------------------------------------------ #
    @property
    def player(self) -> Player:
        return self.players[self.current]

    @property
    def turn_complete(self) -> bool:
        return self.turn.busted or len(self.turn.darts) >= DARTS_PER_TURN

    def note(self, text: str) -> None:
        self.messages.append(text)
        del self.messages[:-6]

    # ------------------------------------------------------------------ #
    def add_dart(self, dart: Dart) -> None:
        """Register one detected dart against the current visit."""
        if self.turn_complete:
            self.note("Visit already complete - remove the darts")
            return
        self.turn.darts.append(dart)

        if not self.start_score:
            self.note(f"{dart.label} ({dart.points})")
            return

        left = self.player.remaining - self.turn.points
        on_a_double = dart.score.multiplier == 2      # includes the 50 (double bull)
        bust = (
            left < 0
            or (self.double_out and left == 1)
            or (self.double_out and left == 0 and not on_a_double)
        )
        if bust:
            self.turn.busted = True
            self.note(f"{dart.label} - BUST")
        elif left == 0:
            self.note(f"{dart.label} - GAME SHOT, {self.player.name}!")
        else:
            self.note(f"{dart.label} ({dart.points}) -> {left} left")

    def undo_dart(self) -> None:
        """Drop the last dart; if the visit is empty, reopen the previous one."""
        if self.turn.darts:
            removed = self.turn.darts.pop()
            self.turn.busted = False
            self.note(f"undid {removed.label}")
            return

        previous = (self.current - 1) % len(self.players)
        if not self.players[previous].history:
            return
        self.current = previous
        self.turn = self.player.history.pop()
        if self.start_score and not self.turn.busted:
            self.player.remaining += self.turn.points   # give the visit back
        self.turn.busted = False
        if self.turn.darts:
            self.turn.darts.pop()
        self.note("reopened the previous visit")

    def end_turn(self) -> None:
        """Called when the board is cleared, or manually."""
        if not self.turn.darts:
            return
        if self.start_score and not self.turn.busted:
            self.player.remaining -= self.turn.points
        self.player.history.append(self.turn)
        self.note(f"{self.player.name} scored {self.turn.points}")
        self.current = (self.current + 1) % len(self.players)
        self.turn = Turn(player=self.current)

    # ------------------------------------------------------------------ #
    def scoreboard(self) -> list[str]:
        lines = []
        if self.start_score:
            for i, p in enumerate(self.players):
                mark = ">" if i == self.current else " "
                lines.append(f"{mark} {p.name}: {p.remaining}")
        thrown = " ".join(f"{d.label}" for d in self.turn.darts) or "-"
        lines.append(f"Visit: {thrown}")
        lines.append(f"Turn total: {self.turn.points}"
                     + ("  BUST" if self.turn.busted else ""))
        return lines
