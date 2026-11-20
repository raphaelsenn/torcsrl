import math
from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class EpisodeStats:
    """Results of one evaluation episode."""

    timesteps: int
    episode_return: float
    episode_steps: int
    distance_raced: float       # m
    average_speed: float        # km/h
    damage: float
    laps_completed: int
    lap_time: float             # s, NaN if no lap was finished
    end_reason: str | None

    @classmethod
    def from_episode(
        cls,
        timesteps: int,
        episode_return: float,
        average_speed: float,
        info: dict[str, Any],
    ) -> "EpisodeStats":
        """Build from the episode return, the average speed and the last step's info."""
        episode_steps = int(info.get("episode_steps", 0))
        laps_completed = int(info.get("laps_completed", 0))
        return cls(
            timesteps=timesteps,
            episode_return=float(episode_return),
            episode_steps=episode_steps,
            distance_raced=float(info.get("distance_raced", 0.0)),
            average_speed=float(average_speed),
            damage=float(info.get("damage", 0.0)),
            laps_completed=laps_completed,
            lap_time=float(info["lap_time"]) if laps_completed > 0 else math.nan,
            end_reason=info.get("end_reason"),
        )


@dataclass
class EvaluationStats:
    """All evaluations on one track."""

    episodes: list[EpisodeStats] = field(default_factory=list)

    def add(self, episode: EpisodeStats) -> None:
        self.episodes.append(episode)

    @property
    def latest(self) -> EpisodeStats | None:
        return self.episodes[-1] if self.episodes else None

    def to_dataframe(self) -> pd.DataFrame:
        columns = list(EpisodeStats.__dataclass_fields__)
        return pd.DataFrame([asdict(e) for e in self.episodes], columns=columns)

    def to_csv(self, path: str) -> None:
        self.to_dataframe().to_csv(path, index=False)

    def __len__(self) -> int:
        return len(self.episodes)
