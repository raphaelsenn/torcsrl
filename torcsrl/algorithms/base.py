from abc import ABC, abstractmethod
from typing import Any, Callable, Dict

import os
import time
import random
import torch
import numpy as np
import gymnasium as gym

from torcsrl.models.base import Actor, Critic
from torcsrl.evaluation.evaluation_stats import EpisodeStats, EvaluationStats
from torcsrl.buffers.replay_buffer import ReplayBuffer
from torcsrl.buffers.rollout_buffer import RolloutBuffer


class RLAlgorithm(ABC):
    """Base actor-critic algorithm class designed for deep reinforcement learning in TORCS.""" 
    def __init__(
        self,
        train_env: gym.Env,
        val_env: gym.Env,
        *,
        ac_kwargs: Dict[str, Any],
        gamma: float,
        tau_polyak: float,
        save_every: int, 
        eval_every: int, 
        verbose: bool = True,
        seed: int = 0,
        device: str = "cpu",
    ) -> None:
        ...
        super().__init__()

        self.env = train_env
        self.val_env = val_env

        # Computing device
        assert device in {"cpu", "cuda", "mps"}, (
            f"Unkdown device, expected device in [`cpu`, `cuda`, `mps`], got: {device}."
        )
        self.device = torch.device(device)

        # Actor/critic models
        self.ac_kwargs = ac_kwargs
        self.actor: Actor = ...
        self.critic: Critic = ...

        # Environment
        self.obs_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.obs_shape = self.env.observation_space.shape
        self.action_shape = self.env.action_space.shape
        self.obs_dim = int(np.prod(self.obs_shape))
        self.action_dim = int(np.prod(self.action_shape))

        self.action_low_np = np.asarray(self.env.action_space.low, dtype=np.float32)
        self.action_high_np = np.asanyarray(self.env.action_space.high, dtype=np.float32)
        self.action_scale_np = (self.action_high_np - self.action_low_np) / 2.0
        self.action_bias_np = (self.action_high_np + self.action_low_np) / 2.0
        
        self.action_low_torch = torch.as_tensor(self.action_low_np, dtype=torch.float32, device=self.device)
        self.action_high_torch = torch.as_tensor(self.action_high_np, dtype=torch.float32, device=self.device)
        self.action_scale_torch = torch.as_tensor(self.action_scale_np, dtype=torch.float32, device=self.device)
        self.action_bias_torch = torch.as_tensor(self.action_bias_np, dtype=torch.float32, device=self.device)

        # Hyperparameters and save/eval settings
        self.gamma = gamma
        self.tau_polyak = tau_polyak
        self.save_every = save_every
        self.eval_every = eval_every 
        self.verbose = verbose
        
        # Counters
        self.episode_counter = 0
        self.grad_step_counter = 0
        self.global_step_counter = 0

        # Evaluation stats
        # One entry per validation track ("category/name")
        self.eval_stats: dict[str, EvaluationStats] = {}

        # Evaluating/saving is due once these steps are reached (see in function on_episode_end)
        # This is because TORCS env can't stop while training... :/
        self.next_eval_step = eval_every
        self.next_save_step = save_every
        self.last_eval_time = time.monotonic()

        # Best evaluation so far (mean return over validation tracks), see save_best()
        self.best_eval_return = -float("inf")
        self.best_eval_step = -1

        # Reproducability
        self.seed = seed 
        self.torch_rng = torch.Generator(device=self.device)
        self.set_seeds()

    @abstractmethod
    def act(self, obs: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def train(self) -> None:
        raise NotImplementedError
    
    @abstractmethod
    def collect_rollouts(self) -> tuple[np.ndarray, Dict[str, Any]]:
        raise NotImplementedError

    @torch.no_grad()
    def act_numpy(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)   # [obs_dim]
        obs_t = obs_t.unsqueeze(0)                                              # [1, obs_dim]
        action = self.act(obs_t, deterministic)                                 # [1, action_dim]
        action = action.flatten().cpu().numpy()                                 # [action_dim]
        return action

    @torch.no_grad()
    def evaluate(self) -> None:
        """Drive one deterministic episode on every validation track and record the stats."""
        n_tracks = len(getattr(self.val_env, "tracks", [None]))

        for episode in range(n_tracks):
            obs, reset_info = self.val_env.reset(seed=self.seed + episode + 1)

            track = reset_info.get("track")
            track_key = f"{track['category']}/{track['name']}" if track else "eval"

            done = False
            n_steps = 0
            speed_sum = 0.0
            episode_return = 0.0

            while not done:
                action = self.act_numpy(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.val_env.step(action)
                done = terminated or truncated

                n_steps += 1
                episode_return += reward
                speed_sum += float(info.get("speed", 0.0))

            episode = EpisodeStats.from_episode(self.global_step_counter, episode_return, speed_sum / max(n_steps, 1), info)
            self.eval_stats.setdefault(track_key, EvaluationStats()).add(episode)

    def set_seeds(self) -> None:
        random.seed(self.seed)
        np.random.seed(self.seed)

        self.torch_rng.manual_seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def on_episode_end(self) -> None:
        """
        Call this when a training episode ends: evaluates and/or saves if due.

        Evaluating/saving only happens between episodes, so a running episode
        on the TORCS server is never interrupted...
        """
        step = self.global_step_counter

        if step >= self.next_eval_step:
            self.evaluate()

            if self.verbose:
                self.print_stats(time.monotonic() - self.last_eval_time)
            self.last_eval_time = time.monotonic()

            # Save the exact model that produced the printed eval stats
            self.save()

            # Additionally keep the best model of the entire run
            eval_return = self.latest_eval_return()
            if eval_return > self.best_eval_return:
                self.best_eval_return = eval_return
                self.best_eval_step = step
                self.save_best()

            self.next_eval_step = (step // self.eval_every + 1) * self.eval_every
            self.next_save_step = (step // self.save_every + 1) * self.save_every

        elif step >= self.next_save_step:
            self.save()
            self.next_save_step = (step // self.save_every + 1) * self.save_every

    def print_stats(self, time: float) -> None:
        header = (
            f"Timestep: {self.global_step_counter:>9d}  "
            f"Episode: {self.episode_counter:>6d}  "
            f"Time: {time:>12.2f}s"
        )
        print(header)

        for track_key, stats in self.eval_stats.items():
            e = stats.latest
            if e is None:
                continue

            report = (
                f"  Track: {track_key:<20s}  "
                f"Return: {e.episode_return:>10.2f}  "
                f"Distance: {e.distance_raced:>8.1f} m  "
                f"Speed: {e.average_speed:>6.1f} km/h  "
                f"Damage: {e.damage:>7.1f}  "
                f"Lap time: {e.lap_time:>7.2f} s  "
                f"End: {e.end_reason}"
            )
            print(report)
    
    def checkpoint_networks(self) -> dict[str, torch.nn.Module]:
        """Networks written to disk by save(), saved as <name>-<algorithm>-<step>.pt."""
        return {"actor": self.actor, "critic": self.critic}

    def latest_eval_return(self) -> float:
        """Mean return of the latest evaluation over all validation tracks."""
        returns = [s.latest.episode_return for s in self.eval_stats.values() if s.latest is not None]
        return float(np.mean(returns)) if returns else -float("inf")

    def checkpoint_dir(self) -> str:
        algo_name = self.__class__.__name__.lower()
        run_name = f"{algo_name}-lr_pi{self.lr_actor}-lr_q{self.lr_critic}-seed{self.seed}"
        save_dir = os.path.join("checkpoints", algo_name, run_name)
        os.makedirs(save_dir, exist_ok=True)
        return save_dir

    def save_best(self) -> None:
        """Overwrite <name>-<algorithm>-best.pt with the current networks."""
        algo_name = self.__class__.__name__.lower()
        save_dir = self.checkpoint_dir()

        for name, net in self.checkpoint_networks().items():
            torch.save(net.state_dict(), os.path.join(save_dir, f"{name}-{algo_name}-best.pt"))

        with open(os.path.join(save_dir, "best.txt"), "w") as f:
            f.write(f"step {self.best_eval_step}\nmean_eval_return {self.best_eval_return}\n")

        if self.verbose:
            print(f"  New best model (mean eval return {self.best_eval_return:.2f}) saved.")

    def save(self) -> None:
        algo_name = self.__class__.__name__.lower()
        save_dir = self.checkpoint_dir()
        run_name = os.path.basename(save_dir)

        for name, net in self.checkpoint_networks().items():
            torch.save(net.state_dict(), os.path.join(save_dir, f"{name}-{algo_name}-{self.global_step_counter}.pt"))
        
        for track_key, stats in self.eval_stats.items():
            safe_track_key = track_key.replace("/", "_")
            save_dir = os.path.join("checkpoints", algo_name, run_name, f"eval_stats_{safe_track_key}.csv")
            stats.to_csv(save_dir)


class OffPolicyAlgorithm(RLAlgorithm, ABC):
    """Base actor-critic off-policy algorithm class designed for deep reinforcement learning in TORCS.""" 

    # Std of the Gaussian noise on tita actions (warmup="expert_noisy")
    EXPERT_NOISE = 0.2

    def __init__(
        self,
        train_env: gym.Env,
        val_env: gym.Env, 
        *,
        ac_kwargs: Dict[str, Any],
        gamma: float,
        tau_polyak: float,
        buffer_size: int,
        buffer_start_size: int,
        batch_size: int,
        gradient_steps: int,
        warmup: str = "random",
        save_every: int,
        eval_every: int,
        verbose: bool = True,
        seed: int = 0,
        device: str = "cpu",
    ) -> None:
        super().__init__(
            train_env=train_env,
            val_env=val_env, 
            ac_kwargs=ac_kwargs,
            gamma=gamma,
            tau_polyak=tau_polyak,
            save_every=save_every,
            eval_every=eval_every,
            verbose=verbose,
            seed=seed,
            device=device,
        )

        self.buffer_size = buffer_size
        self.buffer_start_size = buffer_start_size
        if warmup not in ("random", "expert", "expert_noisy"):
            raise ValueError(f"warmup must be 'random', 'expert' or 'expert_noisy', got {warmup!r}.")
        self.warmup = warmup
        self.batch_size = batch_size
        self.gradient_steps = gradient_steps

        self.replay_buffer = ReplayBuffer(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            capacity=buffer_size,
            batch_size=batch_size,
            device=device,
        )

    def collect_rollouts(self, on_transition: Callable[..., None] | None = None) -> tuple[np.ndarray, Dict[str, Any]]:
        """
        Fill the replay buffer with buffer_start_size steps before training.

        self.warmup picks the actions: "random", "expert" (tita) or "expert_noisy"
        (tita + Gaussian noise with std EXPERT_NOISE, clipped to the action space).
        The steps are split evenly over the training tracks.
        """
        if on_transition is None:
            def on_transition(obs, action, reward, obs_next, terminated, truncated):
                self.replay_buffer.push(obs, action, reward, obs_next, terminated)

        env = self.env  # NOTE: Training environment
        env.action_space.seed(self.seed)
        rng = np.random.default_rng(self.seed)

        def warmup_action(info: Dict[str, Any]) -> np.ndarray:
            if self.warmup == "random":
                return env.action_space.sample()
            action = np.asarray(info["expert_action"], dtype=np.float32)
            if self.warmup == "expert_noisy":
                action = action + rng.normal(0.0, self.EXPERT_NOISE, size=action.shape).astype(np.float32)
                action = np.clip(action, env.action_space.low, env.action_space.high)
            return action

        # Use the inner env, so the track wrapper does not change tracks or count these steps.
        tracks = list(getattr(env, "tracks", [None]))
        inner = env.env if hasattr(env, "tracks") else env
        steps_per_track = -(-self.buffer_start_size // len(tracks))     # ceil

        for i, track in enumerate(tracks):
            options = {"track": track} if track is not None else None
            obs, info = inner.reset(seed=self.seed + i, options=options)
            for step in range(steps_per_track):
                action = warmup_action(info)
                obs_next, reward, terminated, truncated, info = inner.step(action)

                last = step == steps_per_track - 1
                on_transition(obs, action, reward, obs_next, terminated, truncated or last)

                obs = obs_next
                if (terminated or truncated) and not last:
                    obs, info = inner.reset(options=options)

            if self.verbose:
                name = getattr(track, "name", "env")
                print(f"{self.warmup.capitalize()} warm-up on {name}: |B| = {len(self.replay_buffer)}")

        return env.reset(seed=self.seed)


class OnPolicyAlgorithm(RLAlgorithm, ABC):
    """Base actor-critic on-policy algorithm class designed for deep reinforcement learning in TORCS.""" 
    def __init__(
        self,
        train_env: gym.Env,
        val_env: gym.Env, 
        *,
        ac_kwargs: Dict[str, Any],
        gamma: float,
        tau_polyak: float,
        horizon: int,
        batch_size: int,
        epochs: int,
        save_every: int,
        eval_every: int,
        verbose: bool = True,
        seed: int = 0,
        device: str = "cpu",
    ) -> None:
        super().__init__(
            train_env=train_env,
            val_env=val_env,
            ac_kwargs=ac_kwargs,
            gamma=gamma,
            tau_polyak=tau_polyak,
            save_every=save_every,
            eval_every=eval_every,
            verbose=verbose,
            seed=seed,
            device=device,
        )

        self.horizon = horizon
        self.batch_size = batch_size
        self.epochs = epochs

        self.rollout_buffer = RolloutBuffer(
            obs_shape=self.obs_shape,
            action_shape=self.action_shape,
            horizon=horizon,
            batch_size=batch_size,
        )
