"""Launching one TORCS process for one SCR driver slot.

Each server gets its own temporary HOME with a copy of ``~/.torcs``, so the
race file and screen settings written here never touch the user's config.
Only graph.xml, where TORCS saves the camera view, is copied back on close
in GUI mode.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import signal
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from gym_torcs.constants import (
    CLIENT_TIMEOUT_US,
    DEFAULT_TORCS_EXECUTABLE,
    SCR_FIRST_PORT,
    SCR_NUM_SLOTS,
    TRACK_CATEGORIES,
)

logger = logging.getLogger(__name__)

# Menu path from the TORCS start screen to "New Race" of the practice race.
GUI_START_KEYS = ("Return", "Return", "Up", "Up", "Return", "Return")
GUI_MENU_DELAY = 2.0
GUI_KEY_DELAY = 0.1

RACE_TEMPLATE = """\
<?xml version="1.0" encoding="UTF-8"?>
<params name="Practice">
  <section name="Header">
    <attstr name="name" val="Practice"/>
    <attstr name="description" val="Practice"/>
    <attnum name="priority" val="100"/>
    <attstr name="menu image" val="data/img/splash-practice.png"/>
    <attstr name="run image" val="data/img/splash-run-practice.png"/>
  </section>
  <section name="Tracks">
    <attnum name="maximum number" val="1"/>
    <section name="1">
      <attstr name="name" val="{track_name}"/>
      <attstr name="category" val="{track_category}"/>
    </section>
  </section>
  <section name="Races">
    <section name="1">
      <attstr name="name" val="Practice"/>
    </section>
  </section>
  <section name="Practice">
    <attnum name="laps" val="{laps}"/>
    <attstr name="type" val="practice"/>
    <attstr name="starting order" val="drivers list"/>
    <attstr name="restart" val="yes"/>
    <attstr name="display mode" val="normal"/>
    <attstr name="display results" val="no"/>
    <attnum name="distance" val="0" unit="km"/>
    <section name="Starting Grid">
      <attnum name="rows" val="1"/>
      <attnum name="distance to start" val="100"/>
      <attnum name="distance between columns" val="20"/>
      <attnum name="offset within a column" val="10"/>
      <attnum name="initial speed" val="0" unit="km/h"/>
      <attnum name="initial height" val="0.2" unit="m"/>
    </section>
  </section>
  <section name="Drivers">
    <attnum name="maximum number" val="1"/>
    <attstr name="focused module" val="scr_server"/>
    <attnum name="focused idx" val="{driver_idx}"/>
    <section name="1">
      <attstr name="module" val="scr_server"/>
      <attnum name="idx" val="{driver_idx}"/>
    </section>
  </section>
  <section name="Drivers Start List">
    <section name="1">
      <attstr name="module" val="scr_server"/>
      <attnum name="idx" val="{driver_idx}"/>
    </section>
  </section>
  <section name="Configuration">
    <attnum name="current configuration" val="4"/>
    <section name="1">
      <attstr name="type" val="track select"/>
    </section>
    <section name="2">
      <attstr name="type" val="drivers select"/>
    </section>
    <section name="3">
      <attstr name="type" val="race config"/>
      <attstr name="race" val="Practice"/>
      <section name="Options">
        <section name="1">
          <attstr name="type" val="race length"/>
        </section>
        <section name="2">
          <attstr name="type" val="display mode"/>
        </section>
      </section>
    </section>
  </section>
</params>
"""


@dataclass(frozen=True, slots=True)
class TrackSpec:
    """A TORCS track. Without a category it is looked up in TRACK_CATEGORIES."""
    name: str
    category: str | None = None

    def __post_init__(self) -> None:
        if self.category is None:
            object.__setattr__(self, "category", TRACK_CATEGORIES.get(self.name, "road"))


class TorcsServer:
    """Owns one TORCS process driving the SCR robot that listens on ``port``.

    Without ``gui`` TORCS runs headless from the command line. With ``gui`` the
    window opens and the race is started by sending key presses with ``xte``.
    ``window_size`` fixes the window resolution, which vision depends on.
    """

    def __init__(
        self,
        port: int,
        *,
        executable: str = DEFAULT_TORCS_EXECUTABLE,
        gui: bool = False,
        window_size: tuple[int, int] | None = None,
        startup_sleep: float = 1.0,
        client_timeout_us: int = CLIENT_TIMEOUT_US,
        verbose: bool = False,
    ) -> None:
        driver_idx = port - SCR_FIRST_PORT
        if not 0 <= driver_idx < SCR_NUM_SLOTS:
            last_port = SCR_FIRST_PORT + SCR_NUM_SLOTS - 1
            raise ValueError(f"port must be in {SCR_FIRST_PORT}..{last_port}, got {port}.")
        if gui and shutil.which("xte") is None:
            raise RuntimeError("Starting a race in the TORCS window needs xte (package xautomation).")

        self.port = port
        self.driver_idx = driver_idx
        self.executable = executable
        self.gui = gui
        self.startup_sleep = startup_sleep
        self.client_timeout_us = client_timeout_us
        self.verbose = verbose
        self.process: subprocess.Popen[bytes] | None = None

        self.home = Path(tempfile.mkdtemp(prefix=f"torcs-{port}-"))
        self.config_dir = self.home / ".torcs" / "config"
        user_config = Path.home() / ".torcs"
        if user_config.exists():
            shutil.copytree(user_config, self.home / ".torcs")
        if window_size is not None:
            self._set_window_size(*window_size)

    def start(self, track: TrackSpec, laps: int) -> None:
        self.stop()

        race_file = self.config_dir / "raceman" / "practice.xml"
        race_file.parent.mkdir(parents=True, exist_ok=True)
        race_file.write_text(
            RACE_TEMPLATE.format(
                track_name=track.name,
                track_category=track.category,
                laps=laps,
                driver_idx=self.driver_idx,
            )
        )

        command = [self.executable, "-nofuel", "-nodamage", "-nolaptime", "-t", str(self.client_timeout_us)]
        if not self.gui:
            command += ["-r", str(race_file)]

        env = os.environ | {"HOME": str(self.home)}
        env.setdefault("XAUTHORITY", str(Path.home() / ".Xauthority"))

        logger.info("Starting %s on port %d (%s/%s)", " ".join(command), self.port, track.category, track.name)
        output = None if self.verbose else subprocess.DEVNULL
        self.process = subprocess.Popen(
            command, env=env, stdout=output, stderr=output, start_new_session=True
        )
        time.sleep(self.startup_sleep)

        if self.gui:
            time.sleep(GUI_MENU_DELAY)
            for key in GUI_START_KEYS:
                subprocess.run(["xte", f"key {key}"], check=True)
                time.sleep(GUI_KEY_DELAY)

    def stop(self) -> None:
        if self.process is None:
            return
        if self.process.poll() is None:
            # TORCS runs through a launcher script, so signal the whole group.
            os.killpg(self.process.pid, signal.SIGTERM)
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, signal.SIGKILL)
                self.process.wait()
        self.process = None

    def close(self) -> None:
        self.stop()
        if self.gui:
            self._save_graphics_config()
        shutil.rmtree(self.home, ignore_errors=True)

    def _save_graphics_config(self) -> None:
        """Copy graph.xml back to ~/.torcs, so the camera picked with F2..F11 is kept."""
        graph_file = self.config_dir / "graph.xml"
        user_config = Path.home() / ".torcs" / "config"
        if graph_file.exists() and user_config.is_dir():
            shutil.copy2(graph_file, user_config / "graph.xml")

    def _set_window_size(self, width: int, height: int) -> None:
        screen_file = self.config_dir / "screen.xml"
        if not screen_file.exists():
            raise FileNotFoundError(f"{screen_file} not found. Start TORCS once to create ~/.torcs.")

        values = {
            "x": width,
            "y": height,
            "window width": width,
            "window height": height,
        }
        xml = screen_file.read_text()
        for name, value in values.items():
            xml = re.sub(rf'(<attnum name="{name}" val=")[^"]*', rf"\g<1>{value}", xml)
        xml = re.sub(r'(<attstr name="fullscreen"[^>]*val=")[^"]*', r"\g<1>no", xml)
        screen_file.write_text(xml)
