"""Camera frames from the TORCS window, read through System V shared memory.

The patched TORCS (``src/linux/main.cpp``) creates a segment with key 1234:

    struct shared_use_st {
        int written;
        uint8_t data[640 * 480 * 3];
        int pause;
        int zmq_flag;
        int save_flag;
    };

While ``pause`` is 1, the race engine copies the lower left 640x480 pixels of
the OpenGL window into ``data`` about ten times per simulated second and sets
``written`` to 1. Frames therefore only exist in GUI mode, and since the key is
fixed, only one TORCS instance per machine can deliver them.
"""

from __future__ import annotations

import ctypes

import numpy as np

SHM_KEY = 1234
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
IPC_CREAT = 0o1000


class _SharedFrame(ctypes.Structure):
    _fields_ = [
        ("written", ctypes.c_int),
        ("data", ctypes.c_uint8 * (FRAME_WIDTH * FRAME_HEIGHT * 3)),
        ("pause", ctypes.c_int),
        ("zmq_flag", ctypes.c_int),
        ("save_flag", ctypes.c_int),
    ]


_libc = ctypes.CDLL(None, use_errno=True)
_libc.shmget.argtypes = (ctypes.c_int, ctypes.c_size_t, ctypes.c_int)
_libc.shmat.argtypes = (ctypes.c_int, ctypes.c_void_p, ctypes.c_int)
_libc.shmat.restype = ctypes.c_void_p
_libc.shmdt.argtypes = (ctypes.c_void_p,)


class FrameBuffer:
    def __init__(self) -> None:
        shm_id = _libc.shmget(SHM_KEY, ctypes.sizeof(_SharedFrame), IPC_CREAT | 0o666)
        if shm_id == -1:
            raise OSError(ctypes.get_errno(), "shmget failed for the TORCS frame buffer")

        address = _libc.shmat(shm_id, None, 0)
        if address == ctypes.c_void_p(-1).value:
            raise OSError(ctypes.get_errno(), "shmat failed for the TORCS frame buffer")

        self._address = address
        self._shared = _SharedFrame.from_address(address)
        self._pixels = np.ctypeslib.as_array(self._shared.data).reshape(FRAME_HEIGHT, FRAME_WIDTH, 3)

    def start_capture(self) -> None:
        """Ask TORCS to write frames. TORCS clears the flag on every launch."""
        self._shared.written = 0
        self._shared.pause = 1

    @property
    def has_frame(self) -> bool:
        return self._shared.written == 1

    def read(self) -> np.ndarray:
        """Latest frame as an (480, 640, 3) uint8 RGB array, top row first."""
        return self._pixels[::-1].copy()

    def close(self) -> None:
        if self._address is None:
            return
        self._shared.pause = 0
        self._pixels = None
        self._shared = None
        _libc.shmdt(self._address)
        self._address = None


def resize(image: np.ndarray, height: int, width: int) -> np.ndarray:
    """Downscale an image by averaging over the pixel block behind each output pixel."""
    rows = np.linspace(0, image.shape[0], height + 1).astype(int)
    cols = np.linspace(0, image.shape[1], width + 1).astype(int)

    sums = np.add.reduceat(image.astype(np.float32), rows[:-1], axis=0)
    sums = np.add.reduceat(sums, cols[:-1], axis=1)
    areas = np.outer(np.diff(rows), np.diff(cols))[..., None]
    return np.round(sums / areas).astype(np.uint8)
