"""
Local-only mouse mover for automation/testing.

Safety:
- Press Ctrl+C in terminal to stop.
- Move your cursor to the top-left corner to trigger pyautogui fail-safe.
"""

from __future__ import annotations

import random
import time

try:
    import pyautogui
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: pyautogui\n"
        "Install with: pip install pyautogui"
    ) from exc


def countdown(seconds: int = 3) -> None:
    print("Starting local mouse mover...")
    for remaining in range(seconds, 0, -1):
        print(f"Starting in {remaining}...")
        time.sleep(1)


def run_mouse_mover() -> None:
    pyautogui.FAILSAFE = True
    pyautogui.PAUSE = 0.05

    width, height = pyautogui.size()
    print(f"Screen size detected: {width}x{height}")
    print("Move cursor to top-left corner or press Ctrl+C to stop.\n")

    while True:
        x, y = pyautogui.position()

        # Tiny move away and back keeps net cursor position unchanged.
        dx = random.choice([-1, 1]) * random.randint(1, 2)
        dy = random.choice([-1, 1]) * random.randint(1, 2)

        temp_x = min(max(0, x + dx), width - 1)
        temp_y = min(max(0, y + dy), height - 1)

        pyautogui.moveTo(temp_x, temp_y, duration=0.05)
        pyautogui.moveTo(x, y, duration=0.05)
        print(f"Pulse movement: ({x},{y}) -> ({temp_x},{temp_y}) -> ({x},{y})")

        time.sleep(random.uniform(1.2, 2.4))


def main() -> None:
    countdown(3)
    try:
        run_mouse_mover()
    except KeyboardInterrupt:
        print("\nStopped by user.")
    except pyautogui.FailSafeException:
        print("\nFail-safe triggered (cursor moved to top-left). Stopped.")


if __name__ == "__main__":
    main()
