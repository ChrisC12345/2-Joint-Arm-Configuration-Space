# type: ignore
"""simple logger for displaying key-value pairs in a small window during simulation or animation"""

import time
import matplotlib.pyplot as plt

MAX_SINGLE_COL = 12


class Logger:
    _data = {}
    _text_objects = {}
    _fig = None
    _ax = None
    _last_draw_time = 0.0
    _DRAW_INTERVAL = 0.1  # redraw at most 10fps

    @classmethod
    def _init(cls):
        if cls._fig is not None:
            return
        cls._fig, cls._ax = plt.subplots(figsize=(3, 5))
        cls._fig.patch.set_facecolor("#1a1a2e")
        cls._fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        cls._ax.set_facecolor("#1a1a2e")
        cls._ax.axis("off")
        cls._fig.canvas.manager.set_window_title("Logger")
        cls._ax.format_coord = lambda *_: ""
        plt.show(block=False)
        try:
            cls._fig.canvas.manager.window.move(0, 50)
        except Exception:
            pass

    @classmethod
    def recordData(cls, key, value):
        cls._init()
        cls._data[key] = value

    @classmethod
    def update(cls):
        cls._init()

        if not cls._data:
            return

        if set(cls._data.keys()) != set(cls._text_objects.keys()):
            cls._ax.cla()
            cls._ax.axis("off")
            cls._ax.set_facecolor("#1a1a2e")
            cls._ax.set_xlim(0, 1)
            cls._ax.set_ylim(0, 1)
            cls._text_objects.clear()

            colors = ["#56d4f5", "#f5a623", "#a78bfa", "#7defa5", "#f87171"]
            n = len(cls._data)
            two_col = n > MAX_SINGLE_COL
            keys = list(cls._data.keys())

            if two_col:
                cls._fig.set_size_inches(6, 5, forward=True)
                rows_per_col = (n + 1) // 2
                row_height = 0.045
                start_y = 0.97
                col_configs = [
                    (0.03, 0.47, 0.01, 0.49),
                    (0.53, 0.97, 0.51, 0.99),
                ]
            else:
                cls._fig.set_size_inches(3, 5, forward=True)
                rows_per_col = n
                row_height = 0.045
                start_y = 0.97
                col_configs = [(0.04, 0.96, 0.02, 0.98)]

            for i, key in enumerate(keys):
                col = i // rows_per_col if two_col else 0
                row = i % rows_per_col
                color = colors[i % len(colors)]
                x_key, x_val, xmin, xmax = col_configs[col]
                y = start_y - row * row_height * 2.2

                cls._ax.axhline(
                    y + row_height * 0.6, color="#2a2a4a", lw=0.5, xmin=xmin, xmax=xmax
                )
                cls._ax.text(
                    x_key,
                    y,
                    key,
                    color="#8888aa",
                    fontsize=7.5,
                    ha="left",
                    va="top",
                    transform=cls._ax.transAxes,
                    fontfamily="monospace",
                )
                txt = cls._ax.text(
                    x_val,
                    y,
                    "—",
                    color=color,
                    fontsize=8,
                    ha="right",
                    va="top",
                    transform=cls._ax.transAxes,
                    fontfamily="monospace",
                    fontweight="bold",
                )
                cls._text_objects[key] = txt

        for key, txt in cls._text_objects.items():
            val = cls._data[key]
            txt.set_text(f"{val:.4f}" if isinstance(val, float) else str(val))

        now = time.perf_counter()
        if now - cls._last_draw_time >= cls._DRAW_INTERVAL:
            cls._fig.canvas.draw_idle()
            cls._last_draw_time = now
