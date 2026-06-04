# type: ignore
"""simple logger for displaying key-value pairs and live scrolling graphs
in small windows during simulation or animation"""

import math
import time
from collections import deque

import matplotlib.pyplot as plt

MAX_SINGLE_COL = 20

# shared dark theme
_BG = "#1a1a2e"
_GRID = "#2a2a4a"
_FG = "#8888aa"
_COLORS = ["#56d4f5", "#f5a623", "#a78bfa", "#7defa5", "#f87171", "#fb7185"]


class Logger:
    # ---- text dashboard state ----
    _data = {}
    _text_objects = {}
    _fig = None
    _ax = None
    _last_draw_time = 0.0
    _DRAW_INTERVAL = 0.1  # redraw at most 5fps

    # ---- graph state ----
    # series are identified internally by a (group, name) tuple, so the same
    # display name (e.g. "actual") can be reused across different groups.
    # _graph_groups: group name -> list of series names (preserves order)
    # _graph_hist:   (group, name) -> deque of recent values
    # _graph_count:  (group, name) -> total values ever appended (scrolling x)
    _graph_groups = {}
    _graph_hist = {}
    _graph_count = {}
    _graph_lines = {}  # (group, name) -> Line2D
    _graph_axes = {}  # group name -> Axes
    _graph_fig = None
    _graph_bg = {}  # group name -> cached canvas background (for blitting)
    _graph_ylim = {}  # group name -> current (lo, hi) y-limits
    _last_graph_draw_time = 0.0
    _GRAPH_DRAW_INTERVAL = 0.1  # graph redraws at most 5fps
    _GRAPH_WINDOW = 200  # rolling window length (samples)

    # ------------------------------------------------------------------
    # text dashboard
    # ------------------------------------------------------------------
    @classmethod
    def _init(cls):
        if cls._fig is not None:
            return
        cls._fig, cls._ax = plt.subplots(figsize=(2, 5))
        cls._fig.patch.set_facecolor(_BG)
        cls._fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        cls._ax.set_facecolor(_BG)
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

    # ------------------------------------------------------------------
    # graphing
    # ------------------------------------------------------------------
    @classmethod
    def graphData(cls, key, value, group=None):
        """Record a numeric value to be drawn as a live scrolling line graph.

        Parameters
        ----
        key: str
            Name of this series (shown in the legend).
        value: float
            The sample to append. Non-finite/None values are skipped.
        group: str or None
            Series sharing a group are plotted on the same axes (e.g. an
            "actual" and a "setpoint" line). Defaults to ``key`` so each
            series gets its own subplot.
        """
        if value is None:
            return
        try:
            value = float(value)
        except (TypeError, ValueError):
            return
        if not math.isfinite(value):  # skip NaN / inf so axes stay scalable
            return

        cls._init_graph()
        group = group if group is not None else key
        sid = (group, key)

        if sid not in cls._graph_hist:
            cls._graph_hist[sid] = deque(maxlen=cls._GRAPH_WINDOW)
            cls._graph_count[sid] = 0
            cls._graph_groups.setdefault(group, [])
            if key not in cls._graph_groups[group]:
                cls._graph_groups[group].append(key)

        cls._graph_hist[sid].append(value)
        cls._graph_count[sid] += 1

    @classmethod
    def _init_graph(cls):
        if cls._graph_fig is not None:
            return
        cls._graph_fig = plt.figure(figsize=(3, 5))
        cls._graph_fig.patch.set_facecolor(_BG)
        cls._graph_fig.canvas.manager.set_window_title("Logger — Graphs")
        cls._graph_fig.canvas.mpl_connect("draw_event", cls._on_graph_draw)
        plt.show(block=False)
        try:
            cls._graph_fig.canvas.manager.window.move(200, 50)
        except Exception:
            pass

    @classmethod
    def _on_graph_draw(cls, _event):
        """Cache each subplot's background (without the animated lines) so that
        refreshes can blit just the lines instead of redrawing the whole figure."""
        canvas = cls._graph_fig.canvas
        cls._graph_bg = {
            group: canvas.copy_from_bbox(ax.bbox)
            for group, ax in cls._graph_axes.items()
        }

    @classmethod
    def _rebuild_graph_axes(cls):
        """(Re)create one subplot per group and a line per series."""
        cls._graph_fig.clf()
        cls._graph_axes.clear()
        cls._graph_lines.clear()

        groups = list(cls._graph_groups.keys())
        n = len(groups)
        if n == 0:
            return

        axes = cls._graph_fig.subplots(n, 1, squeeze=False)[:, 0]
        for ax, group in zip(axes, groups):
            ax.set_facecolor(_BG)
            ax.tick_params(colors=_FG, labelsize=6)
            for spine in ax.spines.values():
                spine.set_color(_GRID)
            ax.grid(True, color=_GRID, lw=0.4)
            ax.set_title(group, color=_FG, fontsize=8, fontfamily="monospace")
            ax.format_coord = lambda *_: ""

            color_base = groups.index(group)
            for j, name in enumerate(cls._graph_groups[group]):
                color = _COLORS[(color_base + j) % len(_COLORS)]
                (line,) = ax.plot(
                    [], [], color=color, lw=1.2, label=name, animated=True
                )
                cls._graph_lines[(group, name)] = line

            # fixed rolling-window x-axis so the limits never change per-frame
            ax.set_xlim(0, max(1, cls._GRAPH_WINDOW - 1))

            if len(cls._graph_groups[group]) > 1:
                ax.legend(
                    loc="upper left",
                    fontsize=6,
                    facecolor=_BG,
                    edgecolor=_GRID,
                    labelcolor=_FG,
                )
            cls._graph_axes[group] = ax

        cls._graph_fig.tight_layout(pad=0.8)
        cls._graph_bg = {}  # invalidate cached backgrounds after a rebuild
        cls._graph_ylim = {}

    @classmethod
    def _update_graphs(cls):
        if cls._graph_fig is None or not cls._graph_hist:
            return

        # rebuild lines/axes only when the set of series changes
        rebuilt = False
        if set(cls._graph_lines.keys()) != set(cls._graph_hist.keys()):
            cls._rebuild_graph_axes()
            rebuilt = True

        # throttle: skip entirely until the redraw window has elapsed
        now = time.perf_counter()
        if not rebuilt and now - cls._last_graph_draw_time < cls._GRAPH_DRAW_INTERVAL:
            return
        cls._last_graph_draw_time = now

        # refresh line data; x is the index within the rolling window, so the
        # x-limits stay fixed (a moving x-axis would force a full redraw)
        need_full = rebuilt
        for group, ax in cls._graph_axes.items():
            dmin = dmax = None
            for name in cls._graph_groups[group]:
                hist = cls._graph_hist[(group, name)]
                if not hist:
                    continue
                ys = list(hist)
                cls._graph_lines[(group, name)].set_data(range(len(ys)), ys)
                lo, hi = min(ys), max(ys)
                dmin = lo if dmin is None else min(dmin, lo)
                dmax = hi if dmax is None else max(dmax, hi)

            if dmin is None:
                continue
            # only rescale y (and pay for a full redraw) when data exceeds the
            # current view; expand with headroom so this stays infrequent
            cur = cls._graph_ylim.get(group)
            if cur is None or dmin < cur[0] or dmax > cur[1]:
                if dmin == dmax:
                    pad = 1.0 if dmin == 0 else abs(dmin) * 0.2
                else:
                    pad = (dmax - dmin) * 0.2
                cls._graph_ylim[group] = (dmin - pad, dmax + pad)
                ax.set_ylim(*cls._graph_ylim[group])
                need_full = True

        canvas = cls._graph_fig.canvas
        # When layout or y-limits changed, schedule a non-blocking redraw and
        # skip blitting this frame. draw_event fires after the idle repaint and
        # recaptures the background so the next frame can blit cleanly.
        # Using draw_idle() instead of draw() avoids blocking the main
        # animation loop, which was causing lurching on first run.
        if need_full or not cls._graph_bg:
            cls._graph_bg = {}
            canvas.draw_idle()
            return

        # fast path: restore the cached background and blit only the lines
        for group, ax in cls._graph_axes.items():
            bg = cls._graph_bg.get(group)
            if bg is None:
                continue
            canvas.restore_region(bg)
            for name in cls._graph_groups[group]:
                ax.draw_artist(cls._graph_lines[(group, name)])
            canvas.blit(ax.bbox)

    # ------------------------------------------------------------------
    # combined refresh
    # ------------------------------------------------------------------
    @classmethod
    def update(cls):
        cls._update_text()
        cls._update_graphs()

    @classmethod
    def _update_text(cls):
        if not cls._data:
            return
        cls._init()

        if set(cls._data.keys()) != set(cls._text_objects.keys()):
            cls._ax.cla()
            cls._ax.axis("off")
            cls._ax.set_facecolor(_BG)
            cls._ax.set_xlim(0, 1)
            cls._ax.set_ylim(0, 1)
            cls._text_objects.clear()

            colors = ["#56d4f5", "#f5a623", "#a78bfa", "#7defa5", "#f87171"]
            n = len(cls._data)
            two_col = n > MAX_SINGLE_COL
            keys = list(cls._data.keys())

            if two_col:
                rows_per_col = (n + 1) // 2
                col_configs = [
                    (0.03, 0.47, 0.01, 0.49),
                    (0.53, 0.97, 0.51, 0.99),
                ]
            else:
                rows_per_col = n
                col_configs = [(0.04, 0.96, 0.02, 0.98)]

            fig_w = 4 if two_col else 2
            fig_h = max(5, min(16, rows_per_col * 0.5))
            cls._fig.set_size_inches(fig_w, fig_h, forward=True)

            start_y = 0.97
            step = 0.94 / max(rows_per_col, 1)

            for i, key in enumerate(keys):
                col = i // rows_per_col if two_col else 0
                row = i % rows_per_col
                color = colors[i % len(colors)]
                x_key, x_val, xmin, xmax = col_configs[col]
                y = start_y - row * step

                cls._ax.axhline(
                    y - step * 0.55, color="#2a2a4a", lw=0.5, xmin=xmin, xmax=xmax
                )
                cls._ax.text(
                    x_key,
                    y,
                    key,
                    color="#8888aa",
                    fontsize=6.5,
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
                    fontsize=7,
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
