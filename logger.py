#type: ignore
"""simple logger for displaying key-value pairs in a small window during simulation or animation"""

import matplotlib.pyplot as plt

class Logger:
    _data = {}
    _text_objects = {}
    _fig = None
    _ax = None

    @classmethod
    def _init(cls):
        if cls._fig is not None:
            return
        cls._fig, cls._ax = plt.subplots(figsize=(3, 5))
        cls._fig.patch.set_facecolor('#1a1a2e')
        cls._fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        cls._ax.set_facecolor('#1a1a2e')
        cls._ax.axis('off')
        cls._fig.canvas.manager.set_window_title('Logger')
        plt.show(block=False)

    @classmethod
    def recordData(cls, key, value):
        cls._init()
        cls._data[key] = value

    @classmethod
    def update(cls):
        cls._init()

        n = len(cls._data)
        if n == 0:
            return

        if set(cls._data.keys()) != set(cls._text_objects.keys()):
            cls._ax.cla()
            cls._ax.axis('off')
            cls._ax.set_facecolor('#1a1a2e')
            cls._ax.set_xlim(0, 1)
            cls._ax.set_ylim(0, 1)
            cls._text_objects.clear()

            colors = ['#56d4f5', '#f5a623', '#a78bfa', '#7defa5', '#f87171']
            row_height = 0.045
            start_y = 0.97

            for i, key in enumerate(cls._data):
                color = colors[i % len(colors)]
                y = start_y - i * row_height * 2.2

                cls._ax.axhline(y + row_height * 0.6, color='#2a2a4a', lw=0.5, xmin=0.02, xmax=0.98)

                cls._ax.text(0.04, y, key, color='#8888aa', fontsize=7.5,
                            ha='left', va='top', transform=cls._ax.transAxes,
                            fontfamily='monospace')

                txt = cls._ax.text(0.96, y, "—", color=color, fontsize=8,
                                ha='right', va='top', transform=cls._ax.transAxes,
                                fontfamily='monospace', fontweight='bold')
                cls._text_objects[key] = txt

        for key, txt in cls._text_objects.items():
            val = cls._data[key]
            txt.set_text(f"{val:.4f}" if isinstance(val, float) else str(val))

        cls._fig.canvas.draw_idle()
