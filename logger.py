# logger.py
import matplotlib.pyplot as plt

class SimLogger:
    def __init__(self):
        self.data = {}
        self.fig, self.ax = plt.subplots(figsize=(3, 4))
        self.fig.patch.set_facecolor('#1e1e2e')
        self.ax.set_facecolor('#1e1e2e')
        self.ax.axis('off')
        self.ax.set_title("Logger", color='white', fontsize=10)
        self.text_objects = {}

    def recordData(self, key, value):
        self.data[key] = value

    def update(self):
        colors = ['#56d4f5', '#f5a623', '#a78bfa', '#7defa5', '#f87171']
        n = len(self.data)
        if n == 0:
            return

        # First time seeing a new key, rebuild all text
        if set(self.data.keys()) != set(self.text_objects.keys()):
            self.ax.cla()
            self.ax.axis('off')
            self.ax.set_facecolor('#1e1e2e')
            self.ax.set_title("Logger", color='white', fontsize=10)
            self.text_objects.clear()
            for i, key in enumerate(self.data):
                color = colors[i % len(colors)]
                y = 1 - (i + 0.5) / n
                self.ax.text(0.5, y + 0.06, key, color='#aaaacc',
                             fontsize=9, ha='center', transform=self.ax.transAxes)
                txt = self.ax.text(0.5, y - 0.06, "—", color=color,
                                   fontsize=14, fontweight='bold',
                                   ha='center', transform=self.ax.transAxes)
                self.text_objects[key] = txt

        # Update values
        for key, txt in self.text_objects.items():
            val = self.data[key]
            txt.set_text(f"{val:.3f}" if isinstance(val, float) else str(val))

        self.fig.canvas.draw_idle()