import os
import matplotlib.pyplot as plt
import seaborn as sns

class PlotVisualizer:
    @staticmethod
    def setup_theme():
        sns.set_theme(style="whitegrid")
        plt.rcParams.update({
            'font.size': 12,
            'axes.titlesize': 16,
            'axes.labelsize': 14,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'legend.fontsize': 12,
            'figure.titlesize': 18
        })

    @staticmethod
    def save_and_show(fig, filename: str, save_dir: str, save_reports: bool):
        plt.tight_layout()
        if save_reports:
            os.makedirs(save_dir, exist_ok=True)
            fig.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches='tight')
        plt.show(block=False)
        plt.pause(1)
        plt.close(fig)
