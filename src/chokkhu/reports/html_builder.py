from __future__ import annotations

import base64
import os

from chokkhu.core.logger import Logger


class HTMLReportBuilder:
    @staticmethod
    def build(save_dir: str, title: str = "Chokkhu EDA Report"):
        Logger.info(f"Generating HTML Report in {save_dir}...")
        image_files = [f for f in os.listdir(save_dir) if f.endswith(".png")]
        image_files.sort()
        header = (
            '<!DOCTYPE html>\n<html lang="en">\n<head>\n'
            '<meta charset="UTF-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
            f"<title>{title}</title>\n"
            "<style>\n"
            "body { font-family: sans-serif; background: #f4f7f6; color: #333; margin: 0; padding: 20px; }\n"
            "h1 { text-align: center; color: #2c3e50; margin-bottom: 40px; }\n"
            ".container { max-width: 1200px; margin: 0 auto; }\n"
            ".plot-card { background: #fff; border-radius: 8px; margin-bottom: 30px; padding: 20px; text-align: center; }\n"
            ".plot-card img { max-width: 100%; height: auto; border-radius: 4px; }\n"
            ".plot-title { font-size: 1.2em; margin-bottom: 15px; color: #34495e; }\n"
            "</style>\n</head>\n<body>\n"
            f'<div class="container">\n<h1>{title}</h1>\n'
        )
        html_content = header
        for img_file in image_files:
            img_path = os.path.join(save_dir, img_file)
            with open(img_path, "rb") as img_f:
                encoded_string = base64.b64encode(img_f.read()).decode("utf-8")
            display_name = img_file.replace(".png", "").replace("_", " ").title()
            html_content += (
                f'<div class="plot-card">\n'
                f'<div class="plot-title">{display_name}</div>\n'
                f'<img src="data:image/png;base64,{encoded_string}" alt="{display_name}">\n'
                f"</div>\n"
            )
        html_content += "</div>\n</body>\n</html>\n"
        report_path = os.path.join(save_dir, "chokkhu_report.html")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        Logger.info(f"HTML Report generated successfully: {report_path}")
