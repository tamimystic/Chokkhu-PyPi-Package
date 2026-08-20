import base64
import os

from chokkhu.core.logger import Logger


class HTMLReportBuilder:
    @staticmethod
    def build(save_dir: str, title: str = "Chokkhu EDA Report"):
        Logger.info(f"Generating HTML Report in {save_dir}...")

        # Get all images in the save_dir
        image_files = [f for f in os.listdir(save_dir) if f.endswith(".png")]
        image_files.sort()

        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{title}</title>
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    background-color: #f4f7f6;
                    color: #333;
                    margin: 0;
                    padding: 20px;
                }}
                h1 {{
                    text-align: center;
                    color: #2c3e50;
                    margin-bottom: 40px;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                }}
                .plot-card {{
                    background: #fff;
                    border-radius: 8px;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                    margin-bottom: 30px;
                    padding: 20px;
                    text-align: center;
                }}
                .plot-card img {{
                    max-width: 100%;
                    height: auto;
                    border-radius: 4px;
                }}
                .plot-title {{
                    font-size: 1.2em;
                    margin-bottom: 15px;
                    color: #34495e;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>{title}</h1>
        """

        for img_file in image_files:
            img_path = os.path.join(save_dir, img_file)
            with open(img_path, "rb") as img_f:
                encoded_string = base64.b64encode(img_f.read()).decode("utf-8")

            display_name = img_file.replace(".png", "").replace("_", " ").title()

            html_content += f"""
                <div class="plot-card">
                    <div class="plot-title">{display_name}</div>
                    <img src="data:image/png;base64,{encoded_string}" alt="{display_name}">
                </div>
            """

        html_content += """
            </div>
        </body>
        </html>
        """

        report_path = os.path.join(save_dir, "chokkhu_report.html")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        Logger.info(f"HTML Report generated successfully: {report_path}")
