import os
import json
import re

def TransformResume(folder_path, output_path):
    """
    Procesa todos los archivos *_resume.txt en la carpeta indicada y guarda las métricas en formato JSON.

    Args:
        folder_path (str): Ruta a la carpeta que contiene los archivos *_resume.txt.
        output_path (str): Ruta donde se guardará el archivo JSON de salida.

    Returns:
        None
    """
    def extract_metrics_from_resume(file_path):
        metrics = []
        model_name = os.path.basename(file_path).split("_")[0]

        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()

        in_analysis = False
        for line in lines:
            if line.strip().startswith("Analysis Data"):
                in_analysis = True
                continue
            if in_analysis:
                match = re.match(r"\s*([A-Za-z]+):\s+([0-9.]+)", line)
                if match:
                    metric = match.group(1).strip()
                    value = float(match.group(2))
                    metrics.append({
                        "Metric": metric,
                        "Model": model_name,
                        "Value": value
                    })
                elif line.strip() == "":
                    break  # Fin de la sección

        return metrics

    all_metrics = []

    for filename in os.listdir(folder_path):
        if filename.endswith("_resume.txt"):
            file_path = os.path.join(folder_path, filename)
            metrics = extract_metrics_from_resume(file_path)
            all_metrics.extend(metrics)

    with open(output_path, 'w', encoding='utf-8') as out_file:
        json.dump(all_metrics, out_file, indent=4)

    print(f"Archivo JSON generado con éxito en: {output_path}")