import subprocess
import json
import os
from pathlib import Path
import pandas as pd
import time
import torch

# ============================================================
# LISTA DE MODELOS E CONFIGURAÇÕES
# ============================================================
MODELS = [
    # BioBERTpt
    "pucpr/biobertpt-all",
    "pucpr/biobertpt-clin",
    "pucpr/biobertpt-bio",
    # mmBERT
    "jhu-clsp/mmBERT-base",
    "jhu-clsp/mmBERT-small",
    # ModernBERT
    "answerdotai/ModernBERT-base",
    "answerdotai/ModernBERT-large",
    # BERTimbau
    "neuralmind/bert-base-portuguese-cased",
    "neuralmind/bert-large-portuguese-cased",
]

ANNOTATION_TYPES = ["IO", "BIO"]

RESULTS_DIR = Path("results_fast")
SUMMARY_PATH = RESULTS_DIR / "benchmark_summary.csv"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

summary_data = []

# ============================================================
# LOOP PRINCIPAL DE EXECUÇÃO
# ============================================================
for model in MODELS:
    for annot_type in ANNOTATION_TYPES:
        start_time = time.time()
        print(f"\n Rodando modelo {model} ({annot_type})...\n")

        # comando CLI correto com parâmetros explícitos
        cmd = [
            "uv",
            "run",
            "python",
            "-m",
            "scripts.run_benchmark",
            "--model_id",
            model,
            "--annotation_type",
            annot_type,
        ]

        # exibir logs em tempo real do processo
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        for line in process.stdout:
            print(line.strip())

        process.wait()
        elapsed = time.time() - start_time

        # local da pasta correta: inclui o tipo IO/BIO
        model_dir = RESULTS_DIR / f"{model.replace('/', '_')}_{annot_type}"
        metrics_path = model_dir / "metrics.json"

        # leitura das métricas
        if metrics_path.exists():
            with open(metrics_path, "r", encoding="utf-8") as f:
                metrics = json.load(f)
            summary_data.append(
                {
                    "Modelo": model,
                    "Tipo de anotação": annot_type,
                    "Macro-F1": metrics.get("eval_macro_f1"),
                    "Micro-F1": metrics.get("eval_micro_f1"),
                    "Macro-AUC": metrics.get("eval_macro_auc"),
                    "Micro-AUC": metrics.get("eval_micro_auc"),
                    "Tempo (s)": round(elapsed, 1),
                }
            )
        else:
            summary_data.append(
                {
                    "Modelo": model,
                    "Tipo de anotação": annot_type,
                    "Macro-F1": None,
                    "Micro-F1": None,
                    "Macro-AUC": None,
                    "Micro-AUC": None,
                    "Tempo (s)": round(elapsed, 1),
                }
            )
            print(f" Métricas não encontradas para {model} ({annot_type})")

        torch.cuda.empty_cache()

# ============================================================
# SALVAR RELATÓRIO FINAL
# ============================================================
df_summary = pd.DataFrame(summary_data)
df_summary.to_csv(SUMMARY_PATH, index=False)
print(f"\n Benchmark final salvo em: {SUMMARY_PATH.resolve()}")
print(df_summary)
