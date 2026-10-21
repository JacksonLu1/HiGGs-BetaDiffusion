from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path("/data2/wenb/ylu/HiGGs-GBD/HiGGs-main")
DIAGNOSTICS = ROOT / "tools" / "diagnostics"
DGD = ROOT / "dgd"

sys.path.insert(0, str(DIAGNOSTICS))
sys.path.insert(0, str(DGD))

import cora_beta_h1_teacher_forced as teacher

base = teacher.base

CHECKPOINTS = {
    "epoch649_validation_best": (
        ROOT
        / "outputs/2026-07-29/21-46-00/checkpoints"
        / "cora_beta_h1_r2_e1000/epoch=649.ckpt"
    ),
    "epoch849_second_best": (
        ROOT
        / "outputs/2026-07-29/21-46-00/checkpoints"
        / "cora_beta_h1_r2_e1000/epoch=849.ckpt"
    ),
    "epoch999_last": (
        ROOT
        / "outputs/2026-07-29/21-46-00/checkpoints"
        / "cora_beta_h1_r2_e1000/last.ckpt"
    ),
}

TIMES = (0.50, 0.75, 1.00)

OUTPUT_DIR = (
    ROOT / "higgs/outputs"
    / "cora_beta_h1_checkpoint_time_comparison_seed20260807"
)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    selected, selection_histogram = (
        base.extract_validation_endpoints()
    )

    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )

    report = {
        "purpose": (
            "Validation-only fixed-n/fixed-y comparison. "
            "Formal checkpoint selection remains based on "
            "val/h1_total and does not use test data."
        ),
        "times": list(TIMES),
        "selected_count": len(selected),
        "selection_histogram": selection_histogram,
        "checkpoints": {},
    }

    for checkpoint_name, checkpoint_path in CHECKPOINTS.items():
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)

        checkpoint = base._load_checkpoint_cpu(
            checkpoint_path
        )

        model = (
            base.ConditionalGraphBetaDiffusionH1
            .load_from_checkpoint(
                str(checkpoint_path),
                map_location="cpu",
            )
        )
        model = model.to(device)
        model.eval()

        print("\n" + "=" * 120)
        print("checkpoint:", checkpoint_name)
        print("path:", checkpoint_path)
        print("epoch:", checkpoint.get("epoch"))
        print("global_step:", checkpoint.get("global_step"))
        print("device:", device)

        checkpoint_result = {
            "path": str(checkpoint_path),
            "epoch": checkpoint.get("epoch"),
            "global_step": checkpoint.get("global_step"),
            "times": {},
        }

        for t_value in TIMES:
            rows = []

            for record_id, record in enumerate(selected):
                rows.append(
                    teacher.evaluate_endpoint(
                        model,
                        record,
                        t_value,
                        record_id,
                        device,
                    )
                )

            checkpoint_result["times"][str(t_value)] = (
                teacher.summarize_time(
                    rows, model.edge_threshold
                )
            )

            item = checkpoint_result["times"][
                str(t_value)
            ]

            print(
                f"t={t_value:.2f} "
                f"AUROC={item['auroc']:.6f} "
                f"AUPRC={item['auprc']:.6f} "
                f"pos={item['positive_probability_mean']:.6f} "
                f"neg={item['negative_probability_mean']:.6f} "
                f"components={item['components_mean']:.3f} "
                f"clustering={item['predicted_clustering_mean']:.6f} "
                f"node_acc={item['node_accuracy_mean']:.6f} "
                f"cond_match="
                f"{item['condition_majority_match_rate']:.6f}"
            )

        report["checkpoints"][
            checkpoint_name
        ] = checkpoint_result

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    json_path = OUTPUT_DIR / "summary.json"
    json_path.write_text(
        json.dumps(
            report, indent=2, ensure_ascii=False
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 180)
    print("Checkpoint × teacher-forced time comparison")
    print("=" * 180)
    print(
        f"{'checkpoint':>26} {'t':>6} "
        f"{'AUROC':>9} {'AUPRC':>9} "
        f"{'pos prob':>10} {'neg prob':>10} "
        f"{'components':>11} {'clustering':>11} "
        f"{'node acc':>9} {'cond match':>11}"
    )
    print("-" * 180)

    for checkpoint_name, checkpoint_result in (
        report["checkpoints"].items()
    ):
        for t_value in TIMES:
            item = checkpoint_result["times"][
                str(t_value)
            ]
            print(
                f"{checkpoint_name:>26} "
                f"{t_value:6.2f} "
                f"{item['auroc']:9.4f} "
                f"{item['auprc']:9.4f} "
                f"{item['positive_probability_mean']:10.4f} "
                f"{item['negative_probability_mean']:10.4f} "
                f"{item['components_mean']:11.3f} "
                f"{item['predicted_clustering_mean']:11.4f} "
                f"{item['node_accuracy_mean']:9.3f} "
                f"{item['condition_majority_match_rate']:11.3f}"
            )

    print("\nJSON:", json_path)


if __name__ == "__main__":
    main()
