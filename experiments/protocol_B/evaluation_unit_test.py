import csv
import json
import math
import tempfile
from pathlib import Path

import evaluate_gene_ranking as ev


BASE = Path("/mnt/e/codex_file/driver_label_protocol")
OUT = BASE / "evaluation_unit_test.txt"


def write(path, text):
    Path(path).write_text(text, encoding="utf-8")


def read_metrics(path):
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def assert_close(a, b, eps=1e-9):
    assert abs(float(a) - float(b)) <= eps, f"{a} != {b}"


def run():
    results = []
    with tempfile.TemporaryDirectory(dir=BASE) as tmp:
        tmp = Path(tmp)
        ppi = tmp / "ppi.txt"
        write(ppi, "A B\nB C\nC D\nD E\nE F\n")

        # File-order ranking with duplicate and invalid genes.
        ranking = tmp / "ranking_order.csv"
        write(ranking, "Gene\nA\nB\n?\nC\nD\nA\nE\n")
        test = tmp / "test.csv"
        validation = tmp / "validation.csv"
        train = tmp / "train.csv"
        write(test, "Gene\nA\nD\n")
        write(validation, "Gene\nB\nE\n")
        write(train, "Gene\nC\n")
        out = tmp / "eval_order"
        meta = ev.main(
            [
                "--ranking",
                str(ranking),
                "--test-labels",
                str(test),
                "--validation-labels",
                str(validation),
                "--train-labels",
                str(train),
                "--output-dir",
                str(out),
                "--k-values",
                "2,10",
                "--ppi-path",
                str(ppi),
            ]
        )
        rows = read_metrics(out / "ranking_metrics_by_k.csv")
        test_k2 = [r for r in rows if r["Label_set"] == "test" and r["K"] == "2"][0]
        assert int(test_k2["HitCount"]) == 1
        assert_close(test_k2["Precision"], 0.5)
        assert_close(test_k2["Recall"], 0.5)
        expected_ndcg = 1.0 / (1.0 + 1.0 / math.log2(3))
        assert_close(test_k2["NDCG"], expected_ndcg)
        assert meta["duplicate_genes_removed"] == 1
        test_k10 = [r for r in rows if r["Label_set"] == "test" and r["K"] == "10"][0]
        assert int(test_k10["Effective_K"]) == 5
        summary_rows = read_metrics(out / "ranking_label_rank_summary.csv")
        test_summary = [r for r in summary_rows if r["Label_set"] == "test"][0]
        assert_close(test_summary["Mean_rank"], 2.5)
        assert_close(test_summary["Median_rank"], 2.5)
        assert int(float(test_summary["Min_rank"])) == 1
        assert int(float(test_summary["Max_rank"])) == 4
        results.append("file_order_metrics_duplicate_invalid_k_gt_len: PASS")

        # Rank column priority over Q_value.
        ranking_rank = tmp / "ranking_rank.csv"
        write(ranking_rank, "Gene,Rank,Q_value\nD,1,0.1\nA,2,0.9\nB,3,1.0\n")
        out_rank = tmp / "eval_rank"
        meta_rank = ev.main(
            [
                "--ranking",
                str(ranking_rank),
                "--test-labels",
                str(test),
                "--output-dir",
                str(out_rank),
                "--k-values",
                "1",
                "--ppi-path",
                str(ppi),
            ]
        )
        assert meta_rank["sort_mode"] == "Rank_ascending"
        rank_rows = read_metrics(out_rank / "ranking_metrics_by_k.csv")
        assert int(rank_rows[0]["HitCount"]) == 1
        results.append("rank_column_priority: PASS")

        # Q_value descending when no Rank.
        ranking_q = tmp / "ranking_q.csv"
        write(ranking_q, "Gene,Q_value\nB,0.1\nD,0.9\nA,0.2\n")
        out_q = tmp / "eval_q"
        meta_q = ev.main(
            [
                "--ranking",
                str(ranking_q),
                "--test-labels",
                str(test),
                "--output-dir",
                str(out_q),
                "--k-values",
                "1",
                "--ppi-path",
                str(ppi),
            ]
        )
        assert meta_q["sort_mode"] == "Q_value_descending"
        q_rows = read_metrics(out_q / "ranking_metrics_by_k.csv")
        assert int(q_rows[0]["HitCount"]) == 1
        results.append("q_value_descending: PASS")

        # Train/Test overlap must fail.
        train_overlap = tmp / "train_overlap.csv"
        write(train_overlap, "Gene\nA\n")
        try:
            ev.main(
                [
                    "--ranking",
                    str(ranking),
                    "--test-labels",
                    str(test),
                    "--train-labels",
                    str(train_overlap),
                    "--output-dir",
                    str(tmp / "eval_overlap"),
                    "--ppi-path",
                    str(ppi),
                ]
            )
            raise AssertionError("Expected Train/Test overlap error")
        except ValueError as exc:
            assert "Train/Test label overlap" in str(exc)
        results.append("train_test_overlap_error: PASS")

        # Empty test labels must fail.
        empty_test = tmp / "empty_test.csv"
        write(empty_test, "Gene\n?\n")
        try:
            ev.main(
                [
                    "--ranking",
                    str(ranking),
                    "--test-labels",
                    str(empty_test),
                    "--output-dir",
                    str(tmp / "eval_empty"),
                    "--ppi-path",
                    str(ppi),
                ]
            )
            raise AssertionError("Expected empty test error")
        except ValueError as exc:
            assert "Test label set is empty" in str(exc)
        results.append("empty_test_error: PASS")

    text = "evaluation unit tests: PASS\n" + "\n".join(results) + "\n"
    OUT.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    run()
