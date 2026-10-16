from pathlib import Path
import sys
from typing import List
import unittest

root_dir = Path(__file__).resolve().parent.parent
src_dir = root_dir / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from core.algorithms.kdtree import KDTreeIndex
from core.algorithms.linear_scan import LinearScanIndex
from core.benchmark import BenchmarkRunner
from models.record import VectorRecord


class TestBenchmarkRunner(unittest.TestCase):
    """Bộ kiểm thử đơn vị toàn diện cho BenchmarkRunner."""

    def setUp(self) -> None:
        self.dim = 3
        # Tạo tập 30 bản ghi mẫu
        self.records: List[VectorRecord] = [
            VectorRecord(f"id_{i}", f"word_{i}", [float(i), float(i * 2), float(i * 3)])
            for i in range(1, 31)
        ]
        self.runner = BenchmarkRunner(full_records=self.records, dimension=self.dim)

    # 1. Kiểm tra khởi tạo và kiểm tra tham số
    def test_init_validation(self) -> None:
        with self.assertRaises(ValueError):
            BenchmarkRunner(full_records=self.records, dimension=0)
        with self.assertRaises(ValueError):
            BenchmarkRunner(full_records=self.records, dimension=-3)

    # 2. Kiểm tra sample_query_set
    def test_sample_query_set_empty(self) -> None:
        empty_runner = BenchmarkRunner(full_records=[], dimension=3)
        self.assertEqual(empty_runner.sample_query_set(query_count=10), [])

    def test_sample_query_set_invalid_count(self) -> None:
        with self.assertRaises(ValueError):
            self.runner.sample_query_set(query_count=0)
        with self.assertRaises(ValueError):
            self.runner.sample_query_set(query_count=-5)
        with self.assertRaises(ValueError):
            self.runner.sample_query_set(query_count=True)  # type: ignore

    def test_sample_query_set_reproducibility(self) -> None:
        q1 = self.runner.sample_query_set(query_count=5, seed=123)
        q2 = self.runner.sample_query_set(query_count=5, seed=123)
        self.assertEqual(q1, q2)
        self.assertEqual(len(q1), 5)

        # Khác seed phải cho kết quả khác
        q3 = self.runner.sample_query_set(query_count=5, seed=999)
        self.assertNotEqual(q1, q3)

    def test_sample_query_set_capped_by_total_records(self) -> None:
        # Yêu cầu 50 câu hỏi nhưng tập dữ liệu chỉ có 30
        q = self.runner.sample_query_set(query_count=50)
        self.assertEqual(len(q), len(self.records))

    # 3. Kiểm tra run_benchmark
    def test_run_benchmark_empty_records(self) -> None:
        empty_runner = BenchmarkRunner(full_records=[], dimension=3)
        algos = {"linear_scan": LinearScanIndex(metric="euclidean")}
        self.assertEqual(empty_runner.run_benchmark(algos), [])

    def test_run_benchmark_empty_algorithms(self) -> None:
        self.assertEqual(self.runner.run_benchmark(algorithm_instances={}), [])

    def test_run_benchmark_invalid_params(self) -> None:
        algos = {"linear_scan": LinearScanIndex(metric="euclidean")}
        with self.assertRaises(ValueError):
            self.runner.run_benchmark(algos, top_k=0)
        with self.assertRaises(ValueError):
            self.runner.run_benchmark(algos, query_count=-1)

    def test_run_benchmark_execution_and_metrics(self) -> None:
        algos = {
            "linear_scan": LinearScanIndex(metric="euclidean"),
            "kdtree": KDTreeIndex(dimension=self.dim),
        }
        scales = [10, 20]
        results = self.runner.run_benchmark(
            algorithm_instances=algos,
            scale_sizes=scales,
            top_k=3,
            query_count=5,
            seed=42,
        )

        # 2 mốc N * 2 thuật toán = 4 dòng kết quả
        self.assertEqual(len(results), 4)

        for row in results:
            self.assertIn(row["n"], scales)
            self.assertIn("algorithm", row)
            self.assertIn("category", row)
            self.assertIn("speedup", row)
            self.assertIn("build_time_ms", row)
            self.assertIn("mean_latency_ms", row)
            self.assertIn("p95_latency_ms", row)
            self.assertIn("qps", row)
            self.assertIn("recall_percent", row)

            self.assertIsInstance(row["category"], str)
            self.assertGreaterEqual(row["speedup"], 0.0)
            self.assertGreaterEqual(row["build_time_ms"], 0.0)
            self.assertGreaterEqual(row["mean_latency_ms"], 0.0)
            self.assertGreaterEqual(row["p95_latency_ms"], row["mean_latency_ms"] * 0.5)
            self.assertGreaterEqual(row["qps"], 0.0)
            self.assertGreaterEqual(row["recall_percent"], 0.0)
            self.assertLessEqual(row["recall_percent"], 100.0)

            # Linear Scan luôn phải có Recall = 100% và Speedup = 1.0x
            if row["algorithm_key"] == "linear_scan":
                self.assertEqual(row["recall_percent"], 100.0)
                self.assertEqual(row["speedup"], 1.0)
                self.assertIn("Exact", row["category"])


if __name__ == "__main__":
    unittest.main()
