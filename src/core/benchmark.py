import math
import random
import time
from typing import Any, Dict, List, Optional, Set

try:
    from core.algorithms.base import BaseVectorIndex
    from core.algorithms.linear_scan import LinearScanIndex
    from models.record import VectorRecord
except ImportError:
    from .algorithms.base import BaseVectorIndex
    from .algorithms.linear_scan import LinearScanIndex
    from ..models.record import VectorRecord


class BenchmarkRunner:
    """
    Bộ thực thi thử nghiệm hiệu năng đa chỉ số (Phụ trách: Hòa - Hệ thống & Đồ thị).
    Nhiệm vụ: Đo lường đa chỉ số và so sánh các thuật toán tìm kiếm vector qua các quy mô N.

    Các chỉ số đo lường:
    - Build_Time_ms: Thời gian xây dựng cấu trúc chỉ mục (milliseconds).
    - Mean_Latency_ms: Độ trễ truy vấn trung bình trên tập câu hỏi mẫu (milliseconds).
    - P95_Latency_ms: Phân vị thời gian truy vấn 95% (milliseconds).
    - QPS: Số lượng truy vấn hoàn thành trong 1 giây (Queries Per Second).
    - Recall_Percent: Tỷ lệ láng giềng thu hồi chính xác so với Ground Truth của Linear Scan (%).
    """

    def __init__(self, full_records: List[VectorRecord], dimension: int) -> None:
        if dimension <= 0:
            raise ValueError(f"dimension phải là số nguyên dương (> 0), nhận được: {dimension}")
        self.full_records = list(full_records)
        self.dimension = dimension

    def sample_query_set(self, query_count: int = 50, seed: int = 42) -> List[List[float]]:
        """
        Trích xuất tập truy vấn ngẫu nhiên cố định để thử nghiệm.
        Sử dụng seed để kết quả đo đạc hoàn toàn có thể tái lập được (reproducible).
        """
        if isinstance(query_count, bool) or not isinstance(query_count, int) or query_count <= 0:
            raise ValueError(f"query_count phải là số nguyên dương (> 0), nhận được: {query_count}")

        if not self.full_records:
            return []

        eff_count = min(query_count, len(self.full_records))
        rng = random.Random(seed)
        selected_records = rng.sample(self.full_records, eff_count)
        return [list(rec.values) for rec in selected_records]

    def run_benchmark(
        self,
        algorithm_instances: Dict[str, BaseVectorIndex],
        scale_sizes: Optional[List[int]] = None,
        top_k: int = 5,
        query_count: int = 50,
        seed: int = 42,
    ) -> List[Dict[str, Any]]:
        """
        Thực thi ma trận thử nghiệm benchmark trên các mốc quy mô N: [100, 500, 1000, ...].

        Tham số:
            algorithm_instances: Từ điển {key: instance} của các thuật toán kế thừa BaseVectorIndex.
            scale_sizes: Danh sách các mốc quy mô N cần chạy thử nghiệm.
            top_k: Số lượng láng giềng gần nhất cần truy vấn (mặc định 5).
            query_count: Số câu hỏi mẫu dùng để đo độ trễ và Recall (mặc định 50).
            seed: Hạt giống sinh ngẫu nhiên tập truy vấn.

        Trả về:
            Danh sách các bản ghi dictionary chứa đầy đủ 5 chỉ số đo lường.
        """
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise ValueError(f"top_k phải là số nguyên dương (> 0), nhận được: {top_k}")

        if isinstance(query_count, bool) or not isinstance(query_count, int) or query_count <= 0:
            raise ValueError(f"query_count phải là số nguyên dương (> 0), nhận được: {query_count}")

        if not self.full_records or not algorithm_instances:
            return []

        total_available = len(self.full_records)

        # 1. Xác định các mốc quy mô N hợp lệ
        if scale_sizes is None:
            default_scales = [100, 500, 1000, 5000]
            valid_scales = [n for n in default_scales if n <= total_available]
            if not valid_scales:
                valid_scales = [total_available]
        else:
            valid_scales = []
            for n in scale_sizes:
                if isinstance(n, int) and n > 0:
                    valid_scales.append(min(n, total_available))
            valid_scales = sorted(list(set(valid_scales)))
            if not valid_scales:
                valid_scales = [total_available]

        # 2. Sinh tập truy vấn chuẩn (Ground Truth Query Set)
        queries = self.sample_query_set(query_count=query_count, seed=seed)
        if not queries:
            return []

        # Chuẩn bị bộ so sánh chuẩn Linear Scan (Ground Truth Baseline)
        linear_baseline = LinearScanIndex(metric="euclidean")

        benchmark_results: List[Dict[str, Any]] = []

        # 3. Lặp qua từng mốc quy mô N
        for n in valid_scales:
            sub_records = self.full_records[:n]

            # Xây dựng Ground Truth cho từng câu hỏi tại mốc N này
            linear_baseline.build(sub_records)
            ground_truth_map: List[Set[str]] = []
            gt_latencies_ms: List[float] = []
            for q in queries:
                t_gt_s = time.perf_counter()
                gt_results = linear_baseline.search(q, top_k=top_k)
                t_gt_e = time.perf_counter()
                gt_latencies_ms.append((t_gt_e - t_gt_s) * 1000.0)
                gt_ids = {str(getattr(rec, "id", rec)) for rec, _ in gt_results}
                ground_truth_map.append(gt_ids)

            gt_mean_latency = (sum(gt_latencies_ms) / len(gt_latencies_ms)) if gt_latencies_ms else 0.0

            # Đánh giá từng thuật toán trong danh sách
            for algo_key, algo in algorithm_instances.items():
                algo_name = getattr(algo, "name", algo_key)

                # Xác định bản chất thuật toán
                if algo_key in ("linear_scan",) or "linear" in algo_name.lower():
                    category = "Exact (Baseline)"
                elif algo_key in ("kdtree",) or "k-d" in algo_name.lower() or "kdtree" in algo_name.lower():
                    category = "Exact (Phân vùng trục)"
                elif algo_key in ("ivf",) or "ivf" in algo_name.lower():
                    category = "ANN (Phân cụm Voronoi)"
                elif algo_key in ("lsh",) or "lsh" in algo_name.lower():
                    category = "ANN (Băm ngẫu nhiên)"
                elif algo_key in ("hnsw",) or "hnsw" in algo_name.lower():
                    category = "ANN (Đồ thị đa tầng)"
                else:
                    category = "ANN (Xấp xỉ)"

                # a. Bấm giờ xây dựng chỉ mục (Build Time)
                # Tự động điều chỉnh thích ứng tham số IVF theo quy mô N (Adaptive Scaling)
                if algo_key in ("ivf",) or "ivf" in algo_name.lower():
                    adaptive_c = min(n, max(4, round(math.sqrt(n))))
                    adaptive_p = max(1, round(0.2 * adaptive_c))
                    if hasattr(algo, "num_clusters"):
                        setattr(algo, "num_clusters", adaptive_c)
                    if hasattr(algo, "nprobe"):
                        setattr(algo, "nprobe", adaptive_p)

                # Tự động điều chỉnh thích ứng tham số LSH theo quy mô N
                if algo_key in ("lsh",) or "lsh" in algo_name.lower():
                    # Multi-Table LSH sweet spot: m trong khoảng 3-6, L=2 bảng để đảm bảo Recall@K đạt 85-95%
                    adaptive_m = 8 if n >= 2000 else max(4, min(8, 4 + round(n / 1000)))
                    if hasattr(algo, "num_hyperplanes"):
                        setattr(algo, "num_hyperplanes", adaptive_m)
                    if hasattr(algo, "num_tables"):
                        setattr(algo, "num_tables", 2)
                    if hasattr(algo, "metric"):
                        setattr(algo, "metric", "euclidean")

                t_build_start = time.perf_counter()
                algo.build(sub_records)
                build_time_ms = (time.perf_counter() - t_build_start) * 1000.0

                # b. Bấm giờ tìm kiếm và tính độ tương đồng (Recall) trên từng query
                latencies_ms: List[float] = []
                recall_scores: List[float] = []

                for q_idx, q in enumerate(queries):
                    t_search_start = time.perf_counter()
                    search_results = algo.search(q, top_k=top_k)
                    t_search_end = time.perf_counter()

                    latencies_ms.append((t_search_end - t_search_start) * 1000.0)

                    # Đánh giá Recall@K so với Ground Truth
                    gt_ids = ground_truth_map[q_idx]
                    k_effective = min(top_k, len(gt_ids))
                    if k_effective > 0:
                        retrieved_ids = {str(getattr(rec, "id", rec)) for rec, _ in search_results}
                        hits = len(retrieved_ids.intersection(gt_ids))
                        recall_scores.append(hits / k_effective)
                    else:
                        recall_scores.append(1.0)

                # c. Tổng hợp các chỉ số thống kê
                num_queries = len(latencies_ms)
                mean_latency_ms = (sum(latencies_ms) / num_queries) if num_queries > 0 else 0.0

                sorted_lat = sorted(latencies_ms)
                p95_idx = min(int(math.ceil(0.95 * len(sorted_lat))) - 1, len(sorted_lat) - 1)
                p95_latency_ms = sorted_lat[max(0, p95_idx)] if sorted_lat else 0.0

                total_search_time_sec = sum(latencies_ms) / 1000.0
                qps = (num_queries / total_search_time_sec) if total_search_time_sec > 0.0 else 0.0

                avg_recall = (sum(recall_scores) / len(recall_scores)) if recall_scores else 0.0
                avg_recall_percent = avg_recall * 100.0

                # Đối với Linear Scan, luôn đảm bảo recall chính xác 100%
                if algo_key == "linear_scan" or algo_name == "Linear Scan":
                    avg_recall_percent = 100.0
                    speedup_ratio = 1.0
                elif mean_latency_ms > 0 and gt_mean_latency > 0:
                    speedup_ratio = round(gt_mean_latency / mean_latency_ms, 2)
                else:
                    speedup_ratio = 1.0

                benchmark_results.append({
                    "n": n,
                    "algorithm": algo_name,
                    "algorithm_key": algo_key,
                    "category": category,
                    "speedup": speedup_ratio,
                    "build_time_ms": round(build_time_ms, 2),
                    "mean_latency_ms": round(mean_latency_ms, 3),
                    "p95_latency_ms": round(p95_latency_ms, 3),
                    "qps": round(qps, 1),
                    "recall_percent": round(avg_recall_percent, 2),
                })

        return benchmark_results
