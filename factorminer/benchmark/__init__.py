"""Benchmark runners for paper-faithful and Helix research evaluation."""

from factorminer.benchmark.runtime import (
    BenchmarkManifest,
    build_benchmark_library,
    evaluate_frozen_set,
    load_benchmark_dataset,
    run_ablation_memory_benchmark,
    run_benchmark_suite,
    run_cost_pressure_benchmark,
    run_efficiency_benchmark,
    run_table1_benchmark,
    select_frozen_top_k,
)
from factorminer.benchmark.helix_benchmark import (
    HelixBenchmark,
    BenchmarkResult,
    MethodResult,
    DMTestResult,
    StatisticalComparisonTests,
    SpeedBenchmark,
    OperatorSpeedResult,
    PipelineSpeedResult,
)
from factorminer.benchmark.ablation import (
    AblationStudy,
    AblationResult,
    AblatedMethodRunner,
    ABLATION_CONFIGS,
    ABLATION_LABELS,
    run_full_ablation_study,
)

__all__ = [
    # legacy runtime benchmark
    "BenchmarkManifest",
    "build_benchmark_library",
    "evaluate_frozen_set",
    "load_benchmark_dataset",
    "run_ablation_memory_benchmark",
    "run_benchmark_suite",
    "run_cost_pressure_benchmark",
    "run_efficiency_benchmark",
    "run_table1_benchmark",
    "select_frozen_top_k",
    # helix benchmark
    "HelixBenchmark",
    "BenchmarkResult",
    "MethodResult",
    "DMTestResult",
    "StatisticalComparisonTests",
    "SpeedBenchmark",
    "OperatorSpeedResult",
    "PipelineSpeedResult",
    # ablation
    "AblationStudy",
    "AblationResult",
    "AblatedMethodRunner",
    "ABLATION_CONFIGS",
    "ABLATION_LABELS",
    "run_full_ablation_study",
]
