import unittest

from tests.task_resource_monitor import summarize_task


class TaskResourceMonitorTests(unittest.TestCase):
    def test_summarizes_cpu_and_gpu_samples_for_task_window(self):
        window = {
            "task_id": "task-1",
            "trace_id": "trace-1",
            "task": "RAG Analysis",
            "node": "rag_analysis_node",
            "status": "success",
            "start_ns": 0,
            "end_ns": 2_000_000_000,
        }
        samples = [
            {
                "start_ns": 0,
                "end_ns": 1_000_000_000,
                "interval_seconds": 1.0,
                "app_cpu_seconds": 0.1,
                "ollama_cpu_seconds": 1.0,
                "host_cpu_capacity_pct": 10.0,
                "gpu": {
                    "0": {"utilization_pct": 50.0, "memory_mb": 100.0},
                    "1": {"utilization_pct": 25.0, "memory_mb": 80.0},
                },
            },
            {
                "start_ns": 1_000_000_000,
                "end_ns": 2_000_000_000,
                "interval_seconds": 1.0,
                "app_cpu_seconds": 0.2,
                "ollama_cpu_seconds": 2.0,
                "host_cpu_capacity_pct": 20.0,
                "gpu": {
                    "0": {"utilization_pct": 0.0, "memory_mb": 200.0},
                    "1": {"utilization_pct": 75.0, "memory_mb": 180.0},
                },
            },
        ]

        result = summarize_task(window, samples, ["0", "1"], 1.0)

        self.assertEqual(result["wall_time_ms"], 2000.0)
        self.assertEqual(result["app_cpu_time_ms"], 300.0)
        self.assertEqual(result["ollama_cpu_time_ms"], 3000.0)
        self.assertEqual(result["host_cpu_capacity_pct"], 15.0)
        self.assertEqual(result["sample_coverage_pct"], 100.0)
        self.assertEqual(result["gpu"]["0"]["active_time_pct"], 50.0)
        self.assertEqual(result["gpu"]["0"]["average_utilization_pct"], 25.0)
        self.assertEqual(result["gpu"]["0"]["peak_memory_mb"], 200.0)
        self.assertEqual(result["gpu"]["1"]["active_time_pct"], 100.0)
        self.assertEqual(result["gpu"]["1"]["average_utilization_pct"], 50.0)


if __name__ == "__main__":
    unittest.main()
