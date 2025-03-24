#!/usr/bin/env python3
"""
Comprehensive profiling script to compare Rust CSV parser with Python alternatives.
Measures: execution time, CPU usage, memory consumption, and throughput.
"""

import os
import csv
import time
import psutil
import pandas as pd
import gc
import matplotlib.pyplot as plt
from memory_profiler import memory_usage
import argparse
import tempfile

# Import the Rust CSV parser - adjust the import based on your module name
try:
    from csv_reader import FastCSVParser
except ImportError as e:
    print(f"Could not import FastCSVParser: {e}")
    print("Make sure you've built and installed the Rust library correctly.")
    print("Continuing with Python parsers only...")
    RUST_AVAILABLE = False
else:
    RUST_AVAILABLE = True


def generate_test_csv(filepath, num_rows, num_cols):
    """Generate a test CSV file with specified dimensions."""
    print(f"Generating test CSV with {num_rows} rows and {num_cols} columns...")

    with open(filepath, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)

        # Write header
        header = [f"col_{i}" for i in range(num_cols)]
        writer.writerow(header)

        # Write data
        for i in range(num_rows):
            row = [f"value_{i}_{j}" for j in range(num_cols)]
            writer.writerow(row)

    file_size = os.path.getsize(filepath) / (1024 * 1024)  # Size in MB
    print(f"Generated {filepath} ({file_size:.2f} MB)")
    return filepath


class Profiler:
    def __init__(self, csv_path, batch_size=1000):
        self.csv_path = csv_path
        self.batch_size = batch_size
        self.results = []
        self.process = psutil.Process(os.getpid())

    def _reset_state(self):
        """Reset the Python interpreter state as much as possible between runs."""
        gc.collect()
        time.sleep(1)  # Allow system to stabilize

    def profile_func(self, func, name, *args, **kwargs):
        """Profile a function's execution time, CPU and memory usage."""
        self._reset_state()

        # Capture baseline memory
        baseline_mem = self.process.memory_info().rss / (1024 * 1024)  # MB

        # Start monitoring CPU
        self.process.cpu_percent(interval=0.1)  # Reset CPU measurement

        # Measure execution time and peak memory
        start_time = time.time()
        mem_usage, result = memory_usage(
            (func, args, kwargs),
            interval=0.1,
            timeout=None,
            retval=True,
            include_children=True,
            multiprocess=True,
        )
        end_time = time.time()

        # Get final CPU usage
        end_cpu = self.process.cpu_percent(interval=None)

        # Calculate metrics
        elapsed_time = end_time - start_time
        peak_mem = max(mem_usage) if mem_usage else 0
        mem_increase = peak_mem - baseline_mem

        # Count rows processed
        if isinstance(result, int):
            rows_processed = result
        elif (
            isinstance(result, list) and len(result) > 0 and isinstance(result[0], dict)
        ):
            rows_processed = len(result)
        elif (
            isinstance(result, list) and len(result) > 0 and isinstance(result[0], list)
        ):
            # For batch processing, sum up all rows in batches
            rows_processed = sum(len(batch) for batch in result)
        else:
            # Default if we can't determine the row count
            rows_processed = 0

        throughput = rows_processed / elapsed_time if elapsed_time > 0 else 0

        # Store results
        self.results.append(
            {
                "parser": name,
                "elapsed_time": elapsed_time,
                "cpu_percent": end_cpu,
                "memory_increase_mb": mem_increase,
                "peak_memory_mb": peak_mem,
                "rows_processed": rows_processed,
                "throughput": throughput,
            }
        )

        print(
            f"{name} completed in {elapsed_time:.2f}s, processed {rows_processed} rows"
        )
        print(
            f"  CPU: {end_cpu:.1f}%, Memory increase: {mem_increase:.2f} MB, Peak memory: {peak_mem:.2f} MB"
        )
        print(f"  Throughput: {throughput:.2f} rows/second")

        return result

    def python_csv_parser(self):
        """Parse CSV using Python's built-in csv module."""
        rows = []
        with open(self.csv_path, "r", newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                rows.append(row)
        return len(rows)

    def pandas_parser(self):
        """Parse CSV using pandas."""
        df = pd.read_csv(self.csv_path)
        rows = df.to_dict("records")
        return len(rows)

    def pandas_chunked_parser(self):
        """Parse CSV using pandas with chunking."""
        total_rows = 0
        chunks = []
        for chunk in pd.read_csv(self.csv_path, chunksize=self.batch_size):
            chunk_rows = chunk.to_dict("records")
            chunks.append(chunk_rows)
            total_rows += len(chunk_rows)
        return total_rows

    def rust_parser(self):
        """Parse CSV using the Rust FastCSVParser."""
        parser = FastCSVParser(self.csv_path, self.batch_size)
        batches = parser.read()
        return batches

    def run_all(self):
        """Run all parser benchmarks."""
        self.profile_func(self.python_csv_parser, "Python CSV")
        self.profile_func(self.pandas_parser, "Pandas (full)")
        self.profile_func(self.pandas_chunked_parser, "Pandas (chunked)")

        if RUST_AVAILABLE:
            self.profile_func(self.rust_parser, "Rust CSV Parser")

        return self.results

    def generate_report(self, output_dir=None):
        """Generate visualization of the profiling results."""
        if not self.results:
            print("No profiling results to visualize. Run profiling first.")
            return

        if output_dir is None:
            output_dir = tempfile.gettempdir()

        # Convert results to arrays for plotting
        parsers = [r["parser"] for r in self.results]
        times = [r["elapsed_time"] for r in self.results]
        cpus = [r["cpu_percent"] for r in self.results]
        mems = [r["memory_increase_mb"] for r in self.results]
        throughputs = [r["throughput"] for r in self.results]

        # Set up the plots
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("CSV Parser Performance Comparison", fontsize=16)

        # Plot execution time
        axs[0, 0].bar(parsers, times, color="blue")
        axs[0, 0].set_title("Execution Time (lower is better)")
        axs[0, 0].set_ylabel("Time (seconds)")
        axs[0, 0].grid(axis="y", linestyle="--", alpha=0.7)

        # Plot CPU usage
        axs[0, 1].bar(parsers, cpus, color="red")
        axs[0, 1].set_title("CPU Usage (lower is better)")
        axs[0, 1].set_ylabel("CPU %")
        axs[0, 1].grid(axis="y", linestyle="--", alpha=0.7)

        # Plot memory usage
        axs[1, 0].bar(parsers, mems, color="green")
        axs[1, 0].set_title("Memory Increase (lower is better)")
        axs[1, 0].set_ylabel("Memory (MB)")
        axs[1, 0].grid(axis="y", linestyle="--", alpha=0.7)

        # Plot throughput
        axs[1, 1].bar(parsers, throughputs, color="purple")
        axs[1, 1].set_title("Throughput (higher is better)")
        axs[1, 1].set_ylabel("Rows/second")
        axs[1, 1].grid(axis="y", linestyle="--", alpha=0.7)

        # Adjust layout and save
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        output_path = os.path.join(output_dir, "csv_parser_comparison.png")
        plt.savefig(output_path)
        print(f"Performance comparison saved to {output_path}")

        # Also save raw results as CSV
        results_df = pd.DataFrame(self.results)
        csv_path = os.path.join(output_dir, "csv_parser_results.csv")
        results_df.to_csv(csv_path, index=False)
        print(f"Detailed results saved to {csv_path}")

        # Print a summary table
        print("\nPerformance Summary:")
        print("-" * 100)
        print(
            f"{'Parser':<20} {'Time (s)':<12} {'CPU %':<10} {'Memory Increase (MB)':<20} {'Throughput (rows/s)':<20}"
        )
        print("-" * 100)
        for r in self.results:
            print(
                f"{r['parser']:<20} {r['elapsed_time']:<12.2f} {r['cpu_percent']:<10.1f} {r['memory_increase_mb']:<20.2f} {r['throughput']:<20.2f}"
            )

        # Calculate speedup relative to Python CSV
        if len(self.results) > 1:
            baseline = self.results[0]["elapsed_time"]  # Python CSV as baseline
            print("\nSpeedup compared to Python CSV:")
            for r in self.results:
                speedup = (
                    baseline / r["elapsed_time"]
                    if r["elapsed_time"] > 0
                    else float("inf")
                )
                print(f"{r['parser']}: {speedup:.2f}x faster")


def main():
    parser = argparse.ArgumentParser(description="Profile and compare CSV parsers")
    parser.add_argument("--csv", type=str, help="Path to existing CSV file")
    parser.add_argument(
        "--generate", action="store_true", help="Generate a test CSV file"
    )
    parser.add_argument(
        "--rows", type=int, default=100000, help="Number of rows for generated CSV"
    )
    parser.add_argument(
        "--cols", type=int, default=10, help="Number of columns for generated CSV"
    )
    parser.add_argument(
        "--batch-size", type=int, default=5000, help="Batch size for chunked processing"
    )
    parser.add_argument("--output-dir", type=str, help="Directory to save results")

    args = parser.parse_args()

    # Determine CSV file path
    if args.csv:
        csv_path = args.csv
    elif args.generate:
        csv_path = os.path.join(tempfile.gettempdir(), "test_data.csv")
        generate_test_csv(csv_path, args.rows, args.cols)
    else:
        print(
            "Error: Either provide a CSV file path (--csv) or use --generate to create a test file"
        )
        return

    # Create output directory if specified
    output_dir = args.output_dir
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Run profiling
    profiler = Profiler(csv_path, batch_size=args.batch_size)
    profiler.run_all()
    profiler.generate_report(output_dir)


if __name__ == "__main__":
    main()
