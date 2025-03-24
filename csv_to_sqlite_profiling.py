#!/usr/bin/env python3
"""
Comprehensive profiling of CSV parsing and SQLite insertion
Compares Python CSV vs Rust CSV parser performance with database operations
"""

import os
import time
import csv
import sqlite3
import tempfile
import psutil
import gc
import argparse
from memory_profiler import memory_usage
import matplotlib.pyplot as plt
import numpy as np

# Import the Rust CSV parser
try:
    from csv_reader import FastCSVParser

    RUST_AVAILABLE = True
except ImportError:
    print("WARNING: Rust CSV parser not available!")
    RUST_AVAILABLE = False


class CSVtoSQLiteProfiler:
    def __init__(self, csv_file, batch_size=5000):
        self.csv_file = csv_file
        self.batch_size = batch_size
        self.process = psutil.Process(os.getpid())
        self.results = {}

        # Get file info
        self.file_size = os.path.getsize(csv_file)

        print(f"File: {csv_file}")
        print(f"Size: {self.format_bytes(self.file_size)}")
        print(f"Batch size: {batch_size}")
        print("-" * 80)

    def format_bytes(self, bytes_count):
        """Format bytes in a human-readable way"""
        for unit in ["B", "KB", "MB", "GB"]:
            if bytes_count < 1024.0:
                return f"{bytes_count:.2f} {unit}"
            bytes_count /= 1024.0
        return f"{bytes_count:.2f} TB"

    def format_time(self, seconds):
        """Format time in a human-readable way"""
        if seconds < 60:
            return f"{seconds:.2f}s"
        minutes = seconds // 60
        seconds %= 60
        return f"{int(minutes)}m {seconds:.2f}s"

    def _reset_state(self):
        """Reset the process state between tests"""
        gc.collect()
        time.sleep(1)  # Allow system to stabilize

    def create_test_db(self):
        """Create a test database in memory"""
        # Using named memory database so we can use it across connections if needed
        conn = sqlite3.connect(":memory:")
        cursor = conn.cursor()

        # Create a test table matching our CSV structure
        # Assuming we have id (string) and amount (numeric) fields
        cursor.execute("""
        CREATE TABLE transactions (
            id TEXT,
            amount INTEGER
        )
        """)

        conn.commit()
        return conn

    def profile_function(self, name, func, *args, **kwargs):
        """
        Profile a function for memory usage, CPU time, and execution time
        """
        self._reset_state()

        # Get baseline
        baseline_mem = self.process.memory_info().rss / (1024 * 1024)  # MB

        # Measure execution time and memory
        start_time = time.time()

        # Run with memory profiling
        mem_usage, result = memory_usage(
            (func, args, kwargs),
            interval=0.1,
            timeout=None,
            retval=True,
            include_children=True,
        )

        end_time = time.time()

        # Calculate metrics
        elapsed = end_time - start_time
        max_mem = max(mem_usage) if mem_usage else 0
        mem_increase = max_mem - baseline_mem

        # Store result
        metrics = {
            "name": name,
            "elapsed": elapsed,
            "memory_mb": mem_increase,
            "result": result,
        }

        self.results[name] = metrics

        # Print immediate results
        print(f"\n{name}:")
        print(f"  Time: {self.format_time(elapsed)}")
        print(f"  Memory: {mem_increase:.2f} MB")
        if isinstance(result, tuple) and len(result) == 2:
            rows, operations = result
            print(f"  Rows processed: {rows:,}")
            print(f"  DB operations: {operations:,}")
            print(f"  Speed: {rows / elapsed:.2f} rows/sec")
            metrics["rows"] = rows
            metrics["operations"] = operations
            metrics["speed"] = rows / elapsed

        return metrics

    def python_csv_to_sqlite(self):
        """
        Read CSV with Python's CSV module and insert into SQLite
        """
        # Create a fresh database
        conn = self.create_test_db()
        cursor = conn.cursor()

        # Read and insert
        rows_processed = 0
        db_operations = 0

        with open(self.csv_file, "r", newline="") as f:
            reader = csv.DictReader(f)

            # Process in batches for better performance
            batch = []
            for row in reader:
                # Extract id and amount (adjust field names as needed)
                id_value = row.get("id", "")
                amount_value = float(row.get("amount", 0))

                batch.append((id_value, amount_value))
                rows_processed += 1

                # Process batch
                if len(batch) >= self.batch_size:
                    cursor.executemany(
                        "INSERT INTO transactions (id, amount) VALUES (?, ?)", batch
                    )
                    conn.commit()
                    db_operations += 1
                    batch = []

        # Insert any remaining rows
        if batch:
            cursor.executemany(
                "INSERT INTO transactions (id, amount) VALUES (?, ?)", batch
            )
            conn.commit()
            db_operations += 1

        # Close connection
        conn.close()

        return rows_processed, db_operations

    def pandas_csv_to_sqlite(self):
        """
        Read CSV with pandas and insert into SQLite
        """
        try:
            import pandas as pd
        except ImportError:
            print("Pandas not installed, skipping pandas test")
            return 0, 0

        # Create fresh database
        conn = self.create_test_db()

        # Read and insert
        df = pd.read_csv(self.csv_file)
        rows_processed = len(df)

        # Insert data
        df.to_sql("transactions", conn, if_exists="append", index=False)

        # Close connection
        conn.close()

        return rows_processed, 1  # 1 operation (bulk insert)

    def rust_csv_to_sqlite(self):
        """
        Read CSV with Rust parser and insert into SQLite
        """
        if not RUST_AVAILABLE:
            print("Rust CSV parser not available, skipping")
            return 0, 0

        # Create fresh database
        conn = self.create_test_db()
        cursor = conn.cursor()

        # Setup parser
        parser = FastCSVParser(self.csv_file, self.batch_size)
        batches = parser.read()

        rows_processed = 0
        db_operations = 0

        # Process each batch
        for batch in batches:
            db_batch = []

            for row in batch:
                # Extract id and amount (adjust field names as needed)
                id_value = row.get("id", "")

                # Handle amount conversion safely
                try:
                    amount_value = float(row.get("amount", 0))
                except (ValueError, TypeError):
                    amount_value = 0.0

                db_batch.append((id_value, amount_value))
                rows_processed += 1

            # Insert batch
            cursor.executemany(
                "INSERT INTO transactions (id, amount) VALUES (?, ?)", db_batch
            )
            conn.commit()
            db_operations += 1

        # Close connection
        conn.close()

        return rows_processed, db_operations

    def run_benchmarks(self):
        """Run all benchmarks"""
        print("Starting benchmarks...")

        # Python CSV to SQLite
        self.profile_function("Python CSV to SQLite", self.python_csv_to_sqlite)

        # Pandas to SQLite (if available)
        try:
            import pandas

            self.profile_function("Pandas to SQLite", self.pandas_csv_to_sqlite)
        except ImportError:
            print("\nSkipping Pandas benchmark (not installed)")

        # Rust CSV to SQLite
        if RUST_AVAILABLE:
            self.profile_function("Rust CSV to SQLite", self.rust_csv_to_sqlite)

        # Generate visualization
        self.generate_visualization()

        # Print summary
        self.print_summary()

    def print_summary(self):
        """Print a summary of all benchmarks"""
        print("\n" + "=" * 80)
        print("PERFORMANCE SUMMARY")
        print("=" * 80)
        print(f"{'Parser':<25} {'Time':<15} {'Memory (MB)':<15} {'Rows/Sec':<15}")
        print("-" * 80)

        # Get baseline for speedup calculation
        baseline = None
        for name, result in self.results.items():
            if name == "Python CSV to SQLite":
                baseline = result["elapsed"]
                break

        if not baseline:
            baseline = 1.0  # Default if no Python benchmark

        # Print results
        for name, result in self.results.items():
            elapsed = result["elapsed"]
            memory = result["memory_mb"]

            speedup = baseline / elapsed if elapsed > 0 else 0

            if "speed" in result:
                speed = result["speed"]
                print(
                    f"{name:<25} {self.format_time(elapsed):<15} {memory:<15.2f} {speed:<15,.2f}"
                )
            else:
                print(f"{name:<25} {self.format_time(elapsed):<15} {memory:<15.2f} -")

            # Print speedup compared to Python
            if name != "Python CSV to SQLite" and baseline:
                print(f"  Speedup vs Python: {speedup:.2f}x")

    def generate_visualization(self):
        """Generate visualization of the results"""
        if len(self.results) < 2:
            print("Not enough data for visualization")
            return

        try:
            names = [r["name"] for r in self.results.values()]
            times = [r["elapsed"] for r in self.results.values()]
            memories = [r["memory_mb"] for r in self.results.values()]
            speeds = [r.get("speed", 0) for r in self.results.values()]

            # Create figure with 2 subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
            fig.suptitle("CSV to SQLite Performance Comparison", fontsize=16)

            # Time and memory bars (combined in one subplot)
            x = np.arange(len(names))
            width = 0.35

            ax1.bar(x - width / 2, times, width, label="Time (s)", color="skyblue")
            ax1.bar(x + width / 2, memories, width, label="Memory (MB)", color="salmon")

            ax1.set_ylabel("Time (s) / Memory (MB)")
            ax1.set_title("Time and Memory Usage")
            ax1.set_xticks(x)
            ax1.set_xticklabels(names, rotation=45, ha="right")
            ax1.legend()
            ax1.grid(axis="y", linestyle="--", alpha=0.7)

            # Speed bars
            ax2.bar(names, speeds, color="lightgreen")
            ax2.set_ylabel("Rows per Second")
            ax2.set_title("Processing Speed")
            ax2.set_xticklabels(names, rotation=45, ha="right")
            ax2.grid(axis="y", linestyle="--", alpha=0.7)

            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.savefig("csv_to_sqlite_performance.png")
            print("\nVisualization saved to 'csv_to_sqlite_performance.png'")

        except Exception as e:
            print(f"Error generating visualization: {e}")


def generate_test_data(output_file, rows=1000):
    """Generate test data if needed"""
    print(f"Generating test CSV with {rows} rows...")

    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "amount"])

        for i in range(rows):
            # Generate a long ID string (like timestamp + random digits)
            id_str = f"202503060000{i:08d}{1000000 + i}"
            amount = 25000 + (i % 1000)  # Some variation in amounts
            writer.writerow([id_str, amount])

    print(f"Test data generated: {output_file}")
    return output_file


def main():
    parser = argparse.ArgumentParser(description="Profile CSV to SQLite performance")
    parser.add_argument("--csv", type=str, help="CSV file to process")
    parser.add_argument(
        "--generate", type=int, help="Generate test CSV with this many rows"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5000,
        help="Batch size for database operations",
    )

    args = parser.parse_args()

    # Determine CSV file
    csv_file = args.csv

    # Generate test data if requested
    if args.generate:
        if not csv_file:
            csv_file = "test_data.csv"
        generate_test_data(csv_file, args.generate)

    if not csv_file or not os.path.exists(csv_file):
        print("Error: No CSV file specified or file doesn't exist")
        parser.print_help()
        return

    # Run profiling
    profiler = CSVtoSQLiteProfiler(csv_file, args.batch_size)
    profiler.run_benchmarks()


if __name__ == "__main__":
    main()
