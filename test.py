import sqlite3
from csv_reader import CSVParser


def create_test_db():
    """Create a test database in memory"""
    # Using named memory database so we can use it across connections if needed
    conn = sqlite3.connect("tutorial.db")
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


def rust_csv_to_sqlite():
    """
    Read CSV with Rust parser and insert into SQLite
    """

    # Create fresh database
    conn = create_test_db()
    cursor = conn.cursor()

    # Setup parser
    parser = CSVParser("sample2.csv", 5000)
    batches = parser.read()
    # batches = parser.read_chunk(1, 1000)
    print("Batches: ", len(batches))
    print("Count: ", parser.count_rows())
    print("Estimate bytes per row", parser.estimate_bytes_per_row())
    print("Estimate rows per batch", parser.get_file_info())
    # print("Read chunks: ", parser.read_chunks())

    # print(batches)

    rows_processed = 0
    db_operations = 0

    # Process each batch
    for batch in batches:
        db_batch = []

        # For chunk
        # id_value = batch.get("id", "")
        # amount_value = batch.get("amount", 0)
        # db_batch.append((id_value, amount_value))
        # rows_processed += 1
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


rust_csv_to_sqlite()
