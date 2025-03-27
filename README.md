# CSV Reader profiling

This repository for testing this library https://github.com/yosephbernandus/csv-reader

## 3 Option To Test Profiling 
1. Read the CSV
```
uv run main.py --csv sample2.csv
```

2. Read CSV and insert to sqlite
```
uv run csv_to_sqlite_profiling.py --csv sample2.csv 
```

2. Read CSV and insert to postgres
```
uv run csv_to_postgres_profiling.py --csv sample2.csv --dbname <db_name> --user <user> --password <password>
```

## Useful resource
- CSV file https://github.com/datablist/sample-csv-fil

