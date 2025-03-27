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

### Restul
- ![image](https://github.com/yosephbernandus/csv_reader_profiling/blob/master/csv_to_postgres_performance.png)
- ![Screenshot from 2025-03-28 01-36-13](https://github.com/user-attachments/assets/aaa3f9e5-3f20-449a-b3fc-ddd7c60d3ddc)
