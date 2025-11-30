import sqlite3
import pandas as pd
from pathlib import Path

db_path = Path("logs/audit.db")

if db_path.exists():
    conn = sqlite3.connect(str(db_path))
    
    # Get all predictions
    df = pd.read_sql_query("SELECT * FROM predictions ORDER BY created_at DESC LIMIT 50", conn)
    
    print("=" * 80)
    print("Recent Audit Entries (Last 50)")
    print("=" * 80)
    print(df.to_string())
    
    # Get summary statistics
    print("\n" + "=" * 80)
    print("Summary Statistics")
    print("=" * 80)
    
    summary = pd.read_sql_query("""
        SELECT 
            predicted_label,
            COUNT(*) as count,
            AVG(probability) as avg_probability
        FROM predictions 
        GROUP BY predicted_label
    """, conn)
    print(summary.to_string())
    
    conn.close()
else:
    print("audit.db not found. Make predictions first to generate the database.")