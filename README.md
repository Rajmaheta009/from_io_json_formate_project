
# 🧠 Multi-Database Form Metadata Extractor

This project extracts and joins form metadata from multiple SQL Server databases using Python (SQLAlchemy + Pandas). The resulting dataset is unified and ready for analysis, export, or further AI processing (e.g., JSON generation, form builders, ML pipelines).

---

## 📂 Project Structure

```
from_io_json_convert_model/
│
├── sql_connection/
│   └── db.py                # Main script to connect, join, and combine data

├── requirements.txt         # Python dependencies
└── README.md                # This file
```

---

## ⚙️ Requirements

- Python 3.8+
- SQL Server (local or remote)
- ODBC Driver 17 for SQL Server

### Python Packages:

```
pandas
sqlalchemy
pyodbc
```

Install with:

```bash
pip install -r requirements.txt
```

---

## 🛠️ Database Tables Used

The following tables are joined across 4 databases:

| Table Name                | Description                        |
|--------------------------|------------------------------------|
| `TABMD_AppScreen`        | Contains screen-level info         |
| `TABMD_DataSourceQueries`| Links screens to data sources      |
| `TABMD_AppObject`        | Metadata for form objects          |
| `TABMD_AppFields`        | Field-level metadata               |

---

## 🔗 SQL Join Logic

```sql
SELECT 
    scr.Name,
    scr.Description,
    scr.DataSourceQueryID,
    dsq.AppObject AS AppObjectId,
    ao.DisplayName,
    ao.SystemDBTableName,
    ao.Child_Relationship,
    ao.ObjectType,
    ao.AppObjectConfiguration,
    fld.FieldConfiguration,
    dsq.FilterLogic,
    dsq.Configurations,
    scr.Container
FROM dbo.TABMD_AppScreen scr
LEFT JOIN dbo.TABMD_DataSourceQueries dsq ON scr.DataSourceQueryID = dsq.Id
LEFT JOIN dbo.TABMD_AppObject ao ON dsq.AppObject = ao.Id
LEFT JOIN dbo.TABMD_AppFields fld ON ao.Id = fld.AppObjectId
```

---

## 🧪 Databases Queried

The following databases are queried individually:

- `QA_TAB`
- `TAB_MT_3010`
- `TAB_Rewrite`
- `TAB_RM`

Each query result is stored in a DataFrame (`df_qatab`, `df_mt3010`, etc.) and later merged.

---

## 🐍 Python Data Fetch Logic

```python
engine = create_engine(f"mssql+pyodbc://@RAjPC\SQLEXPRESS/DATABASE_NAME?driver=ODBC+Driver+17+for+SQL+Server&trusted_connection=yes")

df = pd.read_sql(JOIN_QUERY, engine)
df['SourceDB'] = 'DATABASE_NAME'
```

All DataFrames are finally combined using:

```python
df_final = pd.concat([df_qatab, df_mt3010, df_rewrite, df_rm], ignore_index=True)
```

---

## 📈 Output

- Combined shape: **14,882 rows × 14 columns**
- Columns include screen names, display names, object types, field configs, etc.
- Each row includes a `SourceDB` column indicating its origin

---

## 💾 Save Output

To save the result:

```python
df_final.to_csv("output/combined_form_data.csv", index=False)
```

---

## 🔮 Next Steps (Optional)

- 🧠 Generate Form.io-compatible JSON from this data
- 🧱 Build a Streamlit dashboard to view/edit form definitions
- 🤖 Train ML models for automatic form generation or analysis

---

## 👨‍💻 Author

Raj | Python + Data Engineering | `from_io_json_convert_model`

---

## 📜 License

MIT License

