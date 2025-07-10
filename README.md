
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
# 🧾 Form.io JSON Generation — Data Preprocessing & Training Pipeline

This project sets up a full data preprocessing and model training pipeline to train a T5-based sequence-to-sequence model (e.g., `google/flan-t5-base`) that generates Form.io-compatible JSON structures from structured tabular input data.

---

## 📂 Directory Structure
project/
│
├── sql_connection/
│ └── db.py # Contains df_final DataFrame loaded from SQL Server
│
├── logs/
│ └── training_log.txt # Training logs written here
│
├── model/ # Model outputs will be saved here
│
├── main.py # Main preprocessing and training script
├── README.md # You're reading it


---

## 🧠 Objective

Train a model that can generate structured Form.io JSON schemas from input text that combines fields like `Name`, `DisplayName`, `Description`, and more.

---

## ⚙️ Setup

### 🔧 Requirements

- Python 3.8+
- Transformers (`pip install transformers datasets`)
- PyTorch
- Pandas

---

## 🚦 Pipeline Steps

### 1. ✅ **Initial Setup**

- Define model name (`google/flan-t5-base`)
- Create directories for logs and model checkpoints
- Set up basic file logging

### 2. 📥 **Load Data**

- Load the DataFrame `df_final` from a SQL database via `sql_connection.db`
- Required columns: `Name`, `DisplayName`, `Description`, `Child_Relationship`, `Container`
- Missing or empty entries in these columns are dropped
- Final shape is validated after cleaning

### 3. 🧼 **Preprocess Data**

- Concatenate key fields into a single string:  
  Format:  

Name: ...; DisplayName: ...; Description: ...; Child_Relationship: ...

- Rename columns:
- `input_text` → `description`
- `Container` → `formio_json`
- Convert cleaned DataFrame into a HuggingFace `Dataset`

### 4. 📦 **Load Tokenizer & Model**

- Load tokenizer and sequence-to-sequence model (`AutoModelForSeq2SeqLM`) from HuggingFace

### 5. 🧪 **Tokenization Function**

- Tokenize both:
- Input (`description`)
- Target (`formio_json`)
- Apply max sequence length of 512
- Tokenized dataset is ready for training

### 6. ⚙️ **Training Arguments**

- Define training hyperparameters:
- Learning rate: `5e-5`
- Epochs: `5`
- Batch size: `4`
- Weight decay: `0.01`
- Logging every `10` steps
- No evaluation or model saving during training

---

## 📌 Key Files

| File              | Purpose                                   |
|-------------------|--------------------------------------------|
| `main.py`         | Core training script with all logic       |
| `db.py`           | Connects to DB and loads `df_final`        |
| `training_log.txt`| Logs info, errors, and processing status  |

---

## 🧪 Example Input → Output

**Input:**  
```text
Name: User; DisplayName: User Profile; Description: User's personal information; Child_Relationship: has_form;



## 👨‍💻 Author

Raj | Python + Data Engineering | `from_io_json_convert_model`

---

## 📜 License

MIT License

