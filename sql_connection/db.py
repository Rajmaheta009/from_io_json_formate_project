import pandas as pd
from sqlalchemy import create_engine

# ✅ Database server
SERVER_NAME = "RAjPC\\SQLEXPRESS"

# ✅ Set Pandas to show all columns
pd.set_option('display.max_columns', None)

# ✅ Create DB engine
def get_engine(database_name):
    return create_engine(
        f"mssql+pyodbc://@{SERVER_NAME}/{database_name}?driver=ODBC+Driver+17+for+SQL+Server&trusted_connection=yes"
    )

# ✅ SQL Join Query (inner + left join)
JOIN_QUERY = """
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
INNER JOIN dbo.TABMD_DataSourceQueries dsq ON dsq.Id = scr.DataSourceQueryID
INNER JOIN dbo.TABMD_AppObject ao ON ao.Id = dsq.AppObject
LEFT JOIN dbo.TABMD_AppFields fld ON fld.AppObjectId = ao.Id
"""

# ✅ Fetch data from a single database
def fetch_data(database_name):
    print(f"📡 Fetching from: {database_name}")
    engine = get_engine(database_name)
    df = pd.read_sql(JOIN_QUERY, engine)
    df["SourceDB"] = database_name
    return df

# ✅ List of databases
databases = ["QA_TAB", "TAB_MT_3010", "TAB_Rewrite", "TAB_RM"]

# ✅ Fetch and combine data
df_all = [fetch_data(db) for db in databases]
df_final = pd.concat(df_all, ignore_index=True)

# ✅ Print % of missing values
print("\n📉 Null percentage per column:")
null_percentage = df_final.isnull().mean().round(4).sort_values(ascending=False) * 100
print(null_percentage)

# ✅ Drop columns that are more than 95% null
df_final = df_final.dropna(thresh=int(len(df_final) * 0.05), axis=1)

# ✅ Final Output
print("\n✅ Final Shape:", df_final.shape)
print(df_final)
print("📊 Total rows:", len(df_final))
print("📈 Total columns:", len(df_final.columns))

# ✅ Optional save
# df_final.to_csv("cleaned_joined_output.csv", index=False)
