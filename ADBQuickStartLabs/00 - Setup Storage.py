# Databricks notebook source
dbutils.widgets.text("ACCOUNT-KEY", "", "ACCOUNT-KEY")
dbutils.widgets.text("BLOB-CONTAINER", "", "BLOB-CONTAINER")
dbutils.widgets.text("BLOB-ACCOUNT-NAME", "", "BLOB-ACCOUNT-NAME")

# COMMAND ----------

BLOB_CONTAINER = dbutils.widgets.get("BLOB-CONTAINER")
BLOB_ACCOUNT = dbutils.widgets.get("BLOB-ACCOUNT-NAME")
ACCOUNT_KEY = dbutils.widgets.get("ACCOUNT-KEY")

# COMMAND ----------

# MAGIC %md
# MAGIC # Read sensitive values from Azure Key Vault / Secret Scope

# COMMAND ----------

BLOB_CONTAINER  = dbutils.secrets.get(scope = "scope-storage-adb", key = "BLOB-CONTAINER")
BLOB_ACCOUNT = dbutils.secrets.get(scope = "scope-storage-adb", key = "BLOB-ACCOUNT-NAME")
ACCOUNT_KEY = dbutils.secrets.get(scope = "scope-storage-adb", key = "ACCOUNT-KEY")

# COMMAND ----------

print(BLOB_CONTAINER)

# COMMAND ----------

# DBTITLE 1,Run this step only if you are re-running the notebook
try:
    dbutils.fs.unmount("/mnt")
except:
  print("The storage isn't mounted so there is nothing to unmount.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Mounting Azure Storage using an Access Key or Service Principal
# MAGIC We will mount an Azure blob storage container to the workspace using a shared Access Key. More instructions can be found [here](https://docs.microsoft.com/en-us/azure/databricks/data/data-sources/azure/azure-storage#--mount-azure-blob-storage-containers-to-dbfs). 
# MAGIC 
# MAGIC #####Note: For this Demo we are using access Key and mounting the blob on DBFS. Ideally one should authenticate using Service Principal and use full abfss path to access data

# COMMAND ----------

MOUNT_PATH = "/mnt"

dbutils.fs.mount(
  source = f"wasbs://{BLOB_CONTAINER}@{BLOB_ACCOUNT}.blob.core.windows.net",
  mount_point = MOUNT_PATH,
  extra_configs = {
    f"fs.azure.account.key.{BLOB_ACCOUNT}.blob.core.windows.net":ACCOUNT_KEY
  }
)
