# Databricks notebook source
# DBTITLE 1,Delete existing files
# delete the old database and tables if needed
_ = spark.sql('DROP DATABASE IF EXISTS bronze CASCADE')

# drop any old delta lake files that might have been created
dbutils.fs.rm('/mnt/bronze', recurse=True)
dbutils.fs.rm('/mnt/gold', recurse=True)
dbutils.fs.rm('/mnt/silver', recurse=True)
dbutils.fs.rm('/mnt/checkpoint', recurse=True)

dbutils.fs.rm('/mnt/silver/member_feature', recurse=True)
dbutils.fs.rm('/mnt/silver/churndata', recurse=True)
dbutils.fs.rm('/mnt/silver/trainingdata', recurse=True)
dbutils.fs.rm('/mnt/gold/scoreddata', recurse=True)
# create database to house SQL tables
_ = spark.sql('CREATE DATABASE bronze')

# COMMAND ----------

# DBTITLE 1,Get Transaction Data
# transaction dataset schema
transaction_schema = StructType([
  StructField('msno', StringType()),
  StructField('payment_method_id', IntegerType()),
  StructField('payment_plan_days', IntegerType()),
  StructField('plan_list_price', IntegerType()),
  StructField('actual_amount_paid', IntegerType()),
  StructField('is_auto_renew', IntegerType()),
  StructField('transaction_date', DateType()),
  StructField('membership_expire_date', DateType()),
  StructField('is_cancel', IntegerType())  
  ])

# read data from parquet
transactions = (
  spark
    .read
    .csv(
      '/mnt/landing/transactions_v2.csv',
      schema=transaction_schema,
      header=True,
      dateFormat='yyyyMMdd'
      )
    )

# persist in delta lake format
( transactions
    .write
    .format('delta')
    .partitionBy('transaction_date')
    .mode('overwrite')
    .save('/mnt/bronze/transactions')
  )

# create table object to make delta lake queriable
spark.sql('''
  CREATE TABLE bronze.transactions
  USING DELTA 
  LOCATION '/mnt/bronze/transactions'
  ''')

# COMMAND ----------

# DBTITLE 1,Get Members Data
# members dataset schema
member_schema = StructType([
  StructField('msno', StringType()),
  StructField('city', IntegerType()),
  StructField('bd', IntegerType()),
  StructField('gender', StringType()),
  StructField('registered_via', IntegerType()),
  StructField('registration_init_time', StringType())
  ])

# read data from csv
members = (
  spark
    .read
    .csv(
      'dbfs:/mnt/landing/members_v3.csv',
      schema=member_schema,
      header=True,
      dateFormat='yyyyMMdd'
      )
    )

# persist in delta lake format
(
  members
    .write
    .format('delta')
    .mode('overwrite')
    .save('/mnt/bronze/members')
  )

# create table object to make delta lake queriable
spark.sql('''
  CREATE TABLE bronze.members 
  USING DELTA 
  LOCATION '/mnt/bronze/members'
  ''')

# COMMAND ----------

# DBTITLE 1,Get User Log Data
log_schema = StructType([
  StructField('msno', StringType()),
  StructField('date', StringType()),
  StructField('num_25', IntegerType()),
  StructField('num_50', StringType()),
  StructField('num_75', IntegerType()),
  StructField('num_985', IntegerType()),
  StructField('num_100', IntegerType()),
  StructField('num_unq', IntegerType()),
  StructField('total_secs', DoubleType())
  ])

# read data from csv
user_log = (
  spark
    .read
    .csv(
      'dbfs:/mnt/landing/user_logs_v2.csv',
      schema=log_schema,
      header=True,
      dateFormat='yyyyMMdd'
      )
    )
# persist in delta lake format
(
  user_log
    .write
    .format('delta')
    .mode('overwrite')
    .save('/mnt/bronze/user_log')
  )

# create table object to make delta lake queriable
spark.sql('''
  CREATE TABLE bronze.user_log 
  USING DELTA 
  LOCATION '/mnt/bronze/user_log'
  ''')

# COMMAND ----------

# DBTITLE 1,Get churn dataset
# train dataset schema
train_schema = StructType([
  StructField('msno', StringType()),
  StructField('is_churn', IntegerType())
  ])

# read data from csv
train = (
  spark
    .read
    .csv(
      'dbfs:/mnt/landing/train_v2.csv',
      schema=train_schema,
      header=True
      )
    )

# persist in delta lake format
(
  train
    .write
    .format('delta')
    .mode('overwrite')
    .save('/mnt/bronze/train')
  )

# create table object to make delta lake queriable
spark.sql('''
  CREATE TABLE bronze.churn 
  USING DELTA 
  LOCATION '/mnt/bronze/train'
  ''')

# COMMAND ----------

Workspace = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().get("browserHostName").getOrElse(None)
