import pandas as pd
import sqlite3


data_frames = pd.read_csv("zepto_v2.csv",encoding="ISO-8859-1")
# print(data_frames.head())

connection = sqlite3.connect("zepto_inventory.db")

data_frames.to_sql("inventory", connection, if_exists="replace", index=False)

# 4. Run a test query to check if data is there
cursor = connection.cursor()
cursor.execute("SELECT * FROM inventory LIMIT 5;")
rows = cursor.fetchall()

print("First 5 rows:")
for row in rows:
    print(row)

connection.close()


# "Which categories have the most products?"

# "What’s the average discount per category?"

# "How many items are out of stock right now?"
# What are the total orders in June?

