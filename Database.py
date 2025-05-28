from datetime import datetime
import sqlite3

conn = sqlite3.connect("ObjectDetection.db")
cursor = conn.cursor()

#Creating table
def creating_table():

        cursor.execute(f'''
                    CREATE TABLE IF NOT EXISTS Objects (
                        ObjectName TEXT NOT NULL,
                        Accuracy TEXT NOT NULL,
                        Date TEXT NOT NULL
                    )
                ''')
        conn.commit()
        print(f"Table Objects has been created.")

#Inserting table
def inserting_table(objectName,accuracy):
    now = datetime.now()
    date_string = now.strftime("%d-%m-%Y %H:%M:%S")

    cursor.execute("INSERT INTO Objects (ObjectName,Accuracy, Date) VALUES (?, ?, ?)",(objectName,accuracy,date_string))
    conn.commit()

#Gathering object
def gathering_objects():
    cursor.execute("SELECT * FROM Objects")
    rows = cursor.fetchall()

    for row in rows:
        print(row)

creating_table()
inserting_table("Object Name","Accuracy")
gathering_objects()

