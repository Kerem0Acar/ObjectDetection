import sqlite3

conn = sqlite3.connect("ObjectDetection.db")
cursor = conn.cursor()

#Get username and password from ui
def register():
    username = input("Enter username: ")
    password = input("Enter password: ")

    try:
        cursor.execute('INSERT INTO Users (username, password) VALUES (?, ?)', (username, password))
        conn.commit()
        print("User added successfully!")
    except sqlite3.IntegrityError:
        print("Error: Username already exists.")

#Get username and password from ui
def login():
    username = input("Enter username: ")
    password = input("Enter password: ")

    try:
        cursor.execute('INSERT INTO Users (username, password) VALUES (?, ?)', (username, password))
        conn.commit()
        print(username + " login successfully!")
    except sqlite3.IntegrityError:
        print("Error: Username or password is wrong!.")

#Creating table --- tableName is the name that get from which user use the program
def creating_table(tableName,objectName,date):

    cursor.execute("SELECT "+tableName+" FROM "+" sqlite_master WHERE type='table' AND name=?:",(tableName,))
    exists = cursor.fetchone()
    if exists:
        print("Table already exists")
    else:
        cursor.execute(f'''
                    CREATE TABLE {tableName} (
                        {objectName} TEXT NOT NULL,
                        {date} TEXT NOT NULL
                    )
                ''')
        conn.commit()
        print(f"Table '{tableName}' has been created.")

#Inserting table
def inserting_table(tableName,objectName,date):
    cursor.execute("INSERT INTO "+tableName+" (objectName, date) VALUES (?, ?)",(objectName,date))
    conn.commit()

#Gathering object
def gathering_objects(tableName):
    cursor.execute("SELECT * FROM Users")
    rows = cursor.fetchall()

    for row in rows:
        print(row)
