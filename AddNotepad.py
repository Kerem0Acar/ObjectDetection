import os
from datetime import datetime
import pandas as pd
import time

notepad = pd.DataFrame(columns=['ObjectName', 'Accuracy', 'Date'])
file_path = "notepad.csv"

def createCSV():
    if checkNotepad():
        print("Creating new CSV format")
    else:
        notepad.to_csv(file_path, index= False)

def checkNotepad():
    if os.path.exists(file_path):
        print(f"{file_path} is already exist" )

def getObject():
    pass


def addNote(noteText,accuracy,date):
    new_note = {'ObjectName': noteText, 'Accuracy': accuracy,'Date': date}
    global notepad
    notepad = notepad._append(new_note,ignore_index=True)
    notepad.to_csv(file_path,index=False)

def check():
    print("Checking")
