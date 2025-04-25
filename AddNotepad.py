import pandas as pd
import os
from datetime import datetime

notepad = pd.DataFrame(columns=['Date', 'Note'])
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


def addNote(noteText):
    date_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    new_note = {'Date': date_str, 'Note': noteText}
    global notepad
    notepad = notepad._append(new_note,ignore_index=True)
    notepad.to_csv("notepad.csv",index=False)
