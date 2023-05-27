import os
from pathlib import Path
import shutil

def clean():
    home = Path(__file__)
    outputs = home.parent / "outputs"
    cnt = 0
    for folder in outputs.glob("*"):
        cnt += 1
        shutil.rmtree(folder)
        
    cnt_file = [path for path in home.glob("cnt_*")]
    counter = 0
    if cnt_file:
        counter = 1
        os.remove(cnt_file[0])
    
    return f"{cnt} folder{'s' if cnt != 1 else ''}, {counter} file{'s' if counter != 1 else ''} removed"
        

if __name__=="__main__":
    status = clean()
    print(status)