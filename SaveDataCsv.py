import csv
import os
newdata =[0,0,0,0,0,0]
Path='./Results/'
FileName='AccuracyRate.csv'
PathFileName=os.path.join(Path,FileName)
def SaveDataCsv(PathFileName,newdata):
    with open(PathFileName,'r') as readFile:
            reader =csv.reader(readFile)
            lines = list(reader)
            lines.append(newdata) 
            print(lines)

    with open(PathFileName,'w') as writeFile:
        writer =csv.writer(writeFile)
        writer.writerows(lines)

    readFile.close()
    writeFile.close()