Learn more or give us feedback
import numpy as np
import matplotlib.pyplot as plt

DataSave=np.load('Logs/OK/AccCoefficientsChanges.csv')

def SaveDataCsv(PathFileName):
    with open(PathFileName,'r') as readFile:
            reader =csv.reader(readFile)
            lines = list(reader)
            print(lines)
            
            
coefficients=DataSave[1]
Error=DataSave[2]
# Error
# coefficients
# Error.size
# coefficients

NewError=[]
plt.style.use('ggplot')
parts=2
for i in range(parts):
    plt.plot(Error[i][0][1:160], lw=1.5)
for i in range(parts,len(coefficients)):
    plt.plot(Error[i][0][1:160],'--', lw=1.5)
    
plt.xlabel('Iterations')
plt.ylabel('Error')

plt.legend(tuple(coefficients))
plt.savefig('ContractionCoefficientsCompare_CIFAR10.png',dpi=600)