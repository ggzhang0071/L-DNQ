import numpy as np
import matplotlib.pyplot as plt

DataSave=np.load('CIFAR10TestAccConvergenceChanges.csv')

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
coefficients=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
plt.style.use('ggplot')
parts=5
for i in range(parts):
    plt.plot(Acc[i], lw=1.5)
for i in range(parts,10):
    plt.plot(Acc[i],'--', lw=1.5)
    
plt.xlabel('Iterations')
plt.ylabel('Accurracy')

plt.legend(tuple(coefficients))
plt.savefig('ContractionCoefficientsCompare_CIFAR10.png',dpi=600)