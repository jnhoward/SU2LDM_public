import numpy as np

def prettyPrintColumns(dfDict):
    
    keyList = list(dfDict.keys())
    nCols  = len(keyList)
    
    dummyString = ""
    for key in keyList:
        dummyCol   = dfDict[key].reshape(-1,1)
        
        if key==keyList[0]:
            dummyStack = dummyCol
        else:
            dummyStack = np.column_stack((dummyStack, dummyCol))
    
        dummyString += "{: >20}"
        if key != keyList[-1]: dummyString += " "
    
    #print(dummyString)
    dummyKeyArr = np.array([keyList])
    for row in dummyKeyArr:
        print(dummyString.format(*row))
    
    rowMax = 5
    iRow = 0
    for row in dummyStack:
        
        if iRow < rowMax:
            print(dummyString.format(*row))
            iRow += 1