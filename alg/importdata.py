def importdata(data,ind):
    l=len(data)
    tempdataline=[]
    result_data=[]

    for k in range(ind,l):
        tempdataline=float(data[k][0:-1])
        result_data.append(tempdataline)

    return result_data