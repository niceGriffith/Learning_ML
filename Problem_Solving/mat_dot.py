def mat_dot(a: list[list[int|float]], b: list[int|float]):
    if len(a[0]) != len(b):
        return -1
    else:
        result =[]
        for i in range(len(a)):
            sum = 0
            for j in range(len(b)): 
                sum +=a[i][j]*b[j]
                result.apend(sum)
    return result 
	