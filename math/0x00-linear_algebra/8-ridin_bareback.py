def mat_mul(mat1, mat2):

    if len(mat1[0]) != len(mat2):
        return None
    multiplication = []
    for i in range(len(mat1)):
        row = []
        for k in range(len(mat2[0])):
            sum = 0
            for j in range(len(mat1[0])):
                sum = sum + mat1[i][j] * mat2[j][k]
            row.append(sum)
        multiplication.append(row)
    return multiplication
