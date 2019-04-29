input = [1,1,1,2]
input = [1,2,5,5,8,2,2,1]

results = []
for j in range(0, len(input)):
    rest_set = set(input)
    remove_set = set([])
    out_list = list(rest_set)
    list.sort(out_list)
    result = []
    for i in range(j, len(input)):
        if len(rest_set) == 0:
            break
        else:
            result.append(input[i])
            if input[i] in rest_set:
                rest_set.remove(input[i])
            remove_set.add(input[i])
    results.append(result)
print(results)