def read_lines():
    temp = []
    # result = dict()
    with open('./omids_labels.txt') as f:
        lines = f.read()
        lines = lines[1:-1:1]
        test = lines.split('},')
        for t in range(len(test)-1):
            test[t] = test[t] + '}'

        for t in test: 
            t = eval(t)
            temp.append(t)
    
    # for i in temp:
    #     descriptions = i['descriptions'].split(', ')
    #     for d in descriptions:
    #         if d in result:
    #             result[d].append(i['image_name'])
    #         else:
    #             result[d] = [i['image_name']]
    
    return temp


items = read_lines()

for i in items:
    print(i['descriptions'][0])
# print(items)


