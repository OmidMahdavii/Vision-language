addresses = ['./groupe1AML.txt', './groupe1DAAI.txt', './groupe2AML.txt', './groupe2DAAI.txt', './groupe3AML.txt',
            './groupe3DAAI.txt', './groupe5AML.txt', './groupe6AML.txt']

def read_lines(index):
    temp = []
    result = dict()
    for a in addresses:
        with open(a) as f:
            lines = f.read()
            lines = lines[1:-1:1]
            test = lines.split('},')
            for t in range(len(test)-1):
                test[t] = test[t] + '}'
                test[t] = eval(test[t])
                test[t]['descriptions'] = test[t]['descriptions'][index]
                temp.append(test[t])
    
    for i in temp:
        descriptions = i['descriptions'].split(', ')
        for d in descriptions:
            if d in result:
                result[d].append(i['image_name'])
            else:
                result[d] = [i['image_name']]
    
    return result


items = read_lines(0) # input parameter determines which category of description you want

# # uncomment to show all the available descriptions for this category with the number of images which have this description

for i in items:
    print(i, len(items[i]))




# # uncomment to show all the images having a specific description
# # for example, this code brings all the images described as 'grainy'
        
# for i in items['grainy']:
#     print(i)
