
from DataFrame import DataFrame
import re


def check_header(arr):
    """
    Return a new header if there are multiple columns with the same name. Otherwise, return 
    the original on

    Parameters
        ----------
        arr: list
            list of the names of header
    Examples
        --------
        >>> check_header(['Hoang', 'zalo', 'Hoang])
        >>> ['Hoang1', 'zalo', 'Hoang2']
        
    """
    d = {}
    d2 = {}
    for i in arr:
        d[i] = d.get(i, 0) + 1
        if d[i] >= 2:
            d2[i] = 1
    for i in range(len(arr)):
        if d[arr[i]]>= 2:
            temp = arr[i]
            arr[i] = temp +  str(d2[temp])
            d2[temp] += 1
    return arr


def read_csv(path, header=True, separation =',', num_skip = 0, dtype=None):
    """
    Return a dict where each key correspond to a column in the original csv file. The value of
    a key is a list containing all the entry in the correspoding column of the key.

    Parameters
        ----------
        path: string
            integer indicates the number of printed rows 
        header: Boolean, deafult True
            bool indicate if the header is already in the csv file. If the csv file does not have 
            header, then the default header will be generated, which are Column1, Column2,....If there
            are multiple columns with the same name, for example Date, then I will transfer them to
            Date1, Date2,...
        num_skip: integer, default 0
            The number of line (after counting header) that user want to skip.
        dtype: list, default None
            The list containing the data type of each column in csv file     
    """
    pattern = separation + '''(?=(?:[^'"]|'[^']*'|"[^"]*")*$)'''
    with open(path, 'r+') as file:
        res = []
        i = None
        
        for line in file:
            
            line = line.rstrip('\n')
            if line.strip() == '':
                continue
            #items = re.split(''',(?=(?:[^'"]|'[^']*'|"[^"]*")*$)''', line)
            items = re.split(pattern, line)
            for k in range(len(items)):
                items[k] = items[k].strip()
                items[k] = items[k].replace('"','')
                if items[k] == '':
                    items[k] = None
            res.append(items)
            
    num_cols = len(res[0])
    header_index = None
    data = {}
    start_row = num_skip
    if header:
        header_index = check_header(res[0])
        start_row += 1
    else:
        header_index = ['Column' + str(i) for i in range(1, num_cols+1)]
    for i in range(num_cols):
        #print(i)
        data[header_index[i]]=[res[j][i] for j in range(start_row, len(res))]

    
    
    return DataFrame(data=data, dtype=dtype)


