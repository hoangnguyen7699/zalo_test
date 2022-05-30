def cast(data, type):
    if type == int:
        return [int(i) if i is not None else None for i in data]
    elif type==float:
        return [float(i) if i is not None else None for i in data]
    elif type==bool:
        return [bool(i) if i is not None else None for i in data]
    else:
        return [str(i) if i is not None else None for i in data]
