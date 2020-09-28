# generates stub functions for fun and profit

def get_def_lines(file):
    func_list = []
    print("get functions for -->", file)
    with open(file) as fp:
        line = fp.readline()
        cnt = 1
        while line:
            if 'def' in line:
                if not 'self' in line:
                    print("Line {}: {}".format(cnt, line.strip()))
                    func_list.append(line.strip())
            line = fp.readline()
            cnt += 1
    fp.close()
    return func_list

file = './tmp/eto_functions.py'
myfunc_list = get_def_lines(file)
print(myfunc_list)

for func in myfunc_list:
    print(func)
    print(f'    print(\'{func}\')')
    print('\n\n')
