from z3 import *


def solver(tree_paths, features, pro_to_check):

    conjunction = locals()
    neg_pro_pre = []

    # define variables of features, minimal label 'lmin' and maximal label 'lmax'
    n = 0
    while n < len(features):
        f_name = features[n]
        locals()[f_name] = Real(f_name)
        n += 1
    lmin = Int('lmin')
    lmax = Int('lmax')

    i = 0
    while i < len(tree_paths):
        conjunction[i] = []
        j = 0
        while j < len(tree_paths[i]):
            p = eval(tree_paths[i][j])
            conjunction[i].append(p)
            j += 1
        print("path" + str(i) + ":")
        print(conjunction[i])
        i += 1

    m = 0

    while m < len(pro_to_check) -1:
        pre = eval(pro_to_check[m])  # pre of pre -> post
        neg_pro_pre.append(Not(pre))
        m += 1

    pro_post = (eval(pro_to_check[len(pro_to_check) - 1]))

    o = 0
    s = Solver()
    s.add(Not(Or(Or(neg_pro_pre), pro_post)))
    result = bool
    while o < len(tree_paths):
        s.push()
        s.add(conjunction[o])
        print(s)
        print(s.check())
        if str(s.check()) == "sat":
            result = 0
            m = s.model()
            r = []
            for x in m:
                print("%s = %s" % (x.name(), m[x]))

            break
        else:
            result = 1
        o += 1
        s.pop()

    if result == 1:
        print("unsat: the property is valid.")
    else:
        print("sat: the property is violated.")