import pickle
import os

path1 = './refine/refine_results/param.pkl'
path2 = './kitti_results1/param.pkl'

a = pickle.load(open(path1, 'r'))
b = pickle.load(open(path2, 'r'))
c = {}
for var in a:
    c[var] = a[var]
for var in b:
    c[var] = b[var]

print len(a)
print len(b)
print len(c)
pickle.dump(c, open('param.pkl', 'w'))

