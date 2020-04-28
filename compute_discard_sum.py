'''
This program computes the sum of the discarded material from a CTH calculation
'''

import numpy as np
import glob


def get_initial_mat_mass():
    '''
    Compute initial material masses from octh file
    '''
    mat_mass = []
    file1 = open('octh','r')

    count = 0

    # loop through octh to find conservation accounting
    while count < 10000:
        line = file1.readline()
        if "CONSERVATION ACCOUNTING" in line:
            break
        count += 1

    # error if didn't find CONSERVATION ACCOUNTING
    if count > 9999:
        print('Error: CONSERVATION ACCOUNTING not found')

    line = file1.readline()
    while True:
        line = file1.readline()
        a = line.split('  ')
        if line != '\n':
            mat_mass.append(float(a[3]))
        else:
            break

    file1.close()
    return mat_mass


# get inital mass of each material into a list
mat_mass_0 = get_initial_mat_mass()
nmat = len(mat_mass_0)

mat_dis = np.zeros(nmat)

flist = glob.glob('octh*')
for fname in flist:
    if len(fname) > 4:
        file1 = open(fname,'r')
        lines = file1.readlines()
        for line in lines:
            a = line.split('  ')
            if a[0] != ' DISCARD MAT':
                i = int(a[4]) - 1
                mat_dis[i] += float(a[-2])
        file1.close()
           


print('\nDiscarded Mass:')
print('mat id\ttotal (kg)\tfraction')

for i in range(nmat):
    print(('%i\t%.4e\t%.4e') % (i+1,mat_dis[i]/1.e3,mat_dis[i]/mat_mass_0[i]))












        

