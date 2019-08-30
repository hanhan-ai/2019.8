import numpy as np
import random

TEST = [0, 0, 1]

test = np.zeros([3])
# test = [0,0,1]
for i in range(1, 20):

    test[(random.randint(1, 100)) % 3] = 1
    print('test', test)
    print('TEST', TEST)
    if all(TEST == test):
        print("they are equal!")
    else:
        print("they are not equal")
    '''    
    if all(TEST == TEST):                         # that's the reason why 'bool' object is not iterable
        print("TEST == TEST")                     # DAMN
    '''
    test = np.zeros([3])

    testRandom = random.randrange(3)
    print(testRandom)

a = [1,2,3,4,5,6,7,8,9]
for i in range(0, len(a)):
    print(a[i])


