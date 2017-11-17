import pickle
import numpy as np

def main(n1,n2,n3,m1,m2):
    
    go_1, go_2, go_3, go_4, go_5 = [pickle.load(open('saves/' + name + '.pkl', "rb")) for name in ['one', 'two', 'three', 'four', 'five']]
    
    c0_ind = np.where(go_4.knotmultiplicities[0] == go_4.degree[0])[0]
    
    if len(c0_ind) > 0:  ## c^0-indices found, ensure that it is evaluated over
        c0_xi = go_4.knots[0][c0_ind]
        n3 = np.array(sorted(list(set(list(n3) + list(c0_xi)))))
        print('added ', c0_xi, 'to n3')
    
    go_1.to_p2dfmt(n1, m2, 'output/one_' + str(len(n1)) + '_' + str(len(m2)))
    go_2.to_p2dfmt(n1, m1, 'output/two_' + str(len(n1)) + '_' + str(len(m1)))
    go_3.to_p2dfmt(n2, m2, 'output/three_' + str(len(n2)) + '_' + str(len(m2)))
    go_4.to_p2dfmt(n3, m2, 'output/four_' + str(len(n3)) + '_' + str(len(m2)))
    go_5.to_p2dfmt(n3, m1, 'output/five_' + str(len(n3)) + '_' + str(len(m1)))
    
    
if __name__ == '__main__':
    main()