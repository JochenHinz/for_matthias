import numpy as np
import script

def main():
    ''' 
        create .p2dfmt-files using the n_i, m_i
        
        Topology:
        
        m2 |1 - 3- 4|
        m1 |2 - x- 5|
            n1-n2-n3
            
        creates files of the form one_i_j.p2dfmt, where i = length(n1), j = length(m2) etc in the 'output' folder.
    
    '''
    
    _n1, _n2, _n3 = 401, 51, 401
    
    _m1, _m2 = 51, 101
    
    ##create uniformly spaced vectors with n-1 elements
    n1, n2, n3, m1, m2 = [np.linspace(0,1, n) for n in [_n1, _n2, _n3, _m1, _m2]]
    
    m2 = np.arctan(6*(m2 - 0.5))  ## create arctan distribution
    m2 = 0.5*m2/m2[-1]  ## m2[0] = -0.5, m2[-1] = 0.5
    m2 += 0.5  ## m2[0] = 0, m2[-1] = 1
    
    script.main(n1, n2, n3, m1, m2) 
    
    
if __name__ == '__main__':
    main()
    