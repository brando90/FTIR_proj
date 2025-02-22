## kernel

from sklearn.metrics.pairwise import euclidean_distances

def get_gauss_coeffs(X,centers,std):
    '''
    X = input
    center = centers of RBF
    std = standard dev of Gaussians
    '''
    #indices=np.random.choice(a=N,size=K,replace=replace) # choose numbers from 0 to D^(1)
    #subsampled_data_points=X[indices,:] # M_sub x D
    beta = np.power(1.0/std,2)
    Kern = np.exp(-beta*euclidean_distances(X=X,Y=centers,squared=True))
    (C,_,_,_) = np.linalg.lstsq(Kern,Y)
    return C

def get_subsampled_data(X,replace=False):
    N = X.shape[0]
    indices=np.random.choice(a=N,size=K,replace=replace) # choose numbers from 0 to D^(1)
    subsampled_data_points=X[indices,:] # M_sub x D
    return subsampled_data_points

def get_kernel_matrix(x,W,S):
    beta = get_beta_np(S)
    #beta = 0.5*tf.pow(tf.div( tf.constant(1.0,dtype=tf.float64),S), 2)
    Z = -beta*euclidean_distances(X=x,Y=W,squared=True)
    K = np.exp(Z)
    return K

def get_z_np(x,W):
    WW = np.sum(np.multiply(W,W), axis=0, dtype=None, keepdims=True)
    XX = np.sum(np.multiply(x,x), axis=1, dtype=None, keepdims=True)
    Delta_tilde = 2.0*np.dot(x,W) - (WW + XX)
    return Delta_tilde

def get_Z_tf(x,W,l='layer'):
    '''
    x = input, M data points of size D^(l-1)
    W = params/centers, D^(l-1) x D^(l) means that each center is of dimension D^(l-1) since
        they are being computed with respect to an element from the previous layer.
        Thus, they are D^(l) of them.

    return:
    Delta_tilde = (M x D^(l))
    '''
    W = tf.Variable(W, name='W'+l, trainable=True, dtype=tf.float64)
    WW =  tf.reduce_sum(W*W, reduction_indices=0, keep_dims=True) #(1 x D^(l))= sum( (D^(l-1) x D^(l)), 0 )
    XX =  tf.reduce_sum(x*x, reduction_indices=1, keep_dims=True) #(M x 1) = sum( (M x D^(l-1)), 1 )
    # -|| x - w ||^2 = -(-2<x,w> + ||x||^2 + ||w||^2) = 2<x,w> - (||x||^2 + ||w||^2)
    xW = tf.matmul(x,W) #(M x D^(l)) = (M x D^(l-1)) * (D^(l-1) x D^(l))
    Delta_tilde = 2.0*xW - tf.add(WW, XX) #(M x D^(l)) = (M x D^(l)) + ( (M x 1) + (1 x D^(l)) )
    return Delta_tilde

def get_beta_np(S):
    beta = np.power(1.0/S,2)
    return beta

def get_beta_tf(S):
    one = tf.constant(1.0,dtype=tf.float64)
    beta = tf.pow( tf.div(one,S), 2)
    return beta
