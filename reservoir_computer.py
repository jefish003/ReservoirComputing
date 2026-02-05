import numpy as np
import networkx as nx
from scipy.linalg import schur

class reservoir_computer:
    
    def __init__(self,data_in,data_out,n=200,spectral_radius=None,gamma=None,Win_dist = 'uniform',Win=None,A=None,reservoir_type='erdos_renyi',res_weights=1,res_distribution='normal',res_density=0.02,ws_p=0.1,a=0,non_normal_type='sparse',scale_A=True,input_noise=True,in_noise=None,alpha=None,activation=np.tanh,beta=None,maintain_res_eigs=False):
        #If below is true, then the eigenvalues will be maintained while the eigenvalues stay the same
        #if false then the eigenvalues of the reservoir are allowed to change. 
        self.maintain_res_eigs = maintain_res_eigs        
        self.data_in = data_in
        self.data_out = data_out
        #spectral radius of the reservoir
        t,dim = data_in.shape
        #number of time units
        self.t = t
        #number of dimensions (in the input)
        self.dim = dim
        #input noise
        if input_noise:
            if in_noise is None:
                in_noise = 1.1549719e-7
            data_in = data_in + np.random.normal(0,in_noise,data_in.shape)
        
        if alpha is None:
            self.alpha = 0.4054178
        else:
            self.alpha = alpha
        #spectral radius
        if spectral_radius is None:
            self.spectral_radius = 4.7846556
        else:
            self.spectral_radius = spectral_radius
        #number of nodes in the reservoir
        self.n = n
        #for determining the uniform distribution to draw from (+-gamma)
        if gamma is None:
            self.gamma = 0.2749319
        else:
            self.gamma = gamma
        
        #the Win 
        if Win is None:
            if Win_dist == 'uniform':
                self.Win = np.random.uniform(-self.gamma,self.gamma,(n,dim))
            
            elif Win_dist == 'normal' or Win_dist == 'gaussian':
                self.Win = np.random.normal(0,self.gamma,(n,dim))
        else:
            self.Win = Win
        
        #The type of reservoir, like Erdos-Renyi for example
        self.reservoir_type = reservoir_type
        #reservoir density
        self.res_density = res_density
        #for use only in Watts-Strogatz graphs
        self.ws_p = ws_p
        #non-normality parameter (a = 0 implies normal)
        self.a = a
        #reservoir distribution weights
        self.res_weights = res_weights
        #reservoir distribution type
        self.res_distribution = res_distribution
        #nonormality type (for ER,WS,and BA types of reservoirs)
        self.non_normal_type = non_normal_type
        #generate or take the reservoir
        if A is None:
            self.generate_A()
        
        else:
            self.A = A
        
        #note scale_A should be set to True unless you passed an already scaled reservoir,
        #so do not set to False unless you have passed your own reservoir
        if scale_A:
            maxEig = np.max(np.abs(np.linalg.eigvals(self.A)))
            self.A = self.spectral_radius*self.A/maxEig
        
        self.act = activation
        if beta is None:
            self.beta = 0.0004361
        else:
            self.beta = beta
        
    def return_Win(self):
        return self.Win
    def generate_res_distribution(self):
        if self.res_distribution == 'normal' or self.res_distribution == 'gaussian':
            R = np.random.normal(0,self.res_weights,(self.n,self.n))
        elif self.res_distribution == 'uniform':
            R = np.random.normal(-self.res_weights,self.res_weights,(self.n,self.n))
        else:
            raise ValueError("Reservoir distribution must either be normal/gaussian or uniform")
        
        return R

    def dense_upper_triangular(self):
        """
        Create an n×n sparse upper triangular matrix with controllable Henrici departure.
    
        Parameters
        ----------
        n : int
            Matrix dimension.
        res_density : float
            Fraction of nonzero entries in the UPPER triangle.
    
        Returns
        -------
        U : ndarray
            n×n matrix with upper-triangular sparsity
        """
        # --- Step 1: build sparse upper triangular matrix ---
        total_upper = self.n * (self.n + 1) // 2
        nnz = int(self.res_density * total_upper)
    
        rows, cols = np.triu_indices(self.n)
        idx = np.random.choice(total_upper, size=nnz, replace=False)
        if self.res_distribution == 'normal' or self.res_distribution =='gaussian':
            U = np.random.normal(0,self.res_weights,(self.n, self.n))
        elif self.res_distribution == 'uniform':
            U = np.random.uniform(-self.res_weights,self.res_weights,(self.n, self.n))
        
        else:
            raise ValueError("(Note dense distribution will match res_distribution) res_distribution must either be normal/gaussian or uniform ")
    
        return U


    def sparse_upper_triangular(self, value_fn=None):
        """
        Create an n×n sparse upper triangular matrix with controllable Henrici departure.
    
        Parameters
        ----------
        n : int
            Matrix dimension.
        res_density : float
            Fraction of nonzero entries in the UPPER triangle.
        value_fn : callable or None
            Function generating values for nonzeros. If None, uses uniform(0,1).
    
        Returns
        -------
        U : ndarray
            n×n matrix with upper-triangular sparsity
        """
        # --- Step 1: build sparse upper triangular matrix ---
        total_upper = self.n * (self.n + 1) // 2
        nnz = int(self.res_density * total_upper)
    
        rows, cols = np.triu_indices(self.n)
        idx = np.random.choice(total_upper, size=nnz, replace=False)
    
        U = np.zeros((self.n, self.n))
    
        if value_fn is None:
            if self.res_distribution == 'normal' or self.res_distribution == 'gaussian':
                values = np.random.normal(0,self.res_weights,nnz)
            elif self.res_distribution == 'uniform':
                values = np.random.uniform(-self.res_weights,self.res_weights,nnz)
            else:
                raise ValueError("(Note sparse distribution will match res_distribution) res_distribution must either be normal/gaussian or uniform ")
        else:
            values = value_fn(nnz)
    
        U[rows[idx], cols[idx]] = values    
    
        return U


    def jacobi_matrix(self,
                      diag_mode="random",
                      diag_value=0.0,
                      offdiag_base=1.0,
                      rng=None):
        """
        Generate an n x n Jacobi matrix J with tunable non-normality.
    
        Steps:
          1. Build symmetric tridiagonal S with positive off-diagonals.
          2. Build positive diagonal D with controlled conditioning.
          3. Form J = D^{-1} S D, which is Jacobi.
    
        Parameters
        ----------
        n : int
            Matrix size.
        diag_mode : "random" or "constant"
            How to choose diagonal entries of S.
        diag_value : float
            Used if diag_mode == "constant".
        offdiag_base : float
            Base value for symmetric off-diagonals of S.
        non_normality : float
            Controls spread of diagonal scaling D.
            0.0 -> symmetric (normal)
            larger -> more non-normality
        rng : numpy random generator or None
            For reproducibility.
    
        Returns
        -------
        J : ndarray
            Jacobi matrix.
        """
    
        if rng is None:
            rng = np.random.default_rng()
    
        # --- 1. Build symmetric tridiagonal S ---
        if diag_mode == "random":
            a = rng.normal(loc=diag_value, scale=1.0, size=self.n)
        elif diag_mode == "constant":
            a = np.full(self.n, diag_value)
        else:
            raise ValueError("diag_mode must be 'random' or 'constant'.")
    
        # Positive off-diagonals
        s = offdiag_base * np.ones(self.n - 1)
    
        S = np.zeros((self.n, self.n))
        np.fill_diagonal(S, a)
        np.fill_diagonal(S[:-1, 1:], s)
        np.fill_diagonal(S[1:, :-1], s)
    
        # --- 2. Build diagonal scaling D ---
        if self.a == 0.0:
            d = np.ones(self.n)
        else:
            # log-normal spread: d_i = exp(non_normality * xi_i)
            xi = rng.normal(0.0,self.a, size=self.n)
            d = np.exp(xi)
    
        D = np.diag(d)
    
        # --- 3. Form J = D^{-1} S D ---
        # Efficient: J_ij = S_ij * d_j / d_i
        J = (S / d[:, None]) * d[None, :]
    
        return J

    def circulant_reservoir(self,first_row=None):
        """
        Create a real circulant reservoir matrix using NumPy only.
    
        Parameters
        ----------
        n : int
            Dimension of the reservoir.
        first_row : ndarray or None
            First row of the circulant matrix. If None, random values are used.
        scale : float
            Optional scaling factor applied to the matrix (e.g., to set spectral radius).
    
        Returns
        -------
        C : ndarray
            n×n real circulant matrix.
        """
        if first_row is None:
            # Random real first row
            first_row = np.random.randn(self.n)
    
        # Build circulant matrix
        C = np.zeros((self.n, self.n), dtype=float)
        for k in range(self.n):
            C[k] = np.roll(first_row, k)
    
        # Optional scaling
        return C


    def toeplitz_reservoir(self, seed=None):
        """
        Real Toeplitz reservoir with controllable non-normality.
    
        Parameters
        ----------
        n : int
            Dimension.
        asymmetry : float
            0.0  -> symmetric Toeplitz (normal)
            1.0  -> maximally asymmetric (strongly non-normal)
            Values in [0, 1] interpolate between them.
        scale : float
            Overall scaling factor (e.g., to adjust spectral radius later).
        seed : int or None
            Random seed for reproducibility.
    
        Returns
        -------
        T : ndarray
            n×n real Toeplitz matrix.
        """
        if seed is not None:
            np.random.seed(seed)
    
        # Base diagonal (main)
        d0 = np.random.randn()
    
        # Upper and lower diagonals (length n-1)
        upper = np.random.randn(self.n - 1)
        lower = np.random.randn(self.n - 1)
    
        # Interpolate between symmetric and fully independent
        # asymmetry = 0 -> lower = upper (symmetric Toeplitz)
        # asymmetry = 1 -> lower = independent random (max asymmetry)
        lower_mixed = (1 - self.a) * upper + self.a * lower
    
        # Build first row and first column
        first_row = np.concatenate(([d0], upper))
        first_col = np.concatenate(([d0], lower_mixed))
    
        # Construct Toeplitz matrix
        T = np.empty((self.n, self.n), dtype=float)
        for i in range(self.n):
            # row i: starts at first_col[i] and then uses first_row[1:]
            T[i, :i+1] = first_col[i::-1]
            if i+1 < self.n:
                T[i, i+1:] = first_row[1:self.n-i]
    
        return T

    def non_normal_maintain_eigs(self,A):
        #In this case, need to make sure that a real matrix is returned, so
        #use output = 'real' in the schur decomposition
        T,Q = schur(A,output='real')
        if self.non_normal_type == 'sparse':
            N = self.sparse_upper_triangular()
        
        else:
            N = self.dense_upper_triangular()
        newA = Q @ (T+N) @ Q.T
        
        
        return newA
        
    def generate_A(self):
        if not self.maintain_res_eigs:
            if self.reservoir_type == 'erdos_renyi':
                G = nx.erdos_renyi_graph(self.n,self.res_density)
                A = nx.to_numpy_array(G)
                R = self.generate_res_distribution()
                R = (R+R.T)/2
                A = A*R
                if self.a>0:
                    if self.non_normal_type=='sparse':
                        A = A+self.a*self.sparse_upper_triangular()
                    elif self.non_normal_type == 'dense':
                        A = A+self.a*self.dense_upper_triangular()
            
            elif self.reservoir_type == 'barabasi_albert':
                m = int(np.floor(self.n*self.res_density))
                G = nx.barabasi_albert_graph(self.n,m)
                A = nx.to_numpy_array(G)
                R = self.generate_res_distribution()
                R = (R+R.T)/2
                A = A*R   
                if self.a>0:
                    if self.non_normal_type=='sparse':
                        A = A+self.a*self.sparse_upper_triangular()
                    elif self.non_normal_type == 'dense':
                        A = A+self.a*self.dense_upper_triangular()            
            
            elif self.reservoir_type == 'watts_strogatz':
                k = int(np.ceil(self.n*self.res_density))
                G = nx.watts_strogatz_graph(self.n,k,self.ws_p)
                A = nx.to_numpy_array(G)
                R = self.generate_res_distribution()
                R = (R+R.T)/2
                A = A*R
                if self.a>0:
                    if self.non_normal_type=='sparse':
                        A = A+self.a*self.sparse_upper_triangular()
                    elif self.non_normal_type == 'dense':
                        A = A+self.a*self.dense_upper_triangular()            
    
            elif self.reservoir_type == 'toeplitz':
                A = self.toeplitz_reservoir()
            
            elif self.reservoir_type == 'jacobi':
                A = self.jacobi_matrix()
            
            elif self.reservoir_type == 'circulant':
                A = self.circulant_reservoir()
            
            else: 
                raise ValueError("Note "+str(self.reservoir_type)+" is not an allowed reservoir type.")
        else:
            if self.reservoir_type == 'erdos_renyi':
                G = nx.erdos_renyi_graph(self.n,self.res_density)
                A = nx.to_numpy_array(G)
                R = self.generate_res_distribution()
                R = (R+R.T)/2
                A = A*R
                if self.a>0:
                    A = self.non_normal_maintain_eigs(A)
            
            elif self.reservoir_type == 'barabasi_albert':
                m = int(np.floor(self.n*self.res_density))
                G = nx.barabasi_albert_graph(self.n,m)
                A = nx.to_numpy_array(G)
                R = self.generate_res_distribution()
                R = (R+R.T)/2
                A = A*R   
                if self.a>0:
                    A = self.non_normal_maintain_eigs(A)         
            
            elif self.reservoir_type == 'watts_strogatz':
                k = int(np.ceil(self.n*self.res_density))
                G = nx.watts_strogatz_graph(self.n,k,self.ws_p)
                A = nx.to_numpy_array(G)
                R = self.generate_res_distribution()
                R = (R+R.T)/2
                A = A*R
                if self.a>0:
                    A = self.non_normal_maintain_eigs(A)          
    
            elif self.reservoir_type == 'toeplitz':
                A = self.toeplitz_reservoir()
            
            elif self.reservoir_type == 'jacobi':
                A = self.jacobi_matrix()
            
            elif self.reservoir_type == 'circulant':
                A = self.circulant_reservoir()
            
            else: 
                raise ValueError("Note "+str(self.reservoir_type)+" is not an allowed reservoir type.")            
        
        self.A = A

    def train_reservoir(self):
        Res = np.zeros((self.t+1,self.n))
        if self.data_in.ndim < 2:
            self.data_in = self.data_in.reshape(-1,1)
        for i in range(self.t):
            u = self.data_in[i,:]
            u = u.reshape(-1,1)
            #u = u.T
            #print("Winpart: ",np.dot(self.Win,u).shape)
            #print("Apart: ",np.dot(self.A,Res[i,:]).shape)
            Res[i+1,:] =  ((1-self.alpha)*Res[i,:].reshape(-1,1) + self.alpha*self.act((np.dot(self.A,Res[i,:].reshape(-1,1)))+(np.dot(self.Win,u)))).flatten()
        
        #eliminate the initial zeros
        Res = Res[1:,:]
        #Linear regression with Tikhonov regularization (of the two norm variety)
        Wout = np.dot(np.dot(self.data_out.T, Res), np.linalg.inv(np.dot(Res.T, Res) + self.beta * np.eye(self.n)) )
        self.Res_last = Res[-1,:]
        self.Wout = Wout
    
    def predict(self,test,nsteps=2000):
        if test.ndim < 2:
            test = test.reshape(-1,1)        
        pred = np.zeros((nsteps,self.data_out.shape[1]))
        Res = self.Res_last
        pred[0,:] = test.flatten()
        for i in range(nsteps-1):
            u = pred[i,:]
            u = u.reshape(-1,1)
            Res = (1-self.alpha)*Res.reshape(-1,1) + self.alpha*self.act((np.dot(self.A,Res.reshape(-1,1)))+(np.dot(self.Win,u)))
            p = np.dot(self.Wout,Res)
            pred[i+1,:] = p.flatten()
        
        return pred

    def valid_prediction_time(self,true, pred, threshold_factor=0.5):
        """
        Compute the valid prediction time (VPT) between true and predicted trajectories.
    
        Parameters
        ----------
        true : ndarray, shape (T,) or (T, d)
            Ground-truth trajectory.
        pred : ndarray, same shape as true
            Model prediction.
        threshold_factor : float
            Threshold = threshold_factor * std(true). 
            Typical values: 0.5, 1.0.
    
        Returns
        -------
        vpt : int
            Index of first time where error exceeds threshold.
            If never exceeded, returns len(true).
        """
        true = np.asarray(true)
        pred = np.asarray(pred)
    
        # Pointwise Euclidean error
        err = np.linalg.norm(true - pred, axis=-1)
    
        # Natural scale of the system
        sigma = np.std(true, axis=0)
        # If multidimensional, use average std across dims
        if np.ndim(sigma) > 0:
            sigma = np.mean(sigma)
    
        threshold = threshold_factor * sigma
    
        # Find first time error exceeds threshold
        idx = np.argmax(err > threshold)
    
        # If never exceeded, return full length
        if err[idx] <= threshold:
            return len(true)
    
        return idx
    
    def henrici_departure(self):
        T,Z = schur(self.A,output='complex')
        henrici = np.linalg.norm(T-np.diag(np.diagonal(T)),'fro')
        return henrici