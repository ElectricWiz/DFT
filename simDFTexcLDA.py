
import numpy as np
import sympy as sym
import seaborn as sns
import pandas as pd

from scipy.integrate import quad, dblquad
from scipy import linalg


x1 =  sym.Symbol("x1")
x2 = sym.Symbol("x2")

Z = 5
n = 3
m = 2

N = 1
lim = .00000133

Esx = np.array([]).reshape(m*n,0)
Ctsx = np.zeros((m*n,m*n))

print(Esx)

def cte_norm(n):
    return 1/np.sqrt(float(sym.integrate(sym.exp(-2*Z*x1/n)*x1**(2*(n-1)),(x1, 0, float("inf")))))

def func_prueba(n):
    return sym.exp(-Z*x2/n)*x2**(n-1)*cte_norm(n)

contadors = [0,0]

h = np.vectorize(np.abs)
g = np.vectorize(np.real)

for i in range(N):
    contadors[1] += 1
    Cte = np.random.rand(n*m, n*m) + np.random.rand(n*m, n*m)*1j
    Cteviej = np.zeros((n*m, n*m))
    Y = h(g(np.diagonal(Cte))).sum()
    contadors[0] = 0
    
    while ((np.abs(Y-(n*m))>(6/15)*(n*m)) & (np.abs((Cte-Cteviej).sum())>.007333*(n*m)**2))|(contadors[0]<3):
        contadors[0] += 1
        
        def poperado(n,p):
            a = sym.diff(x2*func_prueba(n), x2)
            return (sym.diff(a, x2)*x2**(-1))

        def Tpq(p,q):
            if np.abs(p-q)>n-1:
                return 0
            else:
                if p>n and p<=n*2:
                    L = 2
                elif p>n*2:
                    L = 6
                else:
                    

                a = float(sym.re(sym.integrate(.5*L * func_prueba(p) * func_prueba(q) *(Cte[p-1][q-1]*np.conj(Cte[p - 1][q-1]))*x2**(-2), (x2,lim, float("inf")))))
                b = float(sym.re(sym.integrate((-.5)*poperado(p,q)*func_prueba(p)*(Cte[p-1][q-1]*np.conj(Cte[p-1][q-1])), (x2, lim, float("inf")))))
                return a + b

        def dens_cargax1(n,m):
            if np.abs(n-m)>n-1:
                return 0
            else:
                return sym.exp(-2*Z*x1/n)*x1**(n+m-2)*(np.conj(Cte[n-1][m-1])*Cte[n-1][m-1])*cte_norm(n)*cte_norm(m)

        def dens_cargax2(n,m):
            if np.abs(n-m)>n-1:
                return 0
            else:
                return sym.exp(-2*Z*x2/n)*x2**(n+m-2)*(np.conj(Cte[n-1][m-1])*Cte[n-1][m-1])*cte_norm(n)*cte_norm(m)

        dens_tot_para_elecx1 = 0
        for i in range(n*m):
            for k in range(n*m):
                dens_tot_para_elecx1 += dens_cargax1(i+1,k+1)

        dens_tot_para_elecx2 = 0
        for i in range(n*m):
            for k in range(n*m):
                dens_tot_para_elecx2 += dens_cargax2(i+1,k+1)
        
        
        def Vpq(p,q):
            if np.abs(p-q)>n-1:
                return 0
            else:
                g = sym.utilities.lambdify(x2, func_prueba(p), "numpy")
                h = sym.utilities.lambdify(x2, func_prueba(q)*(Cte[p-1][q-1]*np.conj(Cte[p-1][q-1])), "numpy")
                def e(x):
                    return -(g(x)*h(x)*Z)/x
            return quad(e,lim,float("inf"))[0]

        def Expq(p,q):
            if p>n and p<=n*2:
                p1 = p - n
            elif p>n*2:
                p1 = p - 2*n
            else:
                p1 = p
            if q>n and q<=n*2:
                q1 = q - n
            elif q>n*2:
                q1 = q - 2*n
            else:
                q1 = q
            f = sym.utilities.lambdify(x2, dens_tot_para_elecx2**(1/3), "numpy")
            g = sym.utilities.lambdify(x2, func_prueba(p1), "numpy")
            h = sym.utilities.lambdify(x2, func_prueba(q1)*(Cte[p-1][q-1]*np.conj(Cte[p-1][q-1])), "numpy")
            c = -1*np.power(3/np.pi, 1/3)
            def e(x2):
                return c*f(x2)*g(x2)*h(x2)
            return quad(e, lim, np.inf)[0]

        def Zpq(p,q):
            if p>n and p<=n*2:
                p1 = p - n
            elif p>n*2:
                p1 = p - 2*n
            else:
                p1 = p
            if q>n and q<=n*2:
                q1 = q - n
            elif q>n*2:
                q1 = q - 2*n
            else:
                q1 = q
            f = sym.utilities.lambdify(x1, dens_tot_para_elecx1, "numpy")
            g = sym.utilities.lambdify(x2, func_prueba(p1), "numpy")
            h = sym.utilities.lambdify(x2, func_prueba(q1)*(Cte[p-1][q-1]*np.conj(Cte[p-1][q-1])), "numpy")
            def e(x1,x2):
                return f(x1)*g(x2)*h(x2)/(np.abs(x1-x2)+.0000001)
            return dblquad(e,lim, 1000000,lambda x2: lim, lambda x2: 1000000)[0]

        def Hpq(p,q):
            return Tpq(p,q) + Vpq(p,q) + Expq(p,q) + Zpq(p,q)

        H = np.zeros((n*m,n*m))
        S = np.zeros((n*m,n*m))
        for k in range(n*m):
            for i in range(n*m):
                H[i][k]=Hpq(i+1,k+1)


        def Spq(p,q):
            if p>n and p<=n*2:
                p1 = p - n
            elif p>n*2:
                p1 = p - 2*n
            else:
                p1 = p
            if q>n and q<=n*2:
                q1 = q - n
            elif q>n*2:
                q1 = q - 2*n
            else:
                q1 = q
            return float(sym.integrate(func_prueba(p1)*(np.conj(Cte[p-1][q-1])*(Cte[p-1][q-1]))*func_prueba(q1),(x2,lim,float("inf"))))

        for i in range(n*m):
            for k in range(n*m):
                S[i][k]=Spq(i+1,k+1)
           
        h = linalg.eig(H,S, left=False, right=True)
        Cte = h[1]
        Y = g(np.diagonal(Cte)).sum()
        print(contadors)
        
    Eigs = []
    
    for x in linalg.eigvals(H):
        Eigs.append(x)

    Esx = np.concatenate(Esx,np.asarray(Eigs).reshape((m*n,0)),axis=1)

    for i in range(n*m):
        for k in range(n*m):
            Cte[i][k] = np.real(Cte[i][k])

    if contadors[1] == 1:
        Ctsx = Cte
    else:
        Ctsx = np.concatenate((Ctsx,Cte),axis=0)  
        
    print(contadors)
    
Indx1 = []
Indx2 = []
Ind = []

for i in range(N):
    Ind.append(i+1)
    for k in range(m*n):
        Indx1.append(i+1)
        Indx2.append(k+1)

Indx3 = [Indx1,Indx2]
Indx = list(zip(*Indx3))

index = pd.MultiIndex.from_tuples(Indx, names=["#Simulacion", "#Funcion de onda"])

Dentzes = pd.DataFrame(Ctsx,index=index)
Ene = pd.DataFrame(Esx)

dens = Dentzes.mean(level="#Funcion de onda")

df1 = Ene.transpose()
print(dens)
print(Dentzes.std(level="#Funcion de onda"))
df2 = df1.mean(axis=0)
print(df2.sum())
df3 = df1.describe()
print(df3)

dens.to_csv("constantes.csv")
dens.describe().to_csv("ctesstat.csv")
