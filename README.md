# DFT Basic Eigensolver for the calculation of the ionization energy [Hartrees] by resolving the Kohm Sham equations using scipy, numpy and pandas, currently only uses an LDA exchange correlation functional
Warning! be careful with the conditonal of th while cycle for small basis sets be mindful of the expresion np.abs((Cte-Cteviej).sum())>.77333*(n*m)**2)
n=# of radial functions
m=# of spherical harmonics (we asumme m_l = 0)(latex)
Z=atomic number of the respective atom

What I want for the project
*Implementing plotting in spherical coordinates of the charge density
*Improvement of self confistent field methods i.e. when the iterations of the eigensolver should be halted, the limits of integration
*A full array of Exc functionals
*Neural network of any kind for the calculation of the best Exc functional
*Currently only the energies are calculated,any other observables should be benefitial
*Including more than one nucleus
*Possibility for polyatomic molecules
