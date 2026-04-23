# SWAPI Solar Wind Moments Algorithm Notes

Described in a separate document is SWAPI's effective area function $\mathcal{A}^s(\mathbf{v}, V)$ for particles of species $s$ and velocity vector $\mathbf{v}$ while SWAPI is set to ESA voltage $V$.
Given a VDF $f(\mathbf{v})$, the count rate that SWAPI observes is
$$
C(V) = \sum_s \int f^s(\mathbf{v}) \mathcal{A}^s(\mathbf{v}, V)
$$

- **TODO**: Describe the effective area function and how it's integrated

- **TODO**: Describe the proton Maxwellian distribution

The algorithm for fitting the pro