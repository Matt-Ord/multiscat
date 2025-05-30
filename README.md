# Multiscat

Multiscat is a software package for simulating the scattering of atoms off of surfaces.

## Non-Reactive Scattering Calculations

At it's core, Multiscat is designed to perform non-reactive scattering calculations
of single atoms off a periodic surface. The behaviour of the atom can be described
by the Schrödinger equation

$$
i \hbar \frac{\partial}{\partial t} \Psi(\mathbf{r},t) = -\frac{\hbar^2}{2m} \nabla^2 \Psi(\mathbf{r},t) + V(\mathbf{r}) \Psi(\mathbf{r},t)
$$

where $V(\mathbf{r})$ is the potential energy of the atom at position $\mathbf{r}$.

Since we are dealing with non reactive scattering, we want to find a steady state
solution for the scattering wavefunction which enters the surface with a
well defined momentum $\mathbf{k}$. In other words, we want to find the
eigenstates of this equation which satisfy

$$
\frac{\hbar^2 k^2}{2m} \Psi(\mathbf{r})= -\frac{\hbar^2}{2m} \nabla^2 \Psi(\mathbf{r}) + V(\mathbf{r}) \Psi(\mathbf{r})
$$

subject to the aforementioned boundary conditions.

TODO: describe here how to use the code to find such eigenstates

## Scattering Channels and The Close-Coupling Method

In Multiscat, this is done using a technique known as the close-coupling method.
This technique makes use of bloch's theorem to simplify the problem. Since the surface is only
periodic in the direction parallel to the surface, in the following section we will split the
various coordinates into a parallel component which we use uppercase letters for, and a
perpendicular component. For this surface,
$V(\mathbf{X}, z) = V(\mathbf{X} + \mathbf{R}, z) $, and we can write both the potential
and wavefunction as a sum of plane waves

$$
V(\mathbf{r}, z) = \sum_{\mathbf{G}} V_{\mathbf{G}}(z) e^{i \mathbf{G} \cdot \mathbf{r}}
$$

$$
\Psi(\mathbf{r}, z) = \sum_{\mathbf{G}} \Psi_{\mathbf{G}}(z) e^{i (\mathbf{G} + \mathbf{K}) \cdot \mathbf{r}}
$$

where $\mathbf{G}$ are the reciprocal lattice vectors of the periodic surface,

TODO: define $\mathbf{G}$

and $\mathbf{K}$ is the component of momentum of the incoming atom in the direction parallel to the
surface.

$$
\mathbf{k} = \mathbf{K} + k_z \hat{\mathbf{z}}
$$

In principle, the calculation would require
an infinite number of $\mathbf{G}$ vectors, but in practice we can truncate this sum to a
finite number of states. Each vector $\mathbf{G}$ is known as a **Scattering Channel** with a
momentum $\mathbf{K} + \mathbf{G}$. Substituting back into the Schrödinger equation, we
obtain a set of coupled equations for the wavefunction in each scattering channel

$$
\frac{\hbar^2 k^2}{2m} \Psi_{\mathbf{G}}(z) =
-\frac{\hbar^2}{2m} \nabla_z^2 \Psi_{\mathbf{G}}(z)
+\frac{\hbar^2}{2m} (\mathbf{K} + \mathbf{G})^2 \Psi_{\mathbf{G}}(z)
+ \sum_{\mathbf{G}'} V_{ \mathbf{G}' - \mathbf{G}}(z) \Psi_{\mathbf{G}'}(z)
$$

to simplify this equation, we usually introduce $d^2_{\mathbf{G}} = k^2 - (\mathbf{K} + \mathbf{G})^2$
which is the energy associated with the z component of the wavefunction in the scattering channel $\mathbf{G}$.
If this energy is negative, then the channel is known as a **closed channel**, and the wavefunction
in such a channel must fall to zero as $z \to \infty$.

This gives us the final form of the coupled equations

$$
0 =
\nabla_z^2 \Psi_{\mathbf{G}}(z)
+ d^2_{\mathbf{G}} \Psi_{\mathbf{G}}(z)
- \frac{2m}{\hbar^2}\sum_{\mathbf{G}'} V_{ \mathbf{G}' - \mathbf{G}}(z) \Psi_{\mathbf{G}'}(z)
$$

these are then soloved, subject to the boundary conditions

- $\Psi_{\mathbf{G}}(z) \to 0$ as $z \to -\infty$
- $\Psi_{\mathbf{G}}(z) \to 0$ as $z \to \infty$ for closed channels
- $\Psi_{\mathbf{G}=\mathbf{0}}(z) \to q_{\mathbf{0}} + q_{\mathbf{0}}^* S_{\mathbf{0},\mathbf{0}}$ (the incoming channel)
- $\Psi_{\mathbf{G}}(z) \to q_{\mathbf{G}}^* S_{\mathbf{G},\mathbf{0}}$ for open channels

where $q_{\mathbf{G}} = \frac{1}{d_{\mathbf{G}}^{1/2}}e^{i d_{\mathbf{G}} z}$ is the incoming flux in the scattering channel $\mathbf{G}$, and $S_{\mathbf{G},\mathbf{G'}}$ is the **scattering matrix element** for the scattering channel $\mathbf{G}$ and the incoming channel $\mathbf{G'}$. This scattering matrix element, defined such that $||S_{\mathbf{G},\mathbf{G'}}||^2$ is the quantity we are primarily interested in when we do scattering calculations. This
matrix gives us the ratio of the outgoing flux in the scattering channel $\mathbf{G}$ to the incoming flux in the scattering channel $\mathbf{G'}$.

See David Riley's Thesis Chapter 3.2 for an alternative formulation of this method.

## The Lobatto Basis

Up until now we have been vague about the representation for the
wavefunction $\Psi_{\mathbf{G}}(z)$ at each channel. To perform
these calculations we need to choose a basis to represent the wavefunction
in the $z$ direction. In multiscat we use the Lobatto basis, which turns
out to be a really good choice for this purpose. The Lobatto functions are
orthogonal, sample points more densely near the surface, and have simple
cheap-to-calculate derivatives.

## The Log Derivative Method

It is possible to solve the closed coupled equations directly, but this
is poorly converging in general. In the case of surface scattering, we often
find that it is more efficient to minimize a functional proportional to
the log derivative matrix

$$
Y_{\mathbf{G}}(z) = \Psi'_{\mathbf{G}}(z)\Psi^{-1}_{\mathbf{G}}(z)
$$

Solving the closed coupled equations
