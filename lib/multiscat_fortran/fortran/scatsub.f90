subroutine get_momentum_basis( &
& channel_count, specular_channel_index, channel_index_x, channel_index_y, &
& channel_energy_z, max_closed_channel_energy, max_channel_index &
   )
   implicit double precision (a-h,o-z)
!
! calculate reciprocal lattice
! calculate d (z-component of energy of outgoing wave) for
! each channel
!
! gax, gbx  = the unit vector of reciprocal lattice along symmetry
! direction
!
! gay, gby  = y components of unit vector of reciprocal lattice
! along symmetry direction 2
!
! ax, ay  = first in-plane unit-cell vector components
! bx, by  = second in-plane unit-cell vector components
!
!
! For each scattered channel:
! ered = square of incident wavevector for the channel
! eint = squre of surface component of the wavevector for the
! channel
!
! -d(i) = square of the z component of the wavevector for the channel
! energy conservation: ered = eint + (-d(i))
! if d(i) < 0, channel open, possible diffraction spot
! if d(i) > 0, channel closed, no spot
!
   include 'multiscat.inc'
   integer channel_count, specular_channel_index, max_channel_index
   dimension channel_energy_z(nmax)
   integer channel_index_x(nmax), channel_index_y(nmax)

   common /cells/ ax,ay,bx,by,ei,theta,phi,a0,gax,gay,gbx,gby
   common /const/ hemass,rmlmda ! = 2m/h^2 !modified by Boyao on 6 Dec 2020
   DATA   Pi /3.141592653589793d0/

   Auc=dabs(ax*by-ay*bx)
   if (Auc .le. 0.0d0) error stop 'ERROR: unit cell area must be positive.'
   RecUnit=2*Pi/Auc
   gax =  by*RecUnit
   gay = -bx*RecUnit
   gbx = -ay*RecUnit
   gby =  ax*RecUnit

   ered   = rmlmda*ei ! ered is just k_i^2
   thetad = theta*pi/180.0d0
   phid   = phi*pi/180.0d0

   pkx = sqrt(ered)*sin(thetad)*cos(phid)
   pky = sqrt(ered)*sin(thetad)*sin(phid)

   channel_count=0
   do i1 = -max_channel_index,max_channel_index
      do i2 = -max_channel_index,max_channel_index
         gx = gax*i1 + gbx*i2
         gy = gay*i1 + gby*i2
         eint = (pkx+gx)**2 + (pky+gy)**2
         di = eint-ered
         if (di.lt.max_closed_channel_energy) then
            channel_count=channel_count+1
            if (channel_count.le.nmax) then
               channel_index_x(channel_count)=i1
               channel_index_y(channel_count)=i2
               channel_energy_z(channel_count)=di
               if ((i1.eq.0) .and. (i2.eq.0)) &
               & specular_channel_index=channel_count
            else
               error stop 'ERROR: n too big! (basis)'
            end if
         end if
      end do
   end do


   return
end subroutine get_momentum_basis


subroutine build_lobatto_t_matrix (z_min,z_max,n_z_points,w,x,t)
   implicit double precision (a-h,o-z)
   include 'multiscat.inc'
!
! -----------------------------------------------------------------
! This subroutine calculates the kinetic energy matrix, T,
! in a normalised Lobatto shape function basis.
!
! Formula for this are taken from:
! "QUANTUM SCATTERING VIA THE LOG DERIVATIVE VERSION
! OF THE KOHN VARIATIONAL PRINCIPLE"
! D. E. Manolopoulos and R. E. Wyatt,
! Chem. Phys. Lett., 1988, 152,23
!
! In that paper Lobatto shape functions (Lsf) are defined
! -----------------------------------------------------------------
!
   integer n_z_points, alloc_status
   dimension w(n_z_points),x(n_z_points),t(n_z_points,n_z_points)

   double precision, allocatable :: ww(:), xx(:), tt(:,:)
   if (n_z_points .gt. mmax) error stop 'tshape 1'

! I think, that this is needed for the sum defined in Lsf to work
   n = n_z_points+1
   allocate(ww(n), xx(n), tt(n,n), stat=alloc_status)
   if (alloc_status /= 0) error stop 'ERROR: allocation failure (ww, xx, tt).'
! Get points and weights for n point Lobatto quadrature in (z_min,z_max)
   call compute_lobatto_rule (z_min,z_max,n,ww,xx)

! No idea why it's done
   do 1 i = 1,n
      ww(i) = sqrt(ww(i))
1  continue

   do 4 i = 1,n
      ff = 0.0d0
      do 3 j = 1,n
! gg of i = j is trivially = 0, so no need for loops
         if (j .eq. i) go to 3

! gg will be value of derivative of j-th Lsf at
! i-th root, which is: i-th Lsf evaluated at j-th
! root divided by ( i-th root minus j-th root )
         gg = 1.0d0/(xx(i)-xx(j))
         ff = ff+gg

         do 2 k = 1,n

! This loop multiplies gg defined above by j-th Lsf
! evaluated at i-th root, which is itself a Lagrangian interpolation
            if (k.eq.j .or. k.eq.i) go to 2
            gg = gg*(xx(j)-xx(k))/(xx(i)-xx(k))

2        continue

! Write into tt value of derivative of j-th Lsf
! evaluated at i-th root. This relation is described in the paper mentioned

         tt(j,i) = ww(j)*gg/ww(i)
! this appears wrong going by the given paper

3     continue
! tt of i,i is 0 as the i-th Lsf has a maximum at the i-th root, unless i=1 or i=n
      tt(i,i) = ff

4  continue
   do 7 i = 1,n_z_points
! In this approach 1: roots are in decreasing order
! 2: last root ( 0 ) doesn't get included in calculations
! but subroutine lobatto returns roots and weights in
! increasing order so this has to be done manually
      w(i) = ww(i+1)
      x(i) = xx(i+1)

      do 6 j = 1,i

         hh = 0.0d0
         do 5 k = 1,n
! Entries in T matrix are defined as a sum over all k from 0
! to n+1 of:

! [ k-th weight ]*[ derivative
! of i-th Lsf at k-th root ] *[ derivative
! of j-th Lsf at k-th root ]

            hh = hh + tt(k,i+1)*tt(k,j+1)
5        continue
! t is symmetric
         t(i,j) = hh
         t(j,i) = hh

6     continue

7  continue
   if (allocated(ww)) deallocate(ww)
   if (allocated(xx)) deallocate(xx)
   if (allocated(tt)) deallocate(tt)
   return
end subroutine build_lobatto_t_matrix



subroutine compute_lobatto_rule (interval_min,interval_max,node_count,w,x)
   implicit double precision (a-h,o-z)
! -----------------------------------------------------------------
! This subroutine calculates an n-point Gauss-Lobatto
! quadrature rule in the interval a < x < b.
!
! Function localizes zeros of the derivative of n-1 th Legendre
! polynomial and calculates associated weights.
!
! All recursive formulas are on Wikipedia except for P[n]''
! but it can be derived using others
!
! -----------------------------------------------------------------
!
   integer node_count
   double precision interval_min, interval_max
   dimension w(node_count),x(node_count)

! shift and scale have to be used to change integral in (a,b)
! to (-1,1) where lobatto quadrature works.
! I denote n-th Legendre polynomial as P[n] and its derivative as P[n]'
!
   l = (node_count+1)/2
! Isn't pi defined above as a double ?!?!?!
   pi = acos(-1.0d0)
! ---<See Docs/GaussianQuadrature.pdf, top of page 2>
   shift = 0.5d0*(interval_max+interval_min)
   scale = 0.5d0*(interval_max-interval_min)
   weight = (interval_max-interval_min)/ &
   & (node_count*(node_count-1))


! Specific to Lobatto quadrature, First point is interval_min
   x(1) = interval_min
   w(1) = weight

   do 3 k = 2,l
! As zeros are symmetric, there is only need to find positive ones
! z is approximated zero of P[node_count-1] using Francesco Tricomi approximation
! then accuracy of the zero is improved using Newton-Raphson
! few times ( arbitrary so far ). The cosine equation finds a point
! roughly halfway between 2 zeros of P[n].
      z = cos(pi*(4*k-3)/(4*node_count-2))
! Calculate value of P[node_count-1] at z using Bonnets recursive formula
! p1 is P[j], p2 = P[j-1], p3 = P[j-2]
      do 2 i = 1,7
         p2 = 0.0d0
         p1 = 1.0d0
         do 1 j = 1,node_count-1
            p3 = p2
            p2 = p1
            p1 = ((2*j-1)*z*p2-(j-1)*p3)/j
1        continue
! p2 gets overwritten to be P[node_count-1]'
         p2 = (node_count-1)*(p2-z*p1)/(1.0d0-z*z)
! p3 gets overwritten to be P[node_count-1]''
         p3 = (2.0d0*z*p2-node_count*(node_count-1)*p1) &
         & /(1.0d0-z*z)
! Actual Newton-Raphson step
         z = z-p2/p3
2     continue
! Write in shifted and scaled zeros and weights
      x(k) = shift-scale*z
! Write in zero with other sign
      x(node_count+1-k) = shift+scale*z
! Write in weights (they are always positive)
      w(k) = weight/(p1*p1)
      w(node_count+1-k) = w(k)
3  continue
! Specific to Lobatto quadrature, last point is interval_max
   x(node_count) = interval_max
   w(node_count) = weight
   return
end subroutine compute_lobatto_rule

subroutine compute_wave_terms (channel_energy,a,b,c,z_max)
   implicit double precision (a-h,o-z)
!
! ------------------------------------------------------------------
! Construction and storage of the diagonal matrices a,b and c
! that enter the log derivative Kohn expression for the S-matrix.
! ------------------------------------------------------------------
!
   complex(kind=8) a, b, c
!
   dk = sqrt (dabs(channel_energy))
   if (channel_energy .lt. 0.0d0) then
      theta = dk*z_max
      bcc   = cos(2.0d0*theta)
      bcs   = sin(2.0d0*theta)
      cc    = cos(theta)
      cs    = sin(theta)
      a  = cmplx(bcc,-bcs,kind=8)
      b  = (dk**0.5d0)*cmplx(cc,-cs,kind=8)
      c  = cmplx(0.0d0,dk,kind=8)
   else
      a = (0.0d0,0.0d0)
      b = (0.0d0,0.0d0)
      c = cmplx(-dk,0.0d0,kind=8)
   endif
   return
end subroutine compute_wave_terms

subroutine build_preconditioner (n_z_points,channel_count, &
& fourier_values,n_fourier_components, &
& specular_fourier_component_index,channel_energy_z, &
& eigenvalues,preconditioner_factors,kinetic_matrix)
   implicit double precision (a-h,o-z)
   include 'multiscat.inc'
   integer n_z_points, channel_count
   integer n_fourier_components, specular_fourier_component_index
!
! ------------------------------------------------------------------
! This subroutine constructs the matrix factors that are required
! for the block lower triangular preconditioner used in GMRES.
!
! t is the matrix from tshapes
! ------------------------------------------------------------------
!
   complex(kind=8) fourier_values(n_z_points,n_fourier_components)
   dimension channel_energy_z(channel_count)
   dimension eigenvalues(n_z_points)
   dimension preconditioner_factors(n_z_points,channel_count)
   double precision kinetic_matrix(n_z_points,n_z_points)
   dimension g(mmax)

! if m exceeds maximum size of m program terminates printing out precon 1
   if (n_z_points .gt. mmax) error stop 'precon 1'
!
   do k = 1,n_z_points
! dble(x) transfroms input into double precision real
! for complex numbers it will return only real part
! This overwrites t to be H0 (as named in '90 Kohn paper)
      kinetic_matrix(k,k) = kinetic_matrix(k,k) + &
      & dble(fourier_values(k, &
      & specular_fourier_component_index))
   enddo

! get eigenvalues [e] and eigenvectors [ overwrite them on t] of t,
! which is m x m symmetric real matrix. f is temporary storage array
   call rs (n_z_points,n_z_points,kinetic_matrix,eigenvalues, &
   & kinetic_matrix,preconditioner_factors,ierr)
! if ierr != 0 program terminates printing out precon 2, meaning rs failed
   if (ierr .ne. 0) error stop 'precon 2'

   do j = 1,channel_count

      do k = 1,n_z_points
         g(k) = kinetic_matrix(n_z_points,k) &
         & /(channel_energy_z(j)+eigenvalues(k))
         preconditioner_factors(k,j) = 0.0d0
      enddo

      do i = 1,n_z_points
         do k = 1,n_z_points
            preconditioner_factors(k,j) = &
            & preconditioner_factors(k,j) + kinetic_matrix(k,i)*g(i)
         enddo
      enddo

   enddo
   return
end subroutine build_preconditioner

subroutine solve_gmres_system (x,xx,y,n_z_points, &
& channel_index_x,channel_index_y, &
& channel_count,specular_channel_index, &
& fourier_values,fourier_index_x,fourier_index_y, &
& n_fourier_components, &
& a,b,c,d,e,f,p,s,t,convergence_eps, &
& preconditioner_flag,ifail)
   implicit double precision (a-h,o-z)
!
! -----------------------------------------------------------------
! Complex Generalised Minimal Residual Algorithm (GMRES)
! This version written by DEM, 6/12/94
! -----------------------------------------------------------------
!
   integer n_z_points, channel_count, specular_channel_index
   integer n_fourier_components, preconditioner_flag
   integer alloc_status
   complex(kind=8) x(n_z_points*channel_count), y(n_z_points*channel_count)
   complex(kind=8) fourier_values(n_z_points,n_fourier_components)
   complex(kind=8) a(channel_count), b(channel_count), c(channel_count)
   complex(kind=8) s(channel_count)
   dimension d(channel_count), e(n_z_points)
   dimension f(n_z_points,channel_count), p(channel_count)
   dimension t(n_z_points,n_z_points)
   integer channel_index_x(channel_count), channel_index_y(channel_count)
   integer fourier_index_x(n_fourier_components)
   integer fourier_index_y(n_fourier_components)
!
! NB:
! This subroutine implements a preconditioned version of GMRES(l).
! The rate of convergence can be improved (at the expense of a
! greater disk space requirement and more cpu time per iteration)
! by increasing the following parameter:
!
! parameter (l = 100)  increased l to 200, 22/4/1999, APJ
   parameter (l = 2000)
   complex(kind=8), allocatable :: h(:,:), g(:), z(:)
   complex(kind=8), allocatable :: co(:), si(:)
   complex(kind=8) temp
!
! store x matrices in xx rather than write to disk.
!
   complex(kind=8) xx(n_z_points*channel_count,l+1)

   allocate(h(l+1,l+1), g(l+1), z(l+1), co(l+1), si(l+1), stat=alloc_status)
   if (alloc_status /= 0) error stop 'ERROR: allocation failure (h, g, z, co, si).'
!
! Setup for GMRES(l):
!
   mn = n_z_points*channel_count
   do i = 1,mn
      x(i) = (0.0d0,0.0d0)
   enddo

!
! Initial step:
!
   kount = 0
   xx(1:mn,1)=x(1:mn)
   do i = 1,mn
      y(i) = x(i)
   enddo
   call apply_upper_block (x,n_z_points,channel_index_x, &
   & channel_index_y,channel_count,fourier_values, &
   & fourier_index_x,fourier_index_y,n_fourier_components)
   do i = 1,mn
      x(i) = -x(i)
   enddo
   x(n_z_points*specular_channel_index) = &
   & b(specular_channel_index)+x(n_z_points*specular_channel_index)
   call solve_lower_block (x,n_z_points,channel_index_x, &
   & channel_index_y,channel_count,fourier_values, &
   & fourier_index_x,fourier_index_y,n_fourier_components, &
   & c,d,e,f,t)
   do i = 1,mn
      x(i) = x(i)-y(i)
   enddo
   if (preconditioner_flag .eq. 1) then
      do i = 1,mn
         y(i) = x(i)
      enddo
      call apply_upper_block (x,n_z_points,channel_index_x, &
      & channel_index_y,channel_count,fourier_values, &
      & fourier_index_x,fourier_index_y,n_fourier_components)
      call solve_lower_block (x,n_z_points,channel_index_x, &
      & channel_index_y,channel_count,fourier_values, &
      & fourier_index_x,fourier_index_y,n_fourier_components, &
      & c,d,e,f,t)
      do i = 1,mn
         x(i) = y(i)-x(i)
      enddo
   endif
   xnorm = 0.0d0
   do i = 1,mn
      xnorm = xnorm + dble(conjg(x(i))*x(i))
   enddo
   xnorm = sqrt(xnorm)
   g(1) = xnorm
!
! Generic recursion: (you mean iteration?)
!
   kconv = 0
   do j = 1,channel_count
      p(j) = 0.0d0
   enddo
   do k = 1,l
      kount = kount+1
      do i = 1,mn
         x(i) = x(i)/xnorm
      enddo
      xx(1:mn,k+1)=x(1:mn)
      do i = 1,mn
         y(i) = x(i)
      enddo
      call apply_upper_block (x,n_z_points,channel_index_x, &
      & channel_index_y,channel_count,fourier_values, &
      & fourier_index_x,fourier_index_y,n_fourier_components)
      call solve_lower_block (x,n_z_points,channel_index_x, &
      & channel_index_y,channel_count,fourier_values, &
      & fourier_index_x,fourier_index_y,n_fourier_components, &
      & c,d,e,f,t)
      do i = 1,mn
         x(i) = y(i)+x(i)
      enddo
      if (preconditioner_flag .eq. 1) then
         do i = 1,mn
            y(i) = x(i)
         enddo
         call apply_upper_block (x,n_z_points,channel_index_x, &
         & channel_index_y,channel_count,fourier_values, &
         & fourier_index_x,fourier_index_y,n_fourier_components)
         call solve_lower_block (x,n_z_points,channel_index_x, &
         & channel_index_y,channel_count,fourier_values, &
         & fourier_index_x,fourier_index_y,n_fourier_components, &
         & c,d,e,f,t)
         do i = 1,mn
            x(i) = y(i)-x(i)
         enddo
      endif
      y(1:mn)=xx(1:mn,1)
      do i = 1,channel_count
         s(i) = y(n_z_points*i)
      enddo
      do j = 1,k
         y(1:mn)=xx(1:mn,j+1)
         h(j,k) = (0.0d0,0.0d0)
         do i = 1,mn
            h(j,k) = h(j,k)+conjg(y(i))*x(i)
         enddo
         do i = 1,mn
            x(i) = x(i)-y(i)*h(j,k)
         enddo
         if (j .lt. k) then
            do i = 1,channel_count
               s(i) = s(i)+y(n_z_points*i)*z(j)
            enddo
         endif
      enddo
      do i = 1,channel_count
         s(i) = (0.0d0,2.0d0)*b(i)*s(i)
      enddo
      s(specular_channel_index) = &
      & a(specular_channel_index)+s(specular_channel_index)
      xnorm = 0.0d0
      do i = 1,mn
         xnorm = xnorm + dble(conjg(x(i))*x(i))
      enddo
      xnorm = sqrt(xnorm)
      h(k+1,k) = xnorm
      do j = 1,k-1
         temp = co(j)*h(j,k)+conjg(si(j))*h(j+1,k)
         h(j+1,k) = conjg(co(j))*h(j+1,k)-si(j)*h(j,k)
         h(j,k) = temp
      enddo
      call compute_complex_givens_rotation (h(k,k),h(k+1,k),co(k),si(k))
      g(k+1) = -si(k)*g(k)
      g(k) = co(k)*g(k)
      do j = 1,k
         z(j) = g(j)
      enddo
      do j = k,1,-1
         z(j) = z(j)/h(j,j)
         do i = 1,j-1
            z(i) = z(i)-h(i,j)*z(j)
         enddo
      enddo
!
! Convergence test:
!
      unit = 0.0d0
      diff = 0.0d0
      do j = 1,channel_count
         pj = dble(conjg(s(j))*s(j))
         unit = unit+pj
         diff = max(diff,abs(pj-p(j)))
         p(j) = pj
      enddo
      diff = max(diff,abs(unit-1.0d0))
      if (diff .lt. convergence_eps) then
         kconv = kconv+1
      else
         kconv = 0
      endif
      kk = k
      if (kconv.eq.3 .or. xnorm.eq.0.0d0) go to 2
   enddo
2  continue
!
! back substitution for x:
!
   x(1:mn)=xx(1:mn,1)
   do j = 1,kk
      y(1:mn)=xx(1:mn,j+1)
      do i = 1,mn
         x(i) = x(i)+y(i)*z(j)
      enddo
   enddo
!
! all done?
!
   if (kconv.lt.3 .and. xnorm.gt.0.0d0) then
      ifail=1
   endif
!
! yes!
!
   if (allocated(h)) deallocate(h)
   if (allocated(g)) deallocate(g)
   if (allocated(z)) deallocate(z)
   if (allocated(co)) deallocate(co)
   if (allocated(si)) deallocate(si)
   return
end subroutine solve_gmres_system

subroutine apply_upper_block (state_vector,n_z_points, &
& channel_index_x,channel_index_y,channel_count, &
& fourier_values,fourier_index_x,fourier_index_y, &
& n_fourier_components)
   implicit double precision (a-h,o-z)
!
! ----------------------------------------------------------
! This subroutine performs the block upper triangular
! matrix multiplication y = U*x, where A = L+U.
! The result y is overwritten on x on return.
! ----------------------------------------------------------
!
   integer n_z_points, channel_count, n_fourier_components
   complex(kind=8) state_vector(n_z_points,channel_count)
   complex(kind=8) fourier_values(n_z_points,n_fourier_components)
   integer channel_index_x(channel_count), channel_index_y(channel_count)
   integer fourier_index_x(n_fourier_components)
   integer fourier_index_y(n_fourier_components)
!
   do j = 1,channel_count
      do k = 1,n_z_points
         state_vector(k,j) = (0.0d0,0.0d0)
      enddo
      do i = j+1,channel_count
         do l = 1,n_fourier_components
            if (channel_index_x(i) + fourier_index_x(l) &
            & .ne. channel_index_x(j)) go to 1
            if (channel_index_y(i) + fourier_index_y(l) &
            & .ne. channel_index_y(j)) go to 1
            do k = 1,n_z_points
               state_vector(k,j) = state_vector(k,j) + &
               & fourier_values(k,l)*state_vector(k,i)
            enddo
1           continue
         enddo
      enddo
   enddo
   return
end subroutine apply_upper_block

subroutine solve_lower_block (state_vector,n_z_points, &
& channel_index_x,channel_index_y,channel_count, &
& fourier_values,fourier_index_x,fourier_index_y, &
& n_fourier_components,wave_c,channel_energy_z, &
& eigenvalues,preconditioner_factors,kinetic_matrix)
   implicit double precision (a-h,o-z)
   include 'multiscat.inc'
!
! ----------------------------------------------------------
! This subroutine solves the block lower triangular
! linear equation L*y = x, where A = L+U.
! The result y is overwritten on x on return.
! ----------------------------------------------------------
!
   integer n_z_points, channel_count, n_fourier_components
   complex(kind=8) state_vector(n_z_points,channel_count)
   complex(kind=8) fourier_values(n_z_points,n_fourier_components)
   complex(kind=8) wave_c(channel_count)
   integer channel_index_x(channel_count), channel_index_y(channel_count)
   integer fourier_index_x(n_fourier_components)
   integer fourier_index_y(n_fourier_components)
   dimension channel_energy_z(channel_count), eigenvalues(n_z_points)
   dimension preconditioner_factors(n_z_points,channel_count)
   double precision kinetic_matrix(n_z_points,n_z_points)
!

!parameter (mmax = 200)
   complex(kind=8) y(mmax), fac
   if (n_z_points .gt. mmax) error stop 'lower 1'
!
   do j = 1,channel_count
      do i = 1,j-1
         do l = 1,n_fourier_components
            if (channel_index_x(i) + fourier_index_x(l) &
            & .ne. channel_index_x(j)) go to 1
            if (channel_index_y(i) + fourier_index_y(l) &
            & .ne. channel_index_y(j)) go to 1
            do k = 1,n_z_points
               state_vector(k,j) = state_vector(k,j) &
               & - fourier_values(k,l)*state_vector(k,i)
            enddo
1           continue
         enddo
      enddo
      do k = 1,n_z_points
         y(k) = (0.0d0,0.0d0)
         do l = 1,n_z_points
            y(k) = y(k)+state_vector(l,j)*kinetic_matrix(l,k)
         enddo
         y(k) = y(k)/(channel_energy_z(j)+eigenvalues(k))
      enddo
      do k = 1,n_z_points
         state_vector(k,j) = (0.0d0,0.0d0)
      enddo
      do l = 1,n_z_points
         do k = 1,n_z_points
            state_vector(k,j) = state_vector(k,j)+kinetic_matrix(k,l)*y(l)
         enddo
      enddo
      fac = state_vector(n_z_points,j)*wave_c(j)/ &
      & (1.0d0-preconditioner_factors(n_z_points,j)*wave_c(j))
      do k = 1,n_z_points
         state_vector(k,j) = state_vector(k,j) + &
         & fac*preconditioner_factors(k,j)
      enddo
   enddo
   return
end subroutine solve_lower_block

subroutine compute_complex_givens_rotation (input_a,input_b,c,s)
   implicit complex*16 (a-h,o-z)
   double precision scale,r
   complex(kind=8) input_a, input_b
!
! -----------------------------------------------------------------
! This subroutine constructs and performs a complex Givens rotation
! -----------------------------------------------------------------
!
   rho = input_b
   if (abs(input_a) .gt. abs(input_b)) rho = input_a
   scale = abs(input_a) + abs(input_b)
   if (scale .eq. 0.0d0) then
      c = 1.0d0
      s = 0.0d0
      r = 0.0d0
   else
      r = dble((input_a/scale)*conjg(input_a/scale) + &
      & (input_b/scale)*conjg(input_b/scale))
      r = scale*sqrt(r)
      if (dble(rho) .lt. 0.0d0) r = -r
      c = conjg(input_a/r)
      s = input_b/r
   end if
   z = 1.0d0
   if (abs(input_a) .gt. abs(input_b)) z = s
   if (abs(input_b) .ge. abs(input_a) .and. abs(c) .ne. 0.0d0) &
   & z = 1.0d0/c
   input_a = r
   input_b = z
   return
end subroutine compute_complex_givens_rotation
