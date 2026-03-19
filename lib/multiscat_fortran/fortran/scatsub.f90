module scatsub_basis
   use, intrinsic :: iso_fortran_env, only: real64
   implicit none
   private

   integer, parameter :: dp = real64
   integer, parameter :: nmax_basis = 1024
   integer, parameter :: mmax = 550

   public :: ChannelBasisData
   public :: populate_momentum_basis
   public :: build_lobatto_t_matrix
   public :: compute_wave_terms

   type :: ChannelBasisData
      integer :: channel_count = 0
      integer :: specular_channel_index = 0
      integer :: channel_index_x(nmax_basis) = 0
      integer :: channel_index_y(nmax_basis) = 0
      real(dp) :: channel_energy_z(nmax_basis) = 0.0_dp
   end type ChannelBasisData

contains

   subroutine populate_momentum_basis( &
   & basis_data, max_closed_channel_energy, max_channel_index, &
   & unit_cell_ax, unit_cell_ay, unit_cell_bx, unit_cell_by, &
   & helium_mass, incident_energy_mev, theta_degrees, phi_degrees)
      implicit none
      type(ChannelBasisData), intent(inout) :: basis_data
      real(dp), intent(in) :: max_closed_channel_energy
      integer, intent(in) :: max_channel_index
      real(dp), intent(in) :: unit_cell_ax, unit_cell_ay, unit_cell_bx, unit_cell_by
      real(dp), intent(in) :: helium_mass
      real(dp), intent(in) :: incident_energy_mev, theta_degrees, phi_degrees

      real(dp), parameter :: pi = 3.141592653589793_dp
      real(dp) :: rmlmda
      real(dp) :: auc, recunit, ered, thetad, phid, pkx, pky
      real(dp) :: gax, gay, gbx, gby
      real(dp) :: gx, gy, eint, di
      integer :: i1, i2

      rmlmda = 2.0_dp * helium_mass / 4.18020_dp

      auc = abs(unit_cell_ax*unit_cell_by-unit_cell_ay*unit_cell_bx)
      if (auc .le. 0.0d0) error stop 'ERROR: unit cell area must be positive.'
      recunit = 2*pi/auc
      gax =  unit_cell_by*recunit
      gay = -unit_cell_bx*recunit
      gbx = -unit_cell_ay*recunit
      gby =  unit_cell_ax*recunit

      ered   = rmlmda*incident_energy_mev
      thetad = theta_degrees*pi/180.0d0
      phid   = phi_degrees*pi/180.0d0

      pkx = sqrt(ered)*sin(thetad)*cos(phid)
      pky = sqrt(ered)*sin(thetad)*sin(phid)

      basis_data%channel_count = 0
      basis_data%specular_channel_index = 0
      do i1 = -max_channel_index,max_channel_index
         do i2 = -max_channel_index,max_channel_index
            gx = gax*i1 + gbx*i2
            gy = gay*i1 + gby*i2
            eint = (pkx+gx)**2 + (pky+gy)**2
            di = eint-ered
            if (di.lt.max_closed_channel_energy) then
               basis_data%channel_count = basis_data%channel_count+1
               if (basis_data%channel_count.le.nmax_basis) then
                  basis_data%channel_index_x(basis_data%channel_count) = i1
                  basis_data%channel_index_y(basis_data%channel_count) = i2
                  basis_data%channel_energy_z(basis_data%channel_count) = di
                  if ((i1.eq.0) .and. (i2.eq.0)) &
                  & basis_data%specular_channel_index = basis_data%channel_count
               else
                  error stop 'ERROR: n too big! (basis)'
               end if
            end if
         end do
      end do
   end subroutine populate_momentum_basis
   subroutine build_lobatto_t_matrix (z_min,z_max,n_z_points,w,x,t)
      implicit none
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
      real(dp), intent(in) :: z_min, z_max
      integer, intent(in) :: n_z_points
      real(dp), intent(out) :: w(n_z_points),x(n_z_points),t(n_z_points,n_z_points)
      integer :: alloc_status
      integer :: n, i, j, k
      real(dp) :: ff, gg, hh

      real(dp), allocatable :: ww(:), xx(:), tt(:,:)
      if (n_z_points .gt. mmax) error stop 'tshape 1'

! I think, that this is needed for the sum defined in Lsf to work
      n = n_z_points+1
      allocate(ww(n), xx(n), tt(n,n), stat=alloc_status)
      if (alloc_status /= 0) error stop 'ERROR: allocation failure (ww, xx, tt).'
! Get points and weights for n point Lobatto quadrature in (z_min,z_max)
      call compute_lobatto_rule (z_min,z_max,n,ww,xx)

! No idea why it's done
      do i = 1,n
         ww(i) = sqrt(ww(i))
      end do

      do i = 1,n
         ff = 0.0d0
         do j = 1,n
! gg of i = j is trivially = 0, so no need for loops
            if (j .eq. i) cycle

! gg will be value of derivative of j-th Lsf at
! i-th root, which is: i-th Lsf evaluated at j-th
! root divided by ( i-th root minus j-th root )
            gg = 1.0d0/(xx(i)-xx(j))
            ff = ff+gg

            do k = 1,n

! This loop multiplies gg defined above by j-th Lsf
! evaluated at i-th root, which is itself a Lagrangian interpolation
               if (k.eq.j .or. k.eq.i) cycle
               gg = gg*(xx(j)-xx(k))/(xx(i)-xx(k))

            end do

! Write into tt value of derivative of j-th Lsf
! evaluated at i-th root. This relation is described in the paper mentioned

            tt(j,i) = ww(j)*gg/ww(i)
! this appears wrong going by the given paper

         end do
! tt of i,i is 0 as the i-th Lsf has a maximum at the i-th root, unless i=1 or i=n
         tt(i,i) = ff

      end do
      do i = 1,n_z_points
! In this approach 1: roots are in decreasing order
! 2: last root ( 0 ) doesn't get included in calculations
! but subroutine lobatto returns roots and weights in
! increasing order so this has to be done manually
         w(i) = ww(i+1)
         x(i) = xx(i+1)

         do j = 1,i

            hh = 0.0d0
            do k = 1,n
! Entries in T matrix are defined as a sum over all k from 0
! to n+1 of:

! [ k-th weight ]*[ derivative
! of i-th Lsf at k-th root ] *[ derivative
! of j-th Lsf at k-th root ]

               hh = hh + tt(k,i+1)*tt(k,j+1)
            end do
! t is symmetric
            t(i,j) = hh
            t(j,i) = hh

         end do

      end do
      if (allocated(ww)) deallocate(ww)
      if (allocated(xx)) deallocate(xx)
      if (allocated(tt)) deallocate(tt)
      return
   end subroutine build_lobatto_t_matrix


   subroutine compute_lobatto_rule (interval_min,interval_max,node_count,w,x)
      implicit none
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
      integer, intent(in) :: node_count
      real(dp), intent(in) :: interval_min, interval_max
      real(dp), intent(out) :: w(node_count),x(node_count)
      integer :: l, k, i, j
      real(dp) :: pi, shift, scale, weight
      real(dp) :: z, p1, p2, p3

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

      do k = 2,l
! As zeros are symmetric, there is only need to find positive ones
! z is approximated zero of P[node_count-1] using Francesco Tricomi approximation
! then accuracy of the zero is improved using Newton-Raphson
! few times ( arbitrary so far ). The cosine equation finds a point
! roughly halfway between 2 zeros of P[n].
         z = cos(pi*(4*k-3)/(4*node_count-2))
! Calculate value of P[node_count-1] at z using Bonnets recursive formula
! p1 is P[j], p2 = P[j-1], p3 = P[j-2]
         do i = 1,7
            p2 = 0.0d0
            p1 = 1.0d0
            do j = 1,node_count-1
               p3 = p2
               p2 = p1
               p1 = ((2*j-1)*z*p2-(j-1)*p3)/j
            end do
! p2 gets overwritten to be P[node_count-1]'
            p2 = (node_count-1)*(p2-z*p1)/(1.0d0-z*z)
! p3 gets overwritten to be P[node_count-1]''
            p3 = (2.0d0*z*p2-node_count*(node_count-1)*p1) &
            & /(1.0d0-z*z)
! Actual Newton-Raphson step
            z = z-p2/p3
         end do
! Write in shifted and scaled zeros and weights
         x(k) = shift-scale*z
! Write in zero with other sign
         x(node_count+1-k) = shift+scale*z
! Write in weights (they are always positive)
         w(k) = weight/(p1*p1)
         w(node_count+1-k) = w(k)
      end do
! Specific to Lobatto quadrature, last point is interval_max
      x(node_count) = interval_max
      w(node_count) = weight
      return
   end subroutine compute_lobatto_rule

   subroutine compute_wave_terms (channel_energy,a,b,c,z_max)
      implicit none
!
! ------------------------------------------------------------------
! Construction and storage of the diagonal matrices a,b and c
! that enter the log derivative Kohn expression for the S-matrix.
! ------------------------------------------------------------------
!
      real(dp), intent(in) :: channel_energy, z_max
      complex(dp), intent(out) :: a, b, c
      real(dp) :: dk, theta, bcc, bcs, cc, cs
!
      dk = sqrt (abs(channel_energy))
      if (channel_energy .lt. 0.0d0) then
         theta = dk*z_max
         bcc   = cos(2.0d0*theta)
         bcs   = sin(2.0d0*theta)
         cc    = cos(theta)
         cs    = sin(theta)
         a  = cmplx(bcc,-bcs,kind=dp)
         b  = (dk**0.5d0)*cmplx(cc,-cs,kind=dp)
         c  = cmplx(0.0d0,dk,kind=dp)
      else
         a = (0.0d0,0.0d0)
         b = (0.0d0,0.0d0)
         c = cmplx(-dk,0.0d0,kind=dp)
      endif
      return
   end subroutine compute_wave_terms

end module scatsub_basis
