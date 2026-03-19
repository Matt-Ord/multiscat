module scatsub_basis
   use, intrinsic :: iso_fortran_env, only: real64
   implicit none
   private

   integer, parameter :: dp = real64
   integer, parameter :: mmax = 550

   public :: UnitVectors
   public :: ReciprocalVectors
   public :: IncidentWaveData
   public :: ChannelBasisData
   public :: build_reciprocal_vectors
   public :: get_perpendicular_kinetic_difference
   public :: perpendicular_momentum_as_legacy_data
   public :: build_lobatto_t_matrix
   public :: compute_wave_terms

   type :: UnitVectors
      real(dp) :: ax1 = 0.0_dp
      real(dp) :: ay1 = 0.0_dp
      real(dp) :: bx1 = 0.0_dp
      real(dp) :: by1 = 0.0_dp
   end type UnitVectors

   type :: ReciprocalVectors
      real(dp) :: gax = 0.0_dp
      real(dp) :: gay = 0.0_dp
      real(dp) :: gbx = 0.0_dp
      real(dp) :: gby = 0.0_dp
   end type ReciprocalVectors

   type :: IncidentWaveData
      real(dp) :: incident_k(3) = 0.0_dp
   end type IncidentWaveData

   type :: ChannelBasisData
      integer :: channel_count = 0
      integer :: specular_channel_index = 0
      integer, allocatable :: channel_index_x(:)
      integer, allocatable :: channel_index_y(:)
      real(dp), allocatable :: channel_energy_z(:)
   end type ChannelBasisData

contains

   subroutine build_reciprocal_vectors(unit_vectors, reciprocal_vectors)
      implicit none
      real(dp), parameter :: pi = 3.141592653589793_dp
      type(UnitVectors), intent(in) :: unit_vectors
      type(ReciprocalVectors), intent(out) :: reciprocal_vectors
      real(dp) :: auc, recunit

      auc = abs(unit_vectors%ax1*unit_vectors%by1-unit_vectors%ay1*unit_vectors%bx1)
      if (auc .le. 0.0d0) error stop 'ERROR: unit cell area must be positive.'
      recunit = 2*pi/auc
      reciprocal_vectors%gax =  unit_vectors%by1*recunit
      reciprocal_vectors%gay = -unit_vectors%bx1*recunit
      reciprocal_vectors%gbx = -unit_vectors%ay1*recunit
      reciprocal_vectors%gby =  unit_vectors%ax1*recunit
   end subroutine build_reciprocal_vectors

   subroutine get_perpendicular_kinetic_difference( &
   & perpendicular_kinetic_difference, nx, ny, unit_vectors, incident_wave_data)
      implicit none
      real(dp), intent(out) :: perpendicular_kinetic_difference(nx, ny)
      integer, intent(in) :: nx, ny
      type(UnitVectors), intent(in) :: unit_vectors
      type(IncidentWaveData), intent(in) :: incident_wave_data

      real(dp) :: pkx, pky, incident_k2
      real(dp) :: gx, gy
      integer :: i, j, i0, j0, igx, igy
      type(ReciprocalVectors) :: reciprocal_vectors

      call build_reciprocal_vectors(unit_vectors, reciprocal_vectors)

      pkx = incident_wave_data%incident_k(1)
      pky = incident_wave_data%incident_k(2)
      incident_k2 = sum(incident_wave_data%incident_k**2)

      do i = 1, nx
         i0 = i - 1
         igx = i0
         if (i0 .gt. ((nx - 1) / 2)) igx = i0 - nx
         do j = 1, ny
            j0 = j - 1
            igy = j0
            if (j0 .gt. ((ny - 1) / 2)) igy = j0 - ny
            gx = reciprocal_vectors%gax*igx + reciprocal_vectors%gbx*igy
            gy = reciprocal_vectors%gay*igx + reciprocal_vectors%gby*igy
            perpendicular_kinetic_difference(i, j) = (pkx+gx)**2 + (pky+gy)**2 - incident_k2
         end do
      end do
   end subroutine get_perpendicular_kinetic_difference

   function perpendicular_momentum_as_legacy_data(perpendicular_momentum) result(basis_data)
      implicit none
      real(dp), intent(in) :: perpendicular_momentum(:,:)
      type(ChannelBasisData) :: basis_data

      integer :: nx, ny, i, j, idx, alloc_status

      nx = size(perpendicular_momentum, 1)
      ny = size(perpendicular_momentum, 2)
      basis_data%channel_count = nx*ny
      basis_data%specular_channel_index = 0

      allocate(basis_data%channel_index_x(basis_data%channel_count), stat=alloc_status)
      if (alloc_status /= 0) error stop 'ERROR: allocation failure (basis_data%channel_index_x).'
      allocate(basis_data%channel_index_y(basis_data%channel_count), stat=alloc_status)
      if (alloc_status /= 0) error stop 'ERROR: allocation failure (basis_data%channel_index_y).'
      allocate(basis_data%channel_energy_z(basis_data%channel_count), stat=alloc_status)
      if (alloc_status /= 0) error stop 'ERROR: allocation failure (basis_data%channel_energy_z).'

      idx = 0
      do i = 1, nx
         do j = 1, ny
            idx = idx + 1
            basis_data%channel_index_x(idx) = fft_mode_index(i - 1, nx)
            basis_data%channel_index_y(idx) = fft_mode_index(j - 1, ny)
            basis_data%channel_energy_z(idx) = perpendicular_momentum(i, j)
            if (basis_data%channel_index_x(idx) .eq. 0 .and. basis_data%channel_index_y(idx) .eq. 0) then
               basis_data%specular_channel_index = idx
            end if
         end do
      end do
      if (basis_data%specular_channel_index .eq. 0) error stop 'ERROR: specular channel not found.'
   end function perpendicular_momentum_as_legacy_data

   pure integer function fft_mode_index(i0, n) result(ig)
      implicit none
      integer, intent(in) :: i0, n

      ig = i0
      if (i0 .gt. ((n - 1) / 2)) ig = i0 - n
   end function fft_mode_index
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
