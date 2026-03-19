module multiscat_core
   use, intrinsic :: iso_fortran_env, only: real64
   implicit none
   private

   integer, parameter :: dp = real64

   integer, parameter :: NZFIXED_MAX = 550
   integer, parameter :: NVFCFIXED_MAX = 4096
   integer, parameter, public :: nmax = 1024
   integer, parameter :: mmax = 550
   integer, parameter :: n_fourier_components_x = 4096

   public :: OptimizationData
   public :: ScatteringData
   public :: PotentialData
   public :: ChannelBasisData
   public :: ScatteringConditionResult
   public :: OutputData
   public :: calculate_output_data

   type :: OptimizationData
      integer :: output_mode = 0
      integer :: gmres_preconditioner_flag = 0
      integer :: convergence_significant_figures = 2
      real(dp) :: max_closed_channel_energy = 0.0_dp
      integer :: max_channel_index = 0
   end type OptimizationData

   type :: ScatteringData
      real(dp) :: helium_mass = 0.0_dp
      real(dp) :: incident_energy_mev = 0.0_dp
      real(dp) :: theta_degrees = 0.0_dp
      real(dp) :: phi_degrees = 0.0_dp
   end type ScatteringData

   type :: PotentialData
      integer :: z_point_count = 0
      integer :: fourier_component_count = 0
      integer :: specular_component_index = 1
      integer :: fourier_grid_x_count = 0
      integer :: fourier_grid_y_count = 0
      real(dp) :: unit_cell_ax = 0.0_dp
      real(dp) :: unit_cell_ay = 0.0_dp
      real(dp) :: unit_cell_bx = 0.0_dp
      real(dp) :: unit_cell_by = 0.0_dp
      real(dp) :: z_min = 0.0_dp
      real(dp) :: z_max = 0.0_dp
      integer, allocatable :: fourier_indices_x(:)
      integer, allocatable :: fourier_indices_y(:)
      complex(dp), allocatable :: fixed_fourier_values(:,:)
   end type PotentialData

   type :: ChannelBasisData
      integer :: channel_count = 0
      integer :: specular_channel_index = 0
      integer :: channel_index_x(nmax) = 0
      integer :: channel_index_y(nmax) = 0
      real(dp) :: channel_energy_z(nmax) = 0.0_dp
   end type ChannelBasisData

   type :: ScatteringConditionResult
      real(dp) :: incident_energy_mev = 0.0_dp
      real(dp) :: theta_degrees = 0.0_dp
      real(dp) :: phi_degrees = 0.0_dp
      integer :: channel_count = 0
      integer :: specular_channel_index = 0
      integer, allocatable :: channel_ix(:)
      integer, allocatable :: channel_iy(:)
      real(dp), allocatable :: channel_d(:)
      real(dp), allocatable :: channel_intensity(:)
      real(dp) :: specular_intensity = 0.0_dp
      real(dp) :: open_channel_intensity_sum = 0.0_dp
      logical :: gmres_failed = .false.
   end type ScatteringConditionResult

   type :: OutputData
      type(ScatteringConditionResult) :: condition
   end type OutputData

contains

   function calculate_output_data(optimization_data, scatt_conditions_data, potential_data) result(output_data)
      implicit none
      type(OptimizationData), intent(in) :: optimization_data
      type(ScatteringData), intent(in) :: scatt_conditions_data
      type(PotentialData), intent(in) :: potential_data
      type(OutputData) :: output_data

      integer, parameter :: lmax = 901

      integer :: ipc, nsf, max_channel_index
      integer :: n_fourier_components, n_z_points
      integer :: specular_fourier_component_index
      integer :: m, i, j, ifail, alloc_status

      real(dp) :: eps, max_closed_channel_energy
      real(dp) :: ax, ay, bx, by, ei, theta, phi
      real(dp) :: hemass, rmlmda
      real(dp) :: z_min, z_max
      real(dp) :: open_sum

      complex(dp), allocatable :: x(:,:), y(:,:)
      complex(dp), allocatable :: xx(:,:)
      complex(dp), allocatable :: a(:), b(:), c(:), s(:)

      type(ChannelBasisData) :: basis_data

      real(dp), allocatable :: p(:), w(:), z(:)
      real(dp), allocatable :: e(:), f(:,:), t(:,:)

      allocate(x(mmax,nmax), y(mmax,nmax), stat=alloc_status)
      if (alloc_status /= 0) error stop 'ERROR: allocation failure (x, y).'
      allocate(xx(nmax*mmax,lmax), stat=alloc_status)
      if (alloc_status /= 0) error stop 'ERROR: allocation failure (xx).'
      allocate(a(nmax), b(nmax), c(nmax), s(nmax), stat=alloc_status)
      if (alloc_status /= 0) error stop 'ERROR: allocation failure (a, b, c, s).'

      allocate(p(nmax), w(mmax), z(mmax), stat=alloc_status)
      if (alloc_status /= 0) error stop 'ERROR: allocation failure (p, w, z).'
      allocate(e(mmax), f(mmax,nmax), t(mmax,mmax), stat=alloc_status)
      if (alloc_status /= 0) error stop 'ERROR: allocation failure (e, f, t).'

      ipc = optimization_data%gmres_preconditioner_flag
      nsf = optimization_data%convergence_significant_figures
      eps = 0.5_dp*(10.0_dp**(-nsf))
      max_closed_channel_energy = optimization_data%max_closed_channel_energy
      max_channel_index = optimization_data%max_channel_index

      n_fourier_components = potential_data%fourier_component_count
      n_z_points = potential_data%z_point_count
      specular_fourier_component_index = potential_data%specular_component_index
      ax = potential_data%unit_cell_ax
      ay = potential_data%unit_cell_ay
      bx = potential_data%unit_cell_bx
      by = potential_data%unit_cell_by
      z_min = potential_data%z_min
      z_max = potential_data%z_max

      ei = scatt_conditions_data%incident_energy_mev
      theta = scatt_conditions_data%theta_degrees
      phi = scatt_conditions_data%phi_degrees
      hemass = scatt_conditions_data%helium_mass
      rmlmda = 2.0_dp * hemass / 4.18020_dp

      m = n_z_points
      if (m > mmax) error stop 'ERROR: m too big!'

      call build_lobatto_t_matrix(z_min,z_max,m,w,z,t)

      call get_momentum_basis( &
         basis_data%channel_count, basis_data%specular_channel_index, &
         basis_data%channel_index_x, basis_data%channel_index_y, &
         basis_data%channel_energy_z, max_closed_channel_energy, &
         max_channel_index, ax, ay, bx, by, ei, theta, phi, rmlmda &
         )
      if (basis_data%channel_count > nmax) error stop 'ERROR: n too big!'

      do i = 1,basis_data%channel_count
         call compute_wave_terms (basis_data%channel_energy_z(i),a(i),b(i),c(i),z_max)
         b(i) = b(i)/w(m)
         c(i) = c(i)/(w(m)**2)
      end do

      call build_preconditioner ( &
         m, basis_data%channel_count, potential_data%fixed_fourier_values, n_fourier_components, &
         specular_fourier_component_index, basis_data%channel_energy_z, e, f, t &
         )
      ifail = 0
      call solve_gmres_system ( &
         x, xx, y, m, basis_data%channel_index_x, basis_data%channel_index_y, &
         basis_data%channel_count, basis_data%specular_channel_index, &
         potential_data%fixed_fourier_values, &
         potential_data%fourier_indices_x, potential_data%fourier_indices_y, n_fourier_components, &
         a, b, c, basis_data%channel_energy_z, e, f, p, s, t, eps, ipc, ifail &
         )

      if (ifail == 1) then
         p = -1.0_dp
      end if

      output_data%condition%incident_energy_mev = ei
      output_data%condition%theta_degrees = theta
      output_data%condition%phi_degrees = phi
      output_data%condition%channel_count = basis_data%channel_count
      output_data%condition%specular_channel_index = basis_data%specular_channel_index
      output_data%condition%specular_intensity = p(basis_data%specular_channel_index)
      output_data%condition%gmres_failed = (ifail == 1)

      allocate(output_data%condition%channel_ix(basis_data%channel_count), stat=alloc_status)
      if (alloc_status /= 0) error stop 'ERROR: allocation failure (channel_ix).'
      allocate(output_data%condition%channel_iy(basis_data%channel_count), stat=alloc_status)
      if (alloc_status /= 0) error stop 'ERROR: allocation failure (channel_iy).'
      allocate(output_data%condition%channel_d(basis_data%channel_count), stat=alloc_status)
      if (alloc_status /= 0) error stop 'ERROR: allocation failure (channel_d).'
      allocate(output_data%condition%channel_intensity(basis_data%channel_count), stat=alloc_status)
      if (alloc_status /= 0) error stop 'ERROR: allocation failure (channel_intensity).'
      output_data%condition%channel_ix(1:basis_data%channel_count) = &
         basis_data%channel_index_x(1:basis_data%channel_count)
      output_data%condition%channel_iy(1:basis_data%channel_count) = &
         basis_data%channel_index_y(1:basis_data%channel_count)
      output_data%condition%channel_d(1:basis_data%channel_count) = &
         basis_data%channel_energy_z(1:basis_data%channel_count)
      output_data%condition%channel_intensity(1:basis_data%channel_count) = &
         p(1:basis_data%channel_count)

      open_sum = 0.0_dp
      do j = 1,basis_data%channel_count
         if (basis_data%channel_energy_z(j) < 0.0_dp) open_sum = open_sum + p(j)
      end do
      output_data%condition%open_channel_intensity_sum = open_sum

      if (allocated(x)) deallocate(x)
      if (allocated(y)) deallocate(y)
      if (allocated(xx)) deallocate(xx)
      if (allocated(a)) deallocate(a)
      if (allocated(b)) deallocate(b)
      if (allocated(c)) deallocate(c)
      if (allocated(s)) deallocate(s)
      if (allocated(p)) deallocate(p)
      if (allocated(w)) deallocate(w)
      if (allocated(z)) deallocate(z)
      if (allocated(e)) deallocate(e)
      if (allocated(f)) deallocate(f)
      if (allocated(t)) deallocate(t)
   end function calculate_output_data

end module multiscat_core
