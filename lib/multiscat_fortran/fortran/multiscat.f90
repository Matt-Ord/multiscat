module multiscat_core
   use, intrinsic :: iso_fortran_env, only: real64
   use scatsub_basis, only: UnitVectors, ScatteringData, ChannelBasisData, get_perpendicular_momentum, &
      perpendicular_momentum_as_legacy_data, build_lobatto_t_matrix, compute_wave_terms
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
   end type OptimizationData

   type :: PotentialData
      integer :: z_point_count = 0
      integer :: fourier_component_count = 0
      integer :: specular_component_index = 1
      integer :: fourier_grid_x_count = 0
      integer :: fourier_grid_y_count = 0
      type(UnitVectors) :: unit_vectors
      real(dp) :: z_min = 0.0_dp
      real(dp) :: z_max = 0.0_dp
      integer, allocatable :: fourier_indices_x(:)
      integer, allocatable :: fourier_indices_y(:)
      complex(dp), allocatable :: fixed_fourier_values(:,:)
   end type PotentialData

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

      integer :: n_z_points
      integer :: m, i, j, nx, ny, alloc_status

      real(dp) :: eps
      real(dp) :: open_sum

      complex(dp), allocatable :: a(:), b(:), c(:)

      type(ChannelBasisData) :: basis_data

      real(dp), allocatable :: p(:), w(:), z(:)
      real(dp), allocatable :: t(:,:)
      real(dp), allocatable :: perpendicular_momentum(:,:)

      allocate(w(mmax), z(mmax), stat=alloc_status)
      if (alloc_status /= 0) error stop 'ERROR: allocation failure (w, z).'
      allocate(t(mmax,mmax), stat=alloc_status)
      if (alloc_status /= 0) error stop 'ERROR: allocation failure (t).'

      eps = 0.5_dp*(10.0_dp**(-optimization_data%convergence_significant_figures))
      n_z_points = potential_data%z_point_count

      m = n_z_points
      if (m > mmax) error stop 'ERROR: m too big!'

      call build_lobatto_t_matrix(potential_data%z_min,potential_data%z_max,m,w,z,t)

      nx = potential_data%fourier_grid_x_count
      ny = potential_data%fourier_grid_y_count
      allocate(perpendicular_momentum(nx, ny), stat=alloc_status)
      if (alloc_status /= 0) error stop 'ERROR: allocation failure (perpendicular_momentum).'
      call get_perpendicular_momentum( &
         perpendicular_momentum, nx, ny, potential_data%unit_vectors, scatt_conditions_data &
         )

      basis_data = perpendicular_momentum_as_legacy_data(perpendicular_momentum)

      allocate(a(basis_data%channel_count), b(basis_data%channel_count), c(basis_data%channel_count), stat=alloc_status)
      if (alloc_status /= 0) error stop 'ERROR: allocation failure (a, b, c).'
      allocate(p(basis_data%channel_count), stat=alloc_status)
      if (alloc_status /= 0) error stop 'ERROR: allocation failure (p).'

      do i = 1,basis_data%channel_count
         call compute_wave_terms (basis_data%channel_energy_z(i),a(i),b(i),c(i),potential_data%z_max)
         b(i) = b(i)/w(m)
         c(i) = c(i)/(w(m)**2)
      end do

      call run_scattering_linear_step(optimization_data, potential_data, basis_data, m, a, b, c, p, t, eps)

      output_data%condition%incident_energy_mev = scatt_conditions_data%incident_energy_mev
      output_data%condition%theta_degrees = scatt_conditions_data%theta_degrees
      output_data%condition%phi_degrees = scatt_conditions_data%phi_degrees
      output_data%condition%channel_count = basis_data%channel_count
      output_data%condition%specular_channel_index = basis_data%specular_channel_index
      output_data%condition%specular_intensity = p(basis_data%specular_channel_index)
      output_data%condition%gmres_failed = (p(basis_data%specular_channel_index) < 0.0_dp)

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

      if (allocated(a)) deallocate(a)
      if (allocated(b)) deallocate(b)
      if (allocated(c)) deallocate(c)
      if (allocated(p)) deallocate(p)
      if (allocated(perpendicular_momentum)) deallocate(perpendicular_momentum)
      if (allocated(w)) deallocate(w)
      if (allocated(z)) deallocate(z)
      if (allocated(t)) deallocate(t)
   end function calculate_output_data

   subroutine run_scattering_linear_step(optimization_data, potential_data, basis_data, m, a, b, c, p, t, eps)
      implicit none
      type(OptimizationData), intent(in) :: optimization_data
      type(PotentialData), intent(in) :: potential_data
      type(ChannelBasisData), intent(in) :: basis_data
      integer, intent(in) :: m
      complex(dp), intent(in) :: a(:), b(:), c(:)
      real(dp), intent(out) :: p(:)
      real(dp), intent(in) :: t(:,:)
      real(dp), intent(in) :: eps

      call run_preconditioned_gmres ( &
         m, basis_data%channel_index_x, basis_data%channel_index_y, &
         basis_data%channel_count, basis_data%specular_channel_index, &
         potential_data%fixed_fourier_values, &
         potential_data%fourier_indices_x, potential_data%fourier_indices_y, &
         potential_data%fourier_component_count, potential_data%specular_component_index, &
         a, b, c, basis_data%channel_energy_z, p, t, eps, optimization_data%gmres_preconditioner_flag &
         )
   end subroutine run_scattering_linear_step

end module multiscat_core
