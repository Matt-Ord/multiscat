module multiscat_core
   use, intrinsic :: iso_fortran_env, only: real64
   use scatsub_basis, only: UnitVectors, IncidentWaveData, ChannelBasisData, &
      get_perpendicular_kinetic_difference, &
      perpendicular_momentum_as_legacy_data, get_parallel_kinetic_energy, &
      get_lobatto_weights, compute_wave_terms
   implicit none
   private

   integer, parameter :: dp = real64

   public :: OptimizationData
   public :: IncidentWaveData
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

   function calculate_output_data(optimization_data, incident_wave_data, potential_data) result(output_data)
      implicit none
      type(OptimizationData), intent(in) :: optimization_data
      type(IncidentWaveData), intent(in) :: incident_wave_data
      type(PotentialData), intent(in) :: potential_data
      type(OutputData) :: output_data
      type(ChannelBasisData) :: basis_data

      integer :: n_z_points
      integer :: j, nx, ny, alloc_status

      real(dp) :: eps
      real(dp) :: open_sum

      complex(dp), allocatable :: wave_a(:), wave_b(:), wave_c(:)

      real(dp), allocatable :: channel_intensity(:)
      real(dp), allocatable :: lobatto_weights(:), lobatto_points(:)
      real(dp), allocatable :: parallel_kinetic_energy(:,:)
      real(dp), allocatable :: perpendicular_kinetic_difference(:,:)

      eps = 0.5_dp*(10.0_dp**(-optimization_data%convergence_significant_figures))
      n_z_points = potential_data%z_point_count

      allocate(parallel_kinetic_energy(n_z_points,n_z_points), stat=alloc_status)
      if (alloc_status /= 0) error stop 'ERROR: allocation failure (parallel_kinetic_energy).'
      call get_parallel_kinetic_energy( &
         potential_data%z_min, potential_data%z_max, n_z_points, parallel_kinetic_energy &
         )

      nx = potential_data%fourier_grid_x_count
      ny = potential_data%fourier_grid_y_count
      allocate(perpendicular_kinetic_difference(nx, ny), stat=alloc_status)
      if (alloc_status /= 0) error stop 'ERROR: allocation failure (perpendicular_kinetic_difference).'
      call get_perpendicular_kinetic_difference( &
         perpendicular_kinetic_difference, nx, ny, potential_data%unit_vectors, incident_wave_data &
         )

      basis_data = perpendicular_momentum_as_legacy_data(perpendicular_kinetic_difference)

      allocate(wave_a(basis_data%channel_count), wave_b(basis_data%channel_count), &
      & wave_c(basis_data%channel_count), stat=alloc_status)
      if (alloc_status /= 0) error stop 'ERROR: allocation failure (wave_a, wave_b, wave_c).'
      allocate(channel_intensity(basis_data%channel_count), stat=alloc_status)
      if (alloc_status /= 0) error stop 'ERROR: allocation failure (channel_intensity).'
      allocate(lobatto_weights(n_z_points + 1), lobatto_points(n_z_points + 1), stat=alloc_status)
      if (alloc_status /= 0) error stop 'ERROR: allocation failure (lobatto_weights, lobatto_points).'

      call get_bc_arrays( &
         potential_data, basis_data, n_z_points, &
         wave_a, wave_b, wave_c, lobatto_weights, lobatto_points &
         )

      call run_scattering_linear_step( &
         optimization_data, potential_data, basis_data, n_z_points, &
         wave_a, wave_b, wave_c, channel_intensity, parallel_kinetic_energy, eps &
         )

      output_data%condition%channel_count = basis_data%channel_count
      output_data%condition%specular_channel_index = basis_data%specular_channel_index
      output_data%condition%specular_intensity = channel_intensity(basis_data%specular_channel_index)
      output_data%condition%gmres_failed = (channel_intensity(basis_data%specular_channel_index) < 0.0_dp)

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
         channel_intensity(1:basis_data%channel_count)

      open_sum = 0.0_dp
      do j = 1,basis_data%channel_count
         if (basis_data%channel_energy_z(j) < 0.0_dp) open_sum = open_sum + channel_intensity(j)
      end do
      output_data%condition%open_channel_intensity_sum = open_sum

      if (allocated(wave_a)) deallocate(wave_a)
      if (allocated(wave_b)) deallocate(wave_b)
      if (allocated(wave_c)) deallocate(wave_c)
      if (allocated(channel_intensity)) deallocate(channel_intensity)
      if (allocated(perpendicular_kinetic_difference)) deallocate(perpendicular_kinetic_difference)
      if (allocated(lobatto_weights)) deallocate(lobatto_weights)
      if (allocated(lobatto_points)) deallocate(lobatto_points)
      if (allocated(parallel_kinetic_energy)) deallocate(parallel_kinetic_energy)
   end function calculate_output_data

   subroutine get_bc_arrays( &
   & potential_data, basis_data, n_z_points, wave_a, wave_b, wave_c, &
   & lobatto_weights, lobatto_points)
      implicit none
      type(PotentialData), intent(in) :: potential_data
      type(ChannelBasisData), intent(in) :: basis_data
      integer, intent(in) :: n_z_points
      complex(dp), intent(out) :: wave_a(:), wave_b(:), wave_c(:)
      real(dp), intent(inout) :: lobatto_weights(:), lobatto_points(:)

      integer :: i
      call get_lobatto_weights( &
         potential_data%z_min, potential_data%z_max, n_z_points + 1, lobatto_weights, lobatto_points &
         )

      do i = 1,basis_data%channel_count
         call compute_wave_terms( &
            basis_data%channel_energy_z(i), wave_a(i), wave_b(i), wave_c(i), potential_data%z_max &
            )
         wave_b(i) = wave_b(i)/lobatto_weights(n_z_points + 1)
         wave_c(i) = wave_c(i)/(lobatto_weights(n_z_points + 1)**2)
      end do
   end subroutine get_bc_arrays

   subroutine run_scattering_linear_step( &
   & optimization_data, potential_data, basis_data, n_z_points, wave_a, wave_b, &
   & wave_c, channel_intensity, parallel_kinetic_energy, eps)
      implicit none
      type(OptimizationData), intent(in) :: optimization_data
      type(PotentialData), intent(in) :: potential_data
      type(ChannelBasisData), intent(in) :: basis_data
      integer, intent(in) :: n_z_points
      complex(dp), intent(in) :: wave_a(:), wave_b(:), wave_c(:)
      real(dp), intent(out) :: channel_intensity(:)
      real(dp), intent(in) :: parallel_kinetic_energy(:,:)
      real(dp), intent(in) :: eps

      call run_preconditioned_gmres ( &
         n_z_points, basis_data%channel_index_x, basis_data%channel_index_y, &
         basis_data%channel_count, basis_data%specular_channel_index, &
         potential_data%fixed_fourier_values, &
         potential_data%fourier_indices_x, potential_data%fourier_indices_y, &
         potential_data%fourier_component_count, potential_data%specular_component_index, &
         wave_a, wave_b, wave_c, basis_data%channel_energy_z, channel_intensity, &
      & parallel_kinetic_energy, eps, optimization_data%gmres_preconditioner_flag &
         )
   end subroutine run_scattering_linear_step

end module multiscat_core
