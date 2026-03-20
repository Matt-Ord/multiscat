module multiscat_core
   use, intrinsic :: iso_fortran_env, only: real64
   use scatsub_basis, only: UnitVectors, IncidentWaveData, ChannelBasisData, &
      get_perpendicular_kinetic_difference, &
      perpendicular_momentum_as_legacy_data, get_parallel_kinetic_energy, &
      get_abc_arrays
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
      real(dp), allocatable :: channel_intensity_dense(:,:)
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
      integer :: i, j, idx, nx, ny, alloc_status

      real(dp) :: open_sum

      complex(dp), allocatable :: wave_a(:), wave_b(:), wave_c(:)

      real(dp), allocatable :: channel_intensity_dense(:,:)
      real(dp), allocatable :: channel_intensity(:)
      real(dp), allocatable :: parallel_kinetic_energy(:,:)
      real(dp), allocatable :: perpendicular_kinetic_difference(:,:)

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

      allocate(wave_a(nx*ny), wave_b(nx*ny), wave_c(nx*ny), stat=alloc_status)
      if (alloc_status /= 0) error stop 'ERROR: allocation failure (wave_a, wave_b, wave_c).'
      allocate(channel_intensity_dense(nx, ny), stat=alloc_status)
      if (alloc_status /= 0) error stop 'ERROR: allocation failure (channel_intensity_dense).'
      call get_abc_arrays( &
         potential_data%z_min, potential_data%z_max, perpendicular_kinetic_difference, n_z_points, &
         wave_a, wave_b, wave_c &
         )

      call run_scattering_linear_step( &
         optimization_data, potential_data, perpendicular_kinetic_difference, n_z_points, &
         wave_a, wave_b, wave_c, channel_intensity_dense, parallel_kinetic_energy &
         )

      basis_data = perpendicular_momentum_as_legacy_data(perpendicular_kinetic_difference)
      allocate(channel_intensity(basis_data%channel_count), stat=alloc_status)
      if (alloc_status /= 0) error stop 'ERROR: allocation failure (channel_intensity).'

      idx = 0
      do i = 1,nx
         do j = 1,ny
            idx = idx + 1
            channel_intensity(idx) = channel_intensity_dense(i, j)
         end do
      end do

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
      allocate(output_data%condition%channel_intensity_dense(nx, ny), stat=alloc_status)
      if (alloc_status /= 0) error stop 'ERROR: allocation failure (channel_intensity_dense).'
      output_data%condition%channel_ix(1:basis_data%channel_count) = &
         basis_data%channel_index_x(1:basis_data%channel_count)
      output_data%condition%channel_iy(1:basis_data%channel_count) = &
         basis_data%channel_index_y(1:basis_data%channel_count)
      output_data%condition%channel_d(1:basis_data%channel_count) = &
         basis_data%channel_energy_z(1:basis_data%channel_count)
      output_data%condition%channel_intensity(1:basis_data%channel_count) = &
         channel_intensity(1:basis_data%channel_count)
      output_data%condition%channel_intensity_dense(1:nx, 1:ny) = channel_intensity_dense(1:nx, 1:ny)

      open_sum = 0.0_dp
      do j = 1,basis_data%channel_count
         if (basis_data%channel_energy_z(j) < 0.0_dp) open_sum = open_sum + channel_intensity(j)
      end do
      output_data%condition%open_channel_intensity_sum = open_sum

      if (allocated(wave_a)) deallocate(wave_a)
      if (allocated(wave_b)) deallocate(wave_b)
      if (allocated(wave_c)) deallocate(wave_c)
      if (allocated(channel_intensity)) deallocate(channel_intensity)
      if (allocated(channel_intensity_dense)) deallocate(channel_intensity_dense)
      if (allocated(perpendicular_kinetic_difference)) deallocate(perpendicular_kinetic_difference)
      if (allocated(parallel_kinetic_energy)) deallocate(parallel_kinetic_energy)
   end function calculate_output_data

   subroutine run_scattering_linear_step( &
   & optimization_data, potential_data, perpendicular_kinetic_difference, n_z_points, wave_a, wave_b, &
   & wave_c, channel_intensity_dense, parallel_kinetic_energy)
      implicit none
      type(OptimizationData), intent(in) :: optimization_data
      type(PotentialData), intent(in) :: potential_data
      real(dp), intent(in) :: perpendicular_kinetic_difference(:,:)
      integer, intent(in) :: n_z_points
      complex(dp), intent(in) :: wave_a(:), wave_b(:), wave_c(:)
      real(dp), intent(out) :: channel_intensity_dense(:,:)
      real(dp), intent(in) :: parallel_kinetic_energy(:,:)
      type(ChannelBasisData) :: basis_data
      real(dp) :: eps
      integer :: i, j, idx, alloc_status
      real(dp), allocatable :: channel_intensity(:)

      eps = 0.5_dp*(10.0_dp**(-optimization_data%convergence_significant_figures))

      basis_data = perpendicular_momentum_as_legacy_data(perpendicular_kinetic_difference)

      allocate(channel_intensity(basis_data%channel_count), stat=alloc_status)
      if (alloc_status /= 0) error stop 'ERROR: allocation failure (channel_intensity).'

      call run_preconditioned_gmres ( &
         n_z_points, basis_data%channel_index_x, basis_data%channel_index_y, &
         basis_data%channel_count, basis_data%specular_channel_index, &
         potential_data%fixed_fourier_values, &
         potential_data%fourier_indices_x, potential_data%fourier_indices_y, &
         potential_data%fourier_component_count, potential_data%specular_component_index, &
         wave_a, wave_b, wave_c, basis_data%channel_energy_z, channel_intensity, &
      & parallel_kinetic_energy, eps, optimization_data%gmres_preconditioner_flag &
         )

      idx = 0
      do i = 1,size(channel_intensity_dense, 1)
         do j = 1,size(channel_intensity_dense, 2)
            idx = idx + 1
            channel_intensity_dense(i, j) = channel_intensity(idx)
         end do
      end do

      if (allocated(channel_intensity)) deallocate(channel_intensity)
   end subroutine run_scattering_linear_step

end module multiscat_core
