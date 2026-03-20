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
   public :: run_scattering_linear_step

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

contains

   subroutine run_scattering_linear_step( &
   & optimization_data, potential_data, perpendicular_kinetic_difference, wave_a, wave_b, &
   & wave_c, channel_intensity_dense, parallel_kinetic_energy)
      implicit none
      type(OptimizationData), intent(in) :: optimization_data
      type(PotentialData), intent(in) :: potential_data
      real(dp), intent(in) :: perpendicular_kinetic_difference(:,:)
      complex(dp), intent(in) :: wave_a(:), wave_b(:), wave_c(:)
      real(dp), intent(out) :: channel_intensity_dense(:,:)
      real(dp), intent(in) :: parallel_kinetic_energy(:,:)
      type(ChannelBasisData) :: basis_data
      real(dp) :: eps
      integer :: i, j, idx, alloc_status, n_z_points
      real(dp), allocatable :: channel_intensity(:)

      eps = 0.5_dp*(10.0_dp**(-optimization_data%convergence_significant_figures))
      n_z_points = potential_data%z_point_count

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
