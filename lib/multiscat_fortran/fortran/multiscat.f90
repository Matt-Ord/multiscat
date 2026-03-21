module multiscat_core
   use, intrinsic :: iso_fortran_env, only: real64
   use multiscat_gmres, only: run_preconditioned_gmres
   implicit none
   private

   integer, parameter :: dp = real64

   public :: OptimizationData
   public :: run_scattering_linear_step

   type :: OptimizationData
      integer :: output_mode = 0
      integer :: gmres_preconditioner_flag = 0
      integer :: convergence_significant_figures = 2
   end type OptimizationData

contains

   subroutine run_scattering_linear_step( &
   & optimization_data, potential_values, perpendicular_kinetic_difference, wave_a, wave_b, &
   & wave_c, channel_intensity_dense, parallel_kinetic_energy)
      implicit none
      type(OptimizationData), intent(in) :: optimization_data
      complex(dp), intent(in) :: potential_values(:,:,:)
      real(dp), intent(in) :: perpendicular_kinetic_difference(:,:)
      complex(dp), intent(in) :: wave_a(:), wave_b(:), wave_c(:)
      real(dp), intent(out) :: channel_intensity_dense(:,:)
      real(dp), intent(in) :: parallel_kinetic_energy(:,:)
      real(dp) :: eps
      integer :: i, j, idx, alloc_status, n_z_points, channel_count
      real(dp), allocatable :: channel_intensity(:)

      eps = 0.5_dp*(10.0_dp**(-optimization_data%convergence_significant_figures))
      n_z_points = size(potential_values, 3)
      channel_count = size(perpendicular_kinetic_difference, 1) * size(perpendicular_kinetic_difference, 2)

      allocate(channel_intensity(channel_count), stat=alloc_status)
      if (alloc_status /= 0) error stop 'ERROR: allocation failure (channel_intensity).'

      call run_preconditioned_gmres ( &
         n_z_points, &
         potential_values, size(potential_values, 1), size(potential_values, 2), &
         wave_a, wave_b, wave_c, perpendicular_kinetic_difference, channel_intensity, &
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
