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
   & wave_c, scattered_state_dense, parallel_kinetic_energy, gmres_info)
      implicit none
      type(OptimizationData), intent(in) :: optimization_data
      complex(dp), intent(in) :: potential_values(:,:,:)
      real(dp), intent(in) :: perpendicular_kinetic_difference(:,:)
      complex(dp), intent(in) :: wave_a(:), wave_b(:), wave_c(:)
      complex(dp), intent(out) :: scattered_state_dense(:,:,:)
      real(dp), intent(in) :: parallel_kinetic_energy(:,:)
      integer, intent(out) :: gmres_info
      real(dp) :: eps
      integer :: i, j, idx, alloc_status, n_z_points, channel_count
      complex(dp), allocatable :: scattered_state(:,:)

      eps = 0.5_dp*(10.0_dp**(-optimization_data%convergence_significant_figures))
      n_z_points = size(potential_values, 3)
      channel_count = size(perpendicular_kinetic_difference, 1) * size(perpendicular_kinetic_difference, 2)

      if (size(wave_a) /= channel_count .or. size(wave_b) /= channel_count .or. size(wave_c) /= channel_count) then
         error stop 'ERROR: inconsistent wave vector sizes for scattering solve.'
      end if

      allocate(scattered_state(n_z_points,channel_count), stat=alloc_status)
      if (alloc_status /= 0) error stop 'ERROR: allocation failure (scattered_state).'

      call run_preconditioned_gmres ( &
         n_z_points, &
         potential_values, size(potential_values, 1), size(potential_values, 2), &
         wave_b(1), wave_c, perpendicular_kinetic_difference, scattered_state, &
      & parallel_kinetic_energy, eps, optimization_data%gmres_preconditioner_flag, gmres_info &
         )

      idx = 0
      do i = 1,size(scattered_state_dense, 1)
         do j = 1,size(scattered_state_dense, 2)
            idx = idx + 1
            scattered_state_dense(i, j, :) = scattered_state(:, idx)
         end do
      end do

      if (allocated(scattered_state)) deallocate(scattered_state)
   end subroutine run_scattering_linear_step

end module multiscat_core
