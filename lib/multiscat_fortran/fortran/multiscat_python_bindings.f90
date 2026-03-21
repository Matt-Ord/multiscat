subroutine run_multiscat_fortran( &
   gmres_preconditioner_flag, &
   convergence_significant_figures, &
   nkx, &
   nky, &
   nz, &
   potential_values, &
   perpendicular_kinetic_difference, &
   wave_a, &
   wave_b, &
   wave_c, &
   parallel_kinetic_energy, &
   channel_intensity_dense, &
   ierr &
   )
   use, intrinsic :: iso_fortran_env, only: real64
   use multiscat_core, only: OptimizationData, run_scattering_linear_step
   implicit none

   integer, parameter :: dp = real64

   integer, intent(in) :: gmres_preconditioner_flag
   integer, intent(in) :: convergence_significant_figures
   integer, intent(in) :: nkx
   integer, intent(in) :: nky
   integer, intent(in) :: nz
   complex(dp), intent(in) :: potential_values(nkx, nky, nz)
   real(dp), intent(in) :: perpendicular_kinetic_difference(nkx, nky)
   complex(dp), intent(in) :: wave_a(nkx * nky)
   complex(dp), intent(in) :: wave_b(nkx * nky)
   complex(dp), intent(in) :: wave_c(nkx * nky)
   real(dp), intent(in) :: parallel_kinetic_energy(nz, nz)
   real(dp), intent(out) :: channel_intensity_dense(nkx, nky)
   integer, intent(out) :: ierr

   integer :: specular_i, specular_j

   type(OptimizationData) :: optimization_data

   ierr = 0

   if (nkx .le. 0 .or. nky .le. 0 .or. nz .le. 0) then
      ierr = 1
      return
   end if

   channel_intensity_dense = 0.0_dp

   optimization_data%output_mode = 0
   optimization_data%gmres_preconditioner_flag = gmres_preconditioner_flag
   optimization_data%convergence_significant_figures = convergence_significant_figures

   call run_scattering_linear_step( &
      optimization_data, potential_values, perpendicular_kinetic_difference, &
      wave_a, wave_b, wave_c, channel_intensity_dense, parallel_kinetic_energy &
      )

   specular_i = 1
   specular_j = 1
   if (channel_intensity_dense(specular_i, specular_j) < 0.0_dp) then
      ierr = 2
   end if
end subroutine run_multiscat_fortran

subroutine get_perpendicular_kinetic_difference( &
   incident_kx, &
   incident_ky, &
   incident_kz, &
   nx, &
   ny, &
   unit_cell_ax, &
   unit_cell_ay, &
   unit_cell_bx, &
   unit_cell_by, &
   perpendicular_kinetic_difference, &
   ierr &
   )
   use, intrinsic :: iso_fortran_env, only: real64
   use scatsub_basis, only: UnitVectors, IncidentWaveData, &
      get_perpendicular_kinetic_difference_core => &
      get_perpendicular_kinetic_difference
   implicit none

   integer, parameter :: dp = real64

   real(dp), intent(in) :: incident_kx
   real(dp), intent(in) :: incident_ky
   real(dp), intent(in) :: incident_kz
   integer, intent(in) :: nx
   integer, intent(in) :: ny
   real(dp), intent(in) :: unit_cell_ax
   real(dp), intent(in) :: unit_cell_ay
   real(dp), intent(in) :: unit_cell_bx
   real(dp), intent(in) :: unit_cell_by

   real(dp), intent(out) :: perpendicular_kinetic_difference(nx, ny)
   integer, intent(out) :: ierr

   type(UnitVectors) :: unit_vectors
   type(IncidentWaveData) :: incident_wave_data

   ierr = 0
   perpendicular_kinetic_difference = 0.0_dp

   if (nx .le. 0 .or. ny .le. 0) then
      ierr = 1
      return
   end if

   unit_vectors%ax1 = unit_cell_ax
   unit_vectors%ay1 = unit_cell_ay
   unit_vectors%bx1 = unit_cell_bx
   unit_vectors%by1 = unit_cell_by

   incident_wave_data%incident_k(1) = incident_kx
   incident_wave_data%incident_k(2) = incident_ky
   incident_wave_data%incident_k(3) = incident_kz

   call get_perpendicular_kinetic_difference_core( &
      perpendicular_kinetic_difference, &
      nx, &
      ny, &
      unit_vectors, &
      incident_wave_data &
      )
end subroutine get_perpendicular_kinetic_difference

subroutine get_parallel_kinetic_energy( &
   zmin, &
   zmax, &
   nz, &
   parallel_kinetic_energy, &
   ierr &
   )
   use, intrinsic :: iso_fortran_env, only: real64
   use scatsub_basis, only: get_parallel_kinetic_energy_core => get_parallel_kinetic_energy
   implicit none

   integer, parameter :: dp = real64

   real(dp), intent(in) :: zmin
   real(dp), intent(in) :: zmax
   integer, intent(in) :: nz
   real(dp), intent(out) :: parallel_kinetic_energy(nz, nz)
   integer, intent(out) :: ierr

   ierr = 0
   parallel_kinetic_energy = 0.0_dp

   if (nz .le. 0) then
      ierr = 1
      return
   end if

   call get_parallel_kinetic_energy_core(zmin, zmax, nz, parallel_kinetic_energy)
end subroutine get_parallel_kinetic_energy

subroutine get_lobatto_weights( &
   zmin, &
   zmax, &
   node_count, &
   w, &
   x, &
   ierr &
   )
   use, intrinsic :: iso_fortran_env, only: real64
   use scatsub_basis, only: get_lobatto_weights_core => get_lobatto_weights
   implicit none

   integer, parameter :: dp = real64

   real(dp), intent(in) :: zmin
   real(dp), intent(in) :: zmax
   integer, intent(in) :: node_count
   real(dp), intent(out) :: w(node_count)
   real(dp), intent(out) :: x(node_count)
   integer, intent(out) :: ierr

   ierr = 0
   w = 0.0_dp
   x = 0.0_dp

   if (node_count .le. 0) then
      ierr = 1
      return
   end if

   call get_lobatto_weights_core(zmin, zmax, node_count, w, x)
end subroutine get_lobatto_weights

subroutine get_abc_arrays( &
   zmin, &
   zmax, &
   nx, &
   ny, &
   perpendicular_kinetic_difference, &
   n_z_points, &
   wave_a, &
   wave_b, &
   wave_c, &
   ierr &
   )
   use, intrinsic :: iso_fortran_env, only: real64
   use scatsub_basis, only: get_abc_arrays_core => get_abc_arrays
   implicit none

   integer, parameter :: dp = real64

   real(dp), intent(in) :: zmin
   real(dp), intent(in) :: zmax
   integer, intent(in) :: nx
   integer, intent(in) :: ny
   real(dp), intent(in) :: perpendicular_kinetic_difference(nx, ny)
   integer, intent(in) :: n_z_points
   complex(dp), intent(out) :: wave_a(nx*ny)
   complex(dp), intent(out) :: wave_b(nx*ny)
   complex(dp), intent(out) :: wave_c(nx*ny)
   integer, intent(out) :: ierr

   ierr = 0
   wave_a = (0.0_dp, 0.0_dp)
   wave_b = (0.0_dp, 0.0_dp)
   wave_c = (0.0_dp, 0.0_dp)

   if (nx .le. 0 .or. ny .le. 0 .or. n_z_points .le. 0) then
      ierr = 1
      return
   end if

   call get_abc_arrays_core( &
      zmin, zmax, perpendicular_kinetic_difference, n_z_points, wave_a, wave_b, wave_c &
      )
end subroutine get_abc_arrays
