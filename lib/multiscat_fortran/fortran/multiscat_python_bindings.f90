subroutine run_multiscat_fortran( &
   helium_mass, &
   incident_kx, &
   incident_ky, &
   incident_kz, &
   gmres_preconditioner_flag, &
   convergence_significant_figures, &
   nkx, &
   nky, &
   nz, &
   unit_cell_ax, &
   unit_cell_ay, &
   unit_cell_bx, &
   unit_cell_by, &
   zmin, &
   zmax, &
   potential_values, &
   channel_intensity_dense, &
   gmres_failed, &
   ierr &
   )
   use, intrinsic :: iso_fortran_env, only: real64
   use multiscat_core, only: OptimizationData, IncidentWaveData, &
      PotentialData, OutputData, calculate_output_data
   implicit none

   integer, parameter :: dp = real64

   real(dp), intent(in) :: helium_mass
   real(dp), intent(in) :: incident_kx
   real(dp), intent(in) :: incident_ky
   real(dp), intent(in) :: incident_kz
   integer, intent(in) :: gmres_preconditioner_flag
   integer, intent(in) :: convergence_significant_figures
   integer, intent(in) :: nkx
   integer, intent(in) :: nky
   integer, intent(in) :: nz
   real(dp), intent(in) :: unit_cell_ax
   real(dp), intent(in) :: unit_cell_ay
   real(dp), intent(in) :: unit_cell_bx
   real(dp), intent(in) :: unit_cell_by
   real(dp), intent(in) :: zmin
   real(dp), intent(in) :: zmax
   complex(dp), intent(in) :: potential_values(nz, nkx * nky)
   real(dp), intent(out) :: channel_intensity_dense(nkx, nky)
   integer, intent(out) :: gmres_failed
   integer, intent(out) :: ierr

   real(dp), parameter :: hbarsq = 4.18020_dp
   real(dp) :: rmlmda
   integer :: nfc
   integer :: i, j, idx, alloc_status

   type(OptimizationData) :: optimization_data
   type(IncidentWaveData) :: incident_wave_data
   type(PotentialData) :: potential_data
   type(OutputData) :: output_data

   ierr = 0
   channel_intensity_dense = 0.0_dp
   gmres_failed = 0

   if (nkx .le. 0 .or. nky .le. 0 .or. nz .le. 0) then
      ierr = 1
      return
   end if

   nfc = nkx * nky
   optimization_data%output_mode = 0
   optimization_data%gmres_preconditioner_flag = gmres_preconditioner_flag
   optimization_data%convergence_significant_figures = convergence_significant_figures

   incident_wave_data%incident_k(1) = incident_kx
   incident_wave_data%incident_k(2) = incident_ky
   incident_wave_data%incident_k(3) = incident_kz

   potential_data%z_point_count = nz
   potential_data%fourier_component_count = nfc
   potential_data%fourier_grid_x_count = nkx
   potential_data%fourier_grid_y_count = nky
   potential_data%unit_vectors%ax1 = unit_cell_ax
   potential_data%unit_vectors%ay1 = unit_cell_ay
   potential_data%unit_vectors%bx1 = unit_cell_bx
   potential_data%unit_vectors%by1 = unit_cell_by
   potential_data%z_min = zmin
   potential_data%z_max = zmax

   allocate( &
      potential_data%fourier_indices_x(nfc), &
      potential_data%fourier_indices_y(nfc) &
      , stat=alloc_status)
   if (alloc_status /= 0) then
      ierr = 4
      return
   end if
   allocate(potential_data%fixed_fourier_values(nz, nfc), stat=alloc_status)
   if (alloc_status /= 0) then
      ierr = 4
      if (allocated(potential_data%fourier_indices_x)) deallocate(potential_data%fourier_indices_x)
      if (allocated(potential_data%fourier_indices_y)) deallocate(potential_data%fourier_indices_y)
      return
   end if

   idx = 0
   potential_data%specular_component_index = 1
   do i = 0, nkx - 1
      do j = 0, nky - 1
         idx = idx + 1

         potential_data%fourier_indices_x(idx) = i
         if (i .gt. ((nkx - 1) / 2)) then
            potential_data%fourier_indices_x(idx) = i - nkx
         end if

         potential_data%fourier_indices_y(idx) = j
         if (j .gt. ((nky - 1) / 2)) then
            potential_data%fourier_indices_y(idx) = j - nky
         end if

         if ( &
            potential_data%fourier_indices_x(idx) .eq. 0 .and. &
            potential_data%fourier_indices_y(idx) .eq. 0 &
            ) then
            potential_data%specular_component_index = idx
         end if
      end do
   end do

   rmlmda = 2.0_dp * helium_mass / hbarsq

   potential_data%fixed_fourier_values = potential_values * rmlmda
   output_data = calculate_output_data(optimization_data, incident_wave_data, potential_data)
   channel_intensity_dense(1:nkx, 1:nky) = output_data%condition%channel_intensity_dense(1:nkx, 1:nky)
   if (output_data%condition%gmres_failed) then
      gmres_failed = 1
   else
      gmres_failed = 0
   end if

   if (allocated(potential_data%fourier_indices_x)) deallocate(potential_data%fourier_indices_x)
   if (allocated(potential_data%fourier_indices_y)) deallocate(potential_data%fourier_indices_y)
   if (allocated(potential_data%fixed_fourier_values)) deallocate(potential_data%fixed_fourier_values)
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
