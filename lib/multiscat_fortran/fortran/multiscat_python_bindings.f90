subroutine run_multiscat_fortran( &
   helium_mass, &
   incident_energy_mev, &
   theta_degrees, &
   phi_degrees, &
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
   max_channels, &
   channel_count, &
   channel_ix, &
   channel_iy, &
   channel_d, &
   channel_intensity, &
   specular_intensity, &
   open_channel_intensity_sum, &
   gmres_failed, &
   ierr &
   )
   use, intrinsic :: iso_fortran_env, only: real64
   use multiscat_core, only: OptimizationData, ScatteringData, &
      PotentialData, OutputData, calculate_output_data
   implicit none

   integer, parameter :: dp = real64

   real(dp), intent(in) :: helium_mass
   real(dp), intent(in) :: incident_energy_mev
   real(dp), intent(in) :: theta_degrees
   real(dp), intent(in) :: phi_degrees
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
   integer, intent(in) :: max_channels

   integer, intent(out) :: channel_count
   integer, intent(out) :: channel_ix(max_channels)
   integer, intent(out) :: channel_iy(max_channels)
   real(dp), intent(out) :: channel_d(max_channels)
   real(dp), intent(out) :: channel_intensity(max_channels)
   real(dp), intent(out) :: specular_intensity
   real(dp), intent(out) :: open_channel_intensity_sum
   integer, intent(out) :: gmres_failed
   integer, intent(out) :: ierr

   real(dp), parameter :: hbarsq = 4.18020_dp
   real(dp) :: rmlmda
   integer :: nfc
   integer :: i, j, idx, alloc_status

   type(OptimizationData) :: optimization_data
   type(ScatteringData) :: scatt_conditions_data
   type(PotentialData) :: potential_data
   type(OutputData) :: output_data

   ierr = 0
   channel_count = 0
   channel_ix = 0
   channel_iy = 0
   channel_d = 0.0_dp
   channel_intensity = 0.0_dp
   specular_intensity = 0.0_dp
   open_channel_intensity_sum = 0.0_dp
   gmres_failed = 0

   if (nkx .le. 0 .or. nky .le. 0 .or. nz .le. 0) then
      ierr = 1
      return
   end if

   nfc = nkx * nky
   if (max_channels .lt. nfc) then
      ierr = 2
      return
   end if

   optimization_data%output_mode = 0
   optimization_data%gmres_preconditioner_flag = gmres_preconditioner_flag
   optimization_data%convergence_significant_figures = convergence_significant_figures

   scatt_conditions_data%helium_mass = helium_mass
   scatt_conditions_data%incident_energy_mev = incident_energy_mev
   scatt_conditions_data%theta_degrees = theta_degrees
   scatt_conditions_data%phi_degrees = phi_degrees

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

   rmlmda = 2.0_dp * scatt_conditions_data%helium_mass / hbarsq

   potential_data%fixed_fourier_values = potential_values * rmlmda
   output_data = calculate_output_data(optimization_data, scatt_conditions_data, potential_data)

   channel_count = output_data%condition%channel_count
   if (channel_count .gt. max_channels) then
      ierr = 3
      channel_count = 0
      if (allocated(potential_data%fourier_indices_x)) deallocate(potential_data%fourier_indices_x)
      if (allocated(potential_data%fourier_indices_y)) deallocate(potential_data%fourier_indices_y)
      if (allocated(potential_data%fixed_fourier_values)) deallocate(potential_data%fixed_fourier_values)
      return
   end if

   channel_ix(1:channel_count) = output_data%condition%channel_ix(1:channel_count)
   channel_iy(1:channel_count) = output_data%condition%channel_iy(1:channel_count)
   channel_d(1:channel_count) = output_data%condition%channel_d(1:channel_count)
   channel_intensity(1:channel_count) = &
      output_data%condition%channel_intensity(1:channel_count)
   specular_intensity = output_data%condition%specular_intensity
   open_channel_intensity_sum = output_data%condition%open_channel_intensity_sum
   if (output_data%condition%gmres_failed) then
      gmres_failed = 1
   else
      gmres_failed = 0
   end if

   if (allocated(potential_data%fourier_indices_x)) deallocate(potential_data%fourier_indices_x)
   if (allocated(potential_data%fourier_indices_y)) deallocate(potential_data%fourier_indices_y)
   if (allocated(potential_data%fixed_fourier_values)) deallocate(potential_data%fixed_fourier_values)
end subroutine run_multiscat_fortran
