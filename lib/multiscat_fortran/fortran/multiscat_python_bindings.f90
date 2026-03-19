subroutine run_multiscat_fortran( &
  helium_mass, &
  incident_energy_mev, &
  theta_degrees, &
  phi_degrees, &
  gmres_preconditioner_flag, &
  convergence_significant_figures, &
  max_closed_channel_energy, &
  max_channel_index, &
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
  use multiscat_core, only: OptimizationData, ScatteringData, &
    PotentialData, OutputData, calculate_output_data, nmax
  implicit none

  double precision, intent(in) :: helium_mass
  double precision, intent(in) :: incident_energy_mev
  double precision, intent(in) :: theta_degrees
  double precision, intent(in) :: phi_degrees
  integer, intent(in) :: gmres_preconditioner_flag
  integer, intent(in) :: convergence_significant_figures
  double precision, intent(in) :: max_closed_channel_energy
  integer, intent(in) :: max_channel_index
  integer, intent(in) :: nkx
  integer, intent(in) :: nky
  integer, intent(in) :: nz
  double precision, intent(in) :: unit_cell_ax
  double precision, intent(in) :: unit_cell_ay
  double precision, intent(in) :: unit_cell_bx
  double precision, intent(in) :: unit_cell_by
  double precision, intent(in) :: zmin
  double precision, intent(in) :: zmax
  complex*16, intent(in) :: potential_values(nz, nkx * nky)
  integer, intent(in) :: max_channels

  integer, intent(out) :: channel_count
  integer, intent(out) :: channel_ix(max_channels)
  integer, intent(out) :: channel_iy(max_channels)
  double precision, intent(out) :: channel_d(max_channels)
  double precision, intent(out) :: channel_intensity(max_channels)
  double precision, intent(out) :: specular_intensity
  double precision, intent(out) :: open_channel_intensity_sum
  integer, intent(out) :: gmres_failed
  integer, intent(out) :: ierr

  double precision, parameter :: hbarsq = 4.18020d0
  double precision :: hemass, rmlmda
  integer :: nfc
  integer :: i, j, idx

  type(OptimizationData) :: optimization_data
  type(ScatteringData) :: scatt_conditions_data
  type(PotentialData) :: potential_data
  type(OutputData) :: output_data

  common /const/ hemass, rmlmda

  ierr = 0
  channel_count = 0
  channel_ix = 0
  channel_iy = 0
  channel_d = 0.0d0
  channel_intensity = 0.0d0
  specular_intensity = 0.0d0
  open_channel_intensity_sum = 0.0d0
  gmres_failed = 0

  if (nkx .le. 0 .or. nky .le. 0 .or. nz .le. 0) then
    ierr = 1
    return
  end if

  nfc = nkx * nky

  if (max_channels .lt. nmax) then
    ierr = 2
    return
  end if

  optimization_data%output_mode = 0
  optimization_data%gmres_preconditioner_flag = gmres_preconditioner_flag
  optimization_data%convergence_significant_figures = convergence_significant_figures
  optimization_data%max_closed_channel_energy = max_closed_channel_energy
  optimization_data%max_channel_index = max_channel_index

  scatt_conditions_data%helium_mass = helium_mass
  scatt_conditions_data%incident_energy_mev = incident_energy_mev
  scatt_conditions_data%theta_degrees = theta_degrees
  scatt_conditions_data%phi_degrees = phi_degrees

  potential_data%z_point_count = nz
  potential_data%fourier_component_count = nfc
  potential_data%fourier_grid_x_count = nkx
  potential_data%fourier_grid_y_count = nky
  potential_data%unit_cell_ax = unit_cell_ax
  potential_data%unit_cell_ay = unit_cell_ay
  potential_data%unit_cell_bx = unit_cell_bx
  potential_data%unit_cell_by = unit_cell_by
  potential_data%zmin = zmin
  potential_data%zmax = zmax

  allocate( &
    potential_data%fourier_indices_x(nfc), &
    potential_data%fourier_indices_y(nfc) &
  )
  allocate(potential_data%fixed_fourier_values(nz, nfc))

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

  hemass = scatt_conditions_data%helium_mass
  rmlmda = 2.0d0 * hemass / hbarsq

  potential_data%fixed_fourier_values = potential_values * rmlmda
  output_data = calculate_output_data(optimization_data, scatt_conditions_data, potential_data)

  channel_count = output_data%condition%channel_count
  if (channel_count .gt. max_channels) then
    ierr = 3
    channel_count = 0
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
end subroutine run_multiscat_fortran
