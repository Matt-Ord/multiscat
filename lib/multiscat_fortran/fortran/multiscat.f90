module multiscat_core
  implicit none

  integer, parameter :: NZFIXED_MAX = 550
  integer, parameter :: NVFCFIXED_MAX = 4096
  integer, parameter :: nmax = 1024
  integer, parameter :: mmax = 550
  integer, parameter :: n_fourier_components_x = 4096

  type :: OptimizationData
    integer :: output_mode = 0
    integer :: gmres_preconditioner_flag = 0
    integer :: convergence_significant_figures = 2
    double precision :: max_closed_channel_energy = 0.0d0
    integer :: max_channel_index = 0
  end type OptimizationData

  type :: ScatteringData
    double precision :: helium_mass = 0.0d0
    double precision :: incident_energy_mev = 0.0d0
    double precision :: theta_degrees = 0.0d0
    double precision :: phi_degrees = 0.0d0
  end type ScatteringData

  type :: PotentialData
    integer :: z_point_count = 0
    integer :: fourier_component_count = 0
    integer :: specular_component_index = 1
    integer :: fourier_grid_x_count = 0
    integer :: fourier_grid_y_count = 0
    double precision :: unit_cell_ax = 0.0d0
    double precision :: unit_cell_ay = 0.0d0
    double precision :: unit_cell_bx = 0.0d0
    double precision :: unit_cell_by = 0.0d0
    double precision :: z_min = 0.0d0
    double precision :: z_max = 0.0d0
    integer, allocatable :: fourier_indices_x(:)
    integer, allocatable :: fourier_indices_y(:)
    complex*16, allocatable :: fixed_fourier_values(:,:)
  end type PotentialData

  type :: ChannelBasisData
    integer :: channel_count = 0
    integer :: specular_channel_index = 0
    integer :: channel_index_x(nmax) = 0
    integer :: channel_index_y(nmax) = 0
    double precision :: channel_energy_z(nmax) = 0.0d0
  end type ChannelBasisData

  type :: ScatteringConditionResult
    double precision :: incident_energy_mev = 0.0d0
    double precision :: theta_degrees = 0.0d0
    double precision :: phi_degrees = 0.0d0
    integer :: channel_count = 0
    integer :: specular_channel_index = 0
    integer, allocatable :: channel_ix(:)
    integer, allocatable :: channel_iy(:)
    double precision, allocatable :: channel_d(:)
    double precision, allocatable :: channel_intensity(:)
    double precision :: specular_intensity = 0.0d0
    double precision :: open_channel_intensity_sum = 0.0d0
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

    integer, parameter :: lmax = 901

    integer :: ipc, nsf, max_channel_index
    integer :: n_fourier_components, n_z_points
    integer :: specular_fourier_component_index
    integer :: m, i, j, ifail

    double precision :: eps, max_closed_channel_energy
    double precision :: ax, ay, bx, by, ei, theta, phi, a0
    double precision :: hemass, rmlmda
    double precision :: z_min, z_max
    double precision :: open_sum

    complex*16, allocatable :: x(:,:), y(:,:)
    complex*16, allocatable :: xx(:,:)
    complex*16, allocatable :: a(:), b(:), c(:), s(:)

    type(ChannelBasisData) :: basis_data

    double precision, allocatable :: p(:), w(:), z(:)
    double precision, allocatable :: e(:), f(:,:), t(:,:)

    common /const/ hemass, rmlmda
    common /cells/ ax,ay,bx,by,ei,theta,phi,a0

    allocate(x(mmax,nmax), y(mmax,nmax))
    allocate(xx(nmax*mmax,lmax))
    allocate(a(nmax), b(nmax), c(nmax), s(nmax))

    allocate(p(nmax), w(mmax), z(mmax))
    allocate(e(mmax), f(mmax,nmax), t(mmax,mmax))

    ipc = optimization_data%gmres_preconditioner_flag
    nsf = optimization_data%convergence_significant_figures
    eps = 0.5d0*(10.0d0**(-nsf))
    max_closed_channel_energy = optimization_data%max_closed_channel_energy
    max_channel_index = optimization_data%max_channel_index

    n_fourier_components = potential_data%fourier_component_count
    n_z_points = potential_data%z_point_count
    specular_fourier_component_index = potential_data%specular_component_index
    ax = potential_data%unit_cell_ax
    ay = potential_data%unit_cell_ay
    bx = potential_data%unit_cell_bx
    by = potential_data%unit_cell_by
    z_min = potential_data%z_min
    z_max = potential_data%z_max

    ei = scatt_conditions_data%incident_energy_mev
    theta = scatt_conditions_data%theta_degrees
    phi = scatt_conditions_data%phi_degrees

    m = n_z_points
    if (m.gt.mmax) stop 'ERROR: m too big!'

    call build_lobatto_t_matrix(z_min,z_max,m,w,z,t)

    call get_momentum_basis( &
      basis_data%channel_count, basis_data%specular_channel_index, &
      basis_data%channel_index_x, basis_data%channel_index_y, &
      basis_data%channel_energy_z, max_closed_channel_energy, &
      max_channel_index &
    )
    if (basis_data%channel_count.gt.nmax) stop 'ERROR: n too big!'

    do i = 1,basis_data%channel_count
      call compute_wave_terms (basis_data%channel_energy_z(i),a(i),b(i),c(i),z_max)
      b(i) = b(i)/w(m)
      c(i) = c(i)/(w(m)**2)
    end do

    call build_preconditioner ( &
      m, basis_data%channel_count, potential_data%fixed_fourier_values, n_fourier_components, &
      specular_fourier_component_index, basis_data%channel_energy_z, e, f, t &
    )
    ifail = 0
    call solve_gmres_system ( &
      x, xx, y, m, basis_data%channel_index_x, basis_data%channel_index_y, &
      basis_data%channel_count, basis_data%specular_channel_index, &
      potential_data%fixed_fourier_values, &
      potential_data%fourier_indices_x, potential_data%fourier_indices_y, n_fourier_components, &
      a, b, c, basis_data%channel_energy_z, e, f, p, s, t, eps, ipc, ifail &
    )

    if (ifail.eq.1) then
      p = -1
    end if

    output_data%condition%incident_energy_mev = ei
    output_data%condition%theta_degrees = theta
    output_data%condition%phi_degrees = phi
    output_data%condition%channel_count = basis_data%channel_count
    output_data%condition%specular_channel_index = basis_data%specular_channel_index
    output_data%condition%specular_intensity = p(basis_data%specular_channel_index)
    output_data%condition%gmres_failed = (ifail.eq.1)

    allocate(output_data%condition%channel_ix(basis_data%channel_count))
    allocate(output_data%condition%channel_iy(basis_data%channel_count))
    allocate(output_data%condition%channel_d(basis_data%channel_count))
    allocate(output_data%condition%channel_intensity(basis_data%channel_count))
    output_data%condition%channel_ix(1:basis_data%channel_count) = &
      basis_data%channel_index_x(1:basis_data%channel_count)
    output_data%condition%channel_iy(1:basis_data%channel_count) = &
      basis_data%channel_index_y(1:basis_data%channel_count)
    output_data%condition%channel_d(1:basis_data%channel_count) = &
      basis_data%channel_energy_z(1:basis_data%channel_count)
    output_data%condition%channel_intensity(1:basis_data%channel_count) = &
      p(1:basis_data%channel_count)

    open_sum = 0.0d0
    do j = 1,basis_data%channel_count
      if (basis_data%channel_energy_z(j).lt.0.0d0) open_sum = open_sum + p(j)
    end do
    output_data%condition%open_channel_intensity_sum = open_sum

    deallocate(x, y, xx, a, b, c, s)
    deallocate(p, w, z, e, f, t)
  end function calculate_output_data

end module multiscat_core
