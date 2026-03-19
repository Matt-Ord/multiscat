module multiscat_core
  implicit none

  integer, parameter :: NZFIXED_MAX = 550
  integer, parameter :: NVFCFIXED_MAX = 4096
  integer, parameter :: nmax = 1024
  integer, parameter :: mmax = 550
  integer, parameter :: nfcx = 4096

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
    double precision :: zmin = 0.0d0
    double precision :: zmax = 0.0d0
    integer, allocatable :: fourier_indices_x(:)
    integer, allocatable :: fourier_indices_y(:)
    complex*16, allocatable :: fixed_fourier_values(:,:)
  end type PotentialData

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

    integer :: ipc, nsf, imax
    integer :: nfc, nzfixed, nfc00
    integer :: m, i, j, n, n00, ifail

    double precision :: eps, dmax
    double precision :: ax, ay, bx, by, ei, theta, phi, a0
    double precision :: hemass, rmlmda
    double precision :: zmin, zmax
    double precision :: open_sum

    ! --- CONVERTED TO ALLOCATABLE DYNAMIC ARRAYS ---
    complex*16, allocatable :: x(:,:), y(:,:), vfc(:,:)
    complex*16, allocatable :: xx(:,:)
    complex*16, allocatable :: a(:), b(:), c(:), s(:)
    complex*16, allocatable :: vfcfixed(:,:)

    integer, allocatable :: ix(:), iy(:), ivx(:), ivy(:)
    double precision, allocatable :: p(:), w(:), z(:)
    double precision, allocatable :: d(:), e(:), f(:,:), t(:,:)

    common /const/ hemass, rmlmda
    common /cells/ ax,ay,bx,by,ei,theta,phi,a0

    ! --- ALLOCATE MEMORY ON THE HEAP ---
    allocate(x(mmax,nmax), y(mmax,nmax), vfc(mmax,nfcx))
    allocate(xx(nmax*mmax, lmax))
    allocate(a(nmax), b(nmax), c(nmax), s(nmax))
    allocate(vfcfixed(NZFIXED_MAX, NVFCFIXED_MAX))

    allocate(ix(nmax), iy(nmax), ivx(nfcx), ivy(nfcx))
    allocate(p(nmax), w(mmax), z(mmax))
    allocate(d(nmax), e(mmax), f(mmax,nmax), t(mmax,mmax))

    ipc = optimization_data%gmres_preconditioner_flag
    nsf = optimization_data%convergence_significant_figures
    eps = 0.5d0*(10.0d0**(-nsf))
    dmax = optimization_data%max_closed_channel_energy
    imax = optimization_data%max_channel_index

    nfc = potential_data%fourier_component_count
    nzfixed = potential_data%z_point_count
    nfc00 = potential_data%specular_component_index
    ax = potential_data%unit_cell_ax
    ay = potential_data%unit_cell_ay
    bx = potential_data%unit_cell_bx
    by = potential_data%unit_cell_by
    zmin = potential_data%zmin
    zmax = potential_data%zmax

    ivx(1:nfc) = potential_data%fourier_indices_x(1:nfc)
    ivy(1:nfc) = potential_data%fourier_indices_y(1:nfc)
    vfcfixed(1:nzfixed,1:nfc) = potential_data%fixed_fourier_values(1:nzfixed,1:nfc)

    ei = scatt_conditions_data%incident_energy_mev
    theta = scatt_conditions_data%theta_degrees
    phi = scatt_conditions_data%phi_degrees

    m = nzfixed
    if (m.gt.mmax) stop 'ERROR: m too big!'

    call tshape (zmin,zmax,m,w,z,t)

    do i = 1,nfc
      do j = 1,m
        vfc(j,i)=vfcfixed(j,i)
      end do
    end do

    call basis(d,ix,iy,n,n00,dmax,imax)
    if (n.gt.nmax) stop 'ERROR: n too big!'

    do i = 1,n
      call waves (d(i),a(i),b(i),c(i),zmax)
      b(i) = b(i)/w(m)
      c(i) = c(i)/(w(m)**2)
    end do

    call precon (m,n,vfc,nfc,nfc00,d,e,f,t)
    ifail = 0
    call gmres  (x,xx,y,m,ix,iy,n,n00,vfc,ivx,ivy,nfc,a,b,c,d,e,f,p,s,t,eps,ipc,ifail)

    if (ifail.eq.1) then
      p = -1
    end if

    output_data%condition%incident_energy_mev = ei
    output_data%condition%theta_degrees = theta
    output_data%condition%phi_degrees = phi
    output_data%condition%channel_count = n
    output_data%condition%specular_channel_index = n00
    output_data%condition%specular_intensity = p(n00)
    output_data%condition%gmres_failed = (ifail.eq.1)

    allocate(output_data%condition%channel_ix(n))
    allocate(output_data%condition%channel_iy(n))
    allocate(output_data%condition%channel_d(n))
    allocate(output_data%condition%channel_intensity(n))
    output_data%condition%channel_ix(1:n) = ix(1:n)
    output_data%condition%channel_iy(1:n) = iy(1:n)
    output_data%condition%channel_d(1:n) = d(1:n)
    output_data%condition%channel_intensity(1:n) = p(1:n)

    open_sum = 0.0d0
    do j = 1,n
      if (d(j).lt.0.0d0) open_sum = open_sum + p(j)
    end do
    output_data%condition%open_channel_intensity_sum = open_sum

    ! --- FREE MEMORY BEFORE RETURNING ---
    deallocate(x, y, vfc, xx, a, b, c, s, vfcfixed)
    deallocate(ix, iy, ivx, ivy, p, w, z, d, e, f, t)

  end function calculate_output_data

end module multiscat_core