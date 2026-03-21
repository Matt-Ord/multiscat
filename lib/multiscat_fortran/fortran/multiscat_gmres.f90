module multiscat_gmres
   use, intrinsic :: iso_fortran_env, only: real64
   implicit none
   private

   integer, parameter :: dp = real64
   integer, parameter :: l = 2000

   type :: GmresOptions
      real(dp) :: rtol = 1.0d-5
      integer :: maxiter = l
      integer :: preconditioner_flag = 0
      integer :: convergence_window = 3
   end type GmresOptions

   type :: GmresResult
      integer :: info = 0
      integer :: iterations = 0
   end type GmresResult

   type :: ChannelIdxData
      integer, allocatable :: index_x(:)
      integer, allocatable :: index_y(:)
   end type ChannelIdxData

   type :: ScatteringOperator
      integer :: n_z_points = 0
      integer :: nkx = 0
      integer :: nky = 0
      type(ChannelIdxData), pointer :: channel_idx => null()
      complex(dp), pointer :: potential_values(:,:,:) => null()
      complex(dp), pointer :: wave_c(:) => null()
      real(dp), pointer :: perpendicular_kinetic_difference(:,:) => null()
   end type ScatteringOperator

   public :: ChannelIdxData
   public :: GmresOptions
   public :: GmresResult
   public :: ScatteringOperator
   public :: get_channel_index
   public :: get_scattered_intensity
   public :: solve_scattering_gmres
   public :: run_preconditioned_gmres
   public :: debug_build_preconditioner
   public :: debug_apply_upper_block
   public :: debug_solve_lower_block

contains

   pure integer function fft_mode_index(i0, n) result(ig)
      implicit none
      integer, intent(in) :: i0, n

      ig = i0
      if (i0 .gt. ((n - 1) / 2)) ig = i0 - n
   end function fft_mode_index

   subroutine get_channel_index(perpendicular_momentum, channel_idx)
      implicit none
      real(dp), intent(in) :: perpendicular_momentum(:,:)
      type(ChannelIdxData), intent(out) :: channel_idx

      integer :: nx, ny, i, j, idx, channel_count, alloc_status

      nx = size(perpendicular_momentum, 1)
      ny = size(perpendicular_momentum, 2)
      channel_count = nx*ny

      allocate(channel_idx%index_x(channel_count), channel_idx%index_y(channel_count), stat=alloc_status)
      if (alloc_status /= 0) error stop 'ERROR: allocation failure (channel_idx).'

      idx = 0
      do i = 1, nx
         do j = 1, ny
            idx = idx + 1
            channel_idx%index_x(idx) = fft_mode_index(i - 1, nx)
            channel_idx%index_y(idx) = fft_mode_index(j - 1, ny)
         end do
      end do
   end subroutine get_channel_index

   pure integer function mode_to_storage(mode_index, n) result(storage_index)
      implicit none
      integer, intent(in) :: mode_index, n

      storage_index = modulo(mode_index, n) + 1
   end function mode_to_storage

   function get_specular_channel_index(channel_idx) result(specular_channel_index)
      implicit none
      type(ChannelIdxData), intent(in) :: channel_idx
      integer :: specular_channel_index
      integer :: i

      specular_channel_index = 0
      do i = 1,size(channel_idx%index_x)
         if (channel_idx%index_x(i) .eq. 0 .and. channel_idx%index_y(i) .eq. 0) then
            specular_channel_index = i
            exit
         end if
      end do
      if (specular_channel_index .eq. 0) error stop 'ERROR: specular channel not found.'
   end function get_specular_channel_index

   subroutine run_preconditioned_gmres (n_z_points, &
   & potential_values,nkx,nky, &
   & a,b,c,perpendicular_kinetic_difference,scattered_state,kinetic_matrix,convergence_eps, &
   & preconditioner_flag,info)
      implicit none
      integer, intent(in) :: n_z_points
      integer, intent(in) :: nkx, nky, preconditioner_flag
      complex(dp), intent(in) :: potential_values(nkx,nky,n_z_points)
      complex(dp), intent(in) :: a(:), b(:), c(:)
      real(dp), intent(in) :: perpendicular_kinetic_difference(nkx,nky)
      complex(dp), intent(out) :: scattered_state(n_z_points,size(a))
      real(dp), intent(in) :: kinetic_matrix(n_z_points,n_z_points)
      real(dp), intent(in) :: convergence_eps
      integer, intent(out) :: info
      type(GmresOptions) :: options
      type(GmresResult) :: result

      options%rtol = convergence_eps
      options%preconditioner_flag = preconditioner_flag
      options%maxiter = l
      options%convergence_window = 3

      call solve_scattering_gmres ( &
         n_z_points, potential_values, nkx, nky, a, b, c, &
         perpendicular_kinetic_difference, scattered_state, kinetic_matrix, options, result &
         )

      info = result%info
   end subroutine run_preconditioned_gmres

   subroutine solve_scattering_gmres (n_z_points, &
   & potential_values,nkx,nky, &
   & a,b,c,perpendicular_kinetic_difference,scattered_state,kinetic_matrix,options,result)
      implicit none
      integer, intent(in) :: n_z_points
      integer, intent(in) :: nkx, nky
      complex(dp), intent(in) :: potential_values(nkx,nky,n_z_points)
      complex(dp), intent(in) :: a(:), b(:), c(:)
      real(dp), intent(in) :: perpendicular_kinetic_difference(nkx,nky)
      complex(dp), intent(out) :: scattered_state(n_z_points,size(a))
      real(dp), intent(in) :: kinetic_matrix(n_z_points,n_z_points)
      type(GmresOptions), intent(in) :: options
      type(GmresResult), intent(out) :: result

      integer :: alloc_status
      integer :: mn, channel_count
      type(ChannelIdxData) :: channel_idx
      complex(dp), allocatable :: x(:), xx(:,:), y(:), s(:)
      real(dp), allocatable :: intensity_work(:)
      real(dp), allocatable :: channel_eigenvalues(:), channel_preconditioner_factors(:,:), kinetic_matrix_work(:,:)

      channel_count = size(a)
      if (size(b) /= channel_count .or. size(c) /= channel_count) then
         error stop 'ERROR: inconsistent channel vector sizes.'
      end if
      if (size(scattered_state, 1) /= n_z_points .or. size(scattered_state, 2) /= channel_count) then
         error stop 'ERROR: scattered_state shape does not match solver dimensions.'
      end if
      if (options%rtol <= 0.0_dp) error stop 'ERROR: GMRES rtol must be positive.'
      if (options%maxiter <= 0) error stop 'ERROR: GMRES maxiter must be positive.'
      if (options%convergence_window <= 0) error stop 'ERROR: GMRES convergence_window must be positive.'

      call get_channel_index(perpendicular_kinetic_difference, channel_idx)
      if (size(channel_idx%index_x) /= channel_count) then
         error stop 'ERROR: channel count does not match perpendicular momentum grid.'
      end if

      mn = n_z_points*channel_count
      allocate(x(mn), xx(mn,l+1), y(mn), s(channel_count), stat=alloc_status)
      if (alloc_status /= 0) error stop 'ERROR: allocation failure (x, xx, y, s).'
      allocate(intensity_work(channel_count), stat=alloc_status)
      if (alloc_status /= 0) error stop 'ERROR: allocation failure (intensity_work).'
      allocate(channel_eigenvalues(n_z_points), &
      & channel_preconditioner_factors(n_z_points,channel_count), &
      & kinetic_matrix_work(n_z_points,n_z_points), stat=alloc_status)
      if (alloc_status /= 0) then
         error stop 'ERROR: allocation failure (channel_eigenvalues/channel_preconditioner_factors/kinetic_matrix_work).'
      end if

      kinetic_matrix_work = kinetic_matrix

      call build_preconditioner ( &
         n_z_points, channel_idx, potential_values, nkx, nky, &
         perpendicular_kinetic_difference, channel_eigenvalues, channel_preconditioner_factors, kinetic_matrix_work &
         )

      call solve_gmres_system ( &
         x, xx, y, n_z_points, channel_idx, potential_values, nkx, nky, &
         a, b, c, perpendicular_kinetic_difference, &
         channel_eigenvalues, channel_preconditioner_factors, intensity_work, s, &
         kinetic_matrix_work, options, result &
         )

      scattered_state = reshape(x, shape(scattered_state))

      if (allocated(channel_idx%index_x)) deallocate(channel_idx%index_x)
      if (allocated(channel_idx%index_y)) deallocate(channel_idx%index_y)
      if (allocated(x)) deallocate(x)
      if (allocated(xx)) deallocate(xx)
      if (allocated(y)) deallocate(y)
      if (allocated(s)) deallocate(s)
      if (allocated(intensity_work)) deallocate(intensity_work)
      if (allocated(channel_eigenvalues)) deallocate(channel_eigenvalues)
      if (allocated(channel_preconditioner_factors)) deallocate(channel_preconditioner_factors)
      if (allocated(kinetic_matrix_work)) deallocate(kinetic_matrix_work)
   end subroutine solve_scattering_gmres

   subroutine get_scattered_intensity (scattered_state, wave_a, wave_b, channel_intensity)
      implicit none
      complex(dp), intent(in) :: scattered_state(:,:)
      complex(dp), intent(in) :: wave_a(:), wave_b(:)
      real(dp), intent(out) :: channel_intensity(:)

      integer :: j, channel_count, n_z_points
      complex(dp) :: scattered_amplitude

      n_z_points = size(scattered_state, 1)
      channel_count = size(scattered_state, 2)
      if (size(wave_a) /= channel_count .or. size(wave_b) /= channel_count .or. size(channel_intensity) /= channel_count) then
         error stop 'ERROR: inconsistent channel vector sizes in get_scattered_intensity.'
      end if

      do j = 1,channel_count
         scattered_amplitude = (0.0d0,2.0d0)*wave_b(j)*scattered_state(n_z_points,j)
         if (j .eq. 1) scattered_amplitude = wave_a(j)+scattered_amplitude
         channel_intensity(j) = real(conjg(scattered_amplitude)*scattered_amplitude,kind=dp)
      enddo
   end subroutine get_scattered_intensity

   subroutine build_preconditioner (n_z_points, channel_idx, &
   & potential_values, nkx, nky, perpendicular_kinetic_difference, &
   & eigenvalues, preconditioner_factors, kinetic_matrix)
      implicit none
      integer, intent(in) :: n_z_points, nkx, nky
      type(ChannelIdxData), intent(in) :: channel_idx
      complex(dp), intent(in) :: potential_values(nkx,nky,n_z_points)
      real(dp), intent(in) :: perpendicular_kinetic_difference(nkx,nky)
      real(dp), intent(out) :: eigenvalues(n_z_points)
      real(dp), intent(out) :: preconditioner_factors(n_z_points,size(channel_idx%index_x))
      real(dp), intent(inout) :: kinetic_matrix(n_z_points,n_z_points)
      real(dp), allocatable :: g(:)
      integer :: i, j, k, ix, iy, ierr, alloc_status, channel_count
      real(dp) :: channel_energy
      external :: rs

      channel_count = size(channel_idx%index_x)

      allocate(g(n_z_points), stat=alloc_status)
      if (alloc_status /= 0) error stop 'ERROR: allocation failure (g).'

      do k = 1,n_z_points
         kinetic_matrix(k,k) = kinetic_matrix(k,k) + real(potential_values(1,1,k),kind=dp)
      enddo

      call rs (n_z_points,n_z_points,kinetic_matrix,eigenvalues, &
      & kinetic_matrix,preconditioner_factors,ierr)
      if (ierr .ne. 0) error stop 'precon 2'

      do j = 1,channel_count
         ix = mode_to_storage(channel_idx%index_x(j), nkx)
         iy = mode_to_storage(channel_idx%index_y(j), nky)
         channel_energy = perpendicular_kinetic_difference(ix, iy)

         do k = 1,n_z_points
            g(k) = kinetic_matrix(n_z_points,k) /(channel_energy+eigenvalues(k))
            preconditioner_factors(k,j) = 0.0d0
         enddo

         do i = 1,n_z_points
            do k = 1,n_z_points
               preconditioner_factors(k,j) = preconditioner_factors(k,j) + kinetic_matrix(k,i)*g(i)
            enddo
         enddo

      enddo
      if (allocated(g)) deallocate(g)
   end subroutine build_preconditioner

   subroutine debug_build_preconditioner ( &
   & potential_values, perpendicular_kinetic_difference, parallel_kinetic_energy, &
   & eigenvalues, preconditioner_factors, eigenvectors, ierr)
      implicit none
      complex(dp), intent(in) :: potential_values(:,:,:)
      real(dp), intent(in) :: perpendicular_kinetic_difference(:,:)
      real(dp), intent(in) :: parallel_kinetic_energy(:,:)
      real(dp), intent(out) :: eigenvalues(size(potential_values, 3))
      real(dp), intent(out) :: preconditioner_factors(size(potential_values, 3), &
      & size(potential_values, 1) * size(potential_values, 2))
      real(dp), intent(out) :: eigenvectors(size(potential_values, 3), size(potential_values, 3))
      integer, intent(out) :: ierr

      integer :: nkx, nky, n_z_points
      type(ChannelIdxData) :: channel_idx

      ierr = 0
      nkx = size(potential_values, 1)
      nky = size(potential_values, 2)
      n_z_points = size(potential_values, 3)

      if (size(perpendicular_kinetic_difference, 1) /= nkx .or. &
      & size(perpendicular_kinetic_difference, 2) /= nky) then
         ierr = 1
         return
      end if
      if (size(parallel_kinetic_energy, 1) /= n_z_points .or. &
      & size(parallel_kinetic_energy, 2) /= n_z_points) then
         ierr = 2
         return
      end if

      call get_channel_index(perpendicular_kinetic_difference, channel_idx)

      eigenvectors = parallel_kinetic_energy
      call build_preconditioner ( &
      & n_z_points, channel_idx, potential_values, nkx, nky, &
      & perpendicular_kinetic_difference, eigenvalues, preconditioner_factors, eigenvectors &
      & )

      if (allocated(channel_idx%index_x)) deallocate(channel_idx%index_x)
      if (allocated(channel_idx%index_y)) deallocate(channel_idx%index_y)
   end subroutine debug_build_preconditioner

   subroutine init_scattering_operator (operator_data,n_z_points,channel_idx, &
   & potential_values,nkx,nky,c,perpendicular_kinetic_difference)
      implicit none
      type(ScatteringOperator), intent(out) :: operator_data
      integer, intent(in) :: n_z_points, nkx, nky
      type(ChannelIdxData), target, intent(in) :: channel_idx
      complex(dp), target, intent(in) :: potential_values(nkx,nky,n_z_points)
      complex(dp), target, intent(in) :: c(:)
      real(dp), target, intent(in) :: perpendicular_kinetic_difference(nkx,nky)

      operator_data%n_z_points = n_z_points
      operator_data%nkx = nkx
      operator_data%nky = nky
      operator_data%channel_idx => channel_idx
      operator_data%potential_values => potential_values
      operator_data%wave_c => c
      operator_data%perpendicular_kinetic_difference => perpendicular_kinetic_difference
   end subroutine init_scattering_operator

   subroutine debug_apply_upper_block (potential_values, state_in, state_out, ierr)
      implicit none
      complex(dp), intent(in) :: potential_values(:,:,:)
      complex(dp), intent(in) :: state_in(:,:)
      complex(dp), intent(out) :: state_out(size(state_in, 1), size(state_in, 2))
      integer, intent(out) :: ierr

      integer :: nkx, nky, n_z_points, channel_count
      real(dp), allocatable :: dummy_perpendicular(:,:)
      complex(dp), allocatable :: dummy_wave_c(:)
      type(ChannelIdxData) :: channel_idx
      type(ScatteringOperator) :: operator_data
      integer :: alloc_status

      ierr = 0
      nkx = size(potential_values, 1)
      nky = size(potential_values, 2)
      n_z_points = size(potential_values, 3)
      channel_count = nkx * nky

      if (size(state_in, 1) /= n_z_points .or. size(state_in, 2) /= channel_count) then
         ierr = 1
         return
      end if

      allocate(dummy_perpendicular(nkx, nky), stat=alloc_status)
      if (alloc_status /= 0) error stop 'ERROR: allocation failure (dummy_perpendicular).'
      allocate(dummy_wave_c(channel_count), stat=alloc_status)
      if (alloc_status /= 0) error stop 'ERROR: allocation failure (dummy_wave_c).'
      dummy_perpendicular = 0.0_dp
      dummy_wave_c = (0.0_dp, 0.0_dp)

      call get_channel_index(dummy_perpendicular, channel_idx)
      call init_scattering_operator ( &
      & operator_data, n_z_points, channel_idx, potential_values, nkx, nky, dummy_wave_c, dummy_perpendicular &
      & )

      state_out = state_in
      call apply_upper_block(state_out, operator_data)

      if (allocated(channel_idx%index_x)) deallocate(channel_idx%index_x)
      if (allocated(channel_idx%index_y)) deallocate(channel_idx%index_y)
      if (allocated(dummy_perpendicular)) deallocate(dummy_perpendicular)
      if (allocated(dummy_wave_c)) deallocate(dummy_wave_c)
   end subroutine debug_apply_upper_block

   subroutine debug_solve_lower_block ( &
   & potential_values, wave_c, perpendicular_kinetic_difference, parallel_kinetic_energy, &
   & state_in, state_out, ierr)
      implicit none
      complex(dp), intent(in) :: potential_values(:,:,:)
      complex(dp), intent(in) :: wave_c(:)
      real(dp), intent(in) :: perpendicular_kinetic_difference(:,:)
      real(dp), intent(in) :: parallel_kinetic_energy(:,:)
      complex(dp), intent(in) :: state_in(:,:)
      complex(dp), intent(out) :: state_out(size(state_in, 1), size(state_in, 2))
      integer, intent(out) :: ierr

      integer :: nkx, nky, n_z_points, channel_count
      type(ChannelIdxData) :: channel_idx
      type(ScatteringOperator) :: operator_data
      real(dp), allocatable :: eigenvalues(:), preconditioner_factors(:,:), kinetic_matrix_work(:,:)
      integer :: alloc_status

      ierr = 0
      nkx = size(potential_values, 1)
      nky = size(potential_values, 2)
      n_z_points = size(potential_values, 3)
      channel_count = nkx * nky

      if (size(wave_c) /= channel_count) then
         ierr = 1
         return
      end if
      if (size(perpendicular_kinetic_difference, 1) /= nkx .or. &
      & size(perpendicular_kinetic_difference, 2) /= nky) then
         ierr = 2
         return
      end if
      if (size(parallel_kinetic_energy, 1) /= n_z_points .or. &
      & size(parallel_kinetic_energy, 2) /= n_z_points) then
         ierr = 3
         return
      end if
      if (size(state_in, 1) /= n_z_points .or. size(state_in, 2) /= channel_count) then
         ierr = 4
         return
      end if

      allocate(eigenvalues(n_z_points), preconditioner_factors(n_z_points, channel_count), &
      & kinetic_matrix_work(n_z_points, n_z_points), stat=alloc_status)
      if (alloc_status /= 0) error stop 'ERROR: allocation failure (debug_solve_lower_block arrays).'

      call get_channel_index(perpendicular_kinetic_difference, channel_idx)
      call init_scattering_operator ( &
      & operator_data, n_z_points, channel_idx, potential_values, nkx, nky, wave_c, perpendicular_kinetic_difference &
      & )

      kinetic_matrix_work = parallel_kinetic_energy
      call build_preconditioner ( &
      & n_z_points, channel_idx, potential_values, nkx, nky, perpendicular_kinetic_difference, &
      & eigenvalues, preconditioner_factors, kinetic_matrix_work &
      & )

      state_out = state_in
      call solve_lower_block(state_out, operator_data, eigenvalues, preconditioner_factors, kinetic_matrix_work)

      if (allocated(channel_idx%index_x)) deallocate(channel_idx%index_x)
      if (allocated(channel_idx%index_y)) deallocate(channel_idx%index_y)
      if (allocated(eigenvalues)) deallocate(eigenvalues)
      if (allocated(preconditioner_factors)) deallocate(preconditioner_factors)
      if (allocated(kinetic_matrix_work)) deallocate(kinetic_matrix_work)
   end subroutine debug_solve_lower_block

   subroutine apply_scattering_operator (x,y,operator_data,eigenvalues,preconditioner_factors,kinetic_matrix,options)
      implicit none
      type(ScatteringOperator), intent(in) :: operator_data
      type(GmresOptions), intent(in) :: options
      complex(dp), intent(inout) :: x(operator_data%n_z_points*size(operator_data%channel_idx%index_x))
      complex(dp), intent(inout) :: y(operator_data%n_z_points*size(operator_data%channel_idx%index_x))
      real(dp), intent(in) :: eigenvalues(operator_data%n_z_points)
      real(dp), intent(in) :: preconditioner_factors(operator_data%n_z_points,size(operator_data%channel_idx%index_x))
      real(dp), intent(in) :: kinetic_matrix(operator_data%n_z_points,operator_data%n_z_points)
      integer :: mn

      mn = operator_data%n_z_points*size(operator_data%channel_idx%index_x)
      y(1:mn) = x(1:mn)
      call apply_upper_block (x,operator_data)
      call solve_lower_block (x,operator_data,eigenvalues,preconditioner_factors,kinetic_matrix)
      x(1:mn) = y(1:mn) + x(1:mn)

      if (options%preconditioner_flag .eq. 1) then
         y(1:mn) = x(1:mn)
         call apply_upper_block (x,operator_data)
         call solve_lower_block (x,operator_data,eigenvalues,preconditioner_factors,kinetic_matrix)
         x(1:mn) = y(1:mn) - x(1:mn)
      end if
   end subroutine apply_scattering_operator

   subroutine solve_gmres_system (x,xx,y,n_z_points, &
   & channel_idx, potential_values, nkx, nky, &
   & a,b,c,perpendicular_kinetic_difference,channel_eigenvalues,channel_preconditioner_factors,p,s,kinetic_matrix,options,result)
      implicit none
      integer, intent(in) :: n_z_points, nkx, nky
      type(ChannelIdxData), intent(in) :: channel_idx
      complex(dp), intent(out) :: x(n_z_points*size(channel_idx%index_x))
      complex(dp), intent(out) :: xx(n_z_points*size(channel_idx%index_x),l+1)
      complex(dp), intent(inout) :: y(n_z_points*size(channel_idx%index_x))
      complex(dp), target, intent(in) :: potential_values(nkx,nky,n_z_points)
      complex(dp), intent(in) :: a(:), b(:)
      complex(dp), target, intent(in) :: c(:)
      complex(dp), intent(out) :: s(size(channel_idx%index_x))
      real(dp), target, intent(in) :: perpendicular_kinetic_difference(nkx,nky), channel_eigenvalues(n_z_points)
      real(dp), target, intent(in) :: channel_preconditioner_factors(n_z_points,size(channel_idx%index_x))
      real(dp), intent(out) :: p(size(channel_idx%index_x))
      real(dp), target, intent(in) :: kinetic_matrix(n_z_points,n_z_points)
      type(GmresOptions), intent(in) :: options
      type(GmresResult), intent(out) :: result

      integer :: alloc_status
      integer :: mn, i, j, k, kconv, kk, channel_count, maxiter
      integer :: specular_channel_index
      real(dp) :: xnorm, unit, diff, pj
      type(ScatteringOperator) :: operator_data
      complex(dp), allocatable :: h(:,:), g(:), z(:)
      complex(dp), allocatable :: co(:), si(:)
      complex(dp) temp

      channel_count = size(channel_idx%index_x)
      specular_channel_index = 0
      specular_channel_index = get_specular_channel_index(channel_idx)

      call init_scattering_operator ( &
      & operator_data, n_z_points, channel_idx, potential_values, nkx, nky, c, &
      & perpendicular_kinetic_difference &
      & )

      allocate(h(l+1,l+1), g(l+1), z(l+1), co(l+1), si(l+1), stat=alloc_status)
      if (alloc_status /= 0) error stop 'ERROR: allocation failure (h, g, z, co, si).'

      result%info = 0
      result%iterations = 0
      mn = n_z_points*channel_count
      do i = 1,mn
         x(i) = (0.0d0,0.0d0)
      enddo

      xx(1:mn,1)=x(1:mn)
      y(1:mn) = x(1:mn)
      call apply_upper_block (x,operator_data)
      do i = 1,mn
         x(i) = -x(i)
      enddo
      x(n_z_points*specular_channel_index) = b(specular_channel_index)+x(n_z_points*specular_channel_index)
      call solve_lower_block (x,operator_data,channel_eigenvalues,channel_preconditioner_factors,kinetic_matrix)
      x(1:mn) = x(1:mn)-y(1:mn)
      if (options%preconditioner_flag .eq. 1) then
         y(1:mn) = x(1:mn)
         call apply_upper_block (x,operator_data)
         call solve_lower_block (x,operator_data,channel_eigenvalues,channel_preconditioner_factors,kinetic_matrix)
         x(1:mn) = y(1:mn)-x(1:mn)
      end if
      xnorm = 0.0d0
      do i = 1,mn
         xnorm = xnorm + real(conjg(x(i))*x(i),kind=dp)
      enddo
      xnorm = sqrt(xnorm)
      g(1) = xnorm

      kconv = 0
      do j = 1,channel_count
         p(j) = 0.0d0
      enddo
      kk = 0
      maxiter = min(options%maxiter, l)
      do k = 1,maxiter
         do i = 1,mn
            x(i) = x(i)/xnorm
         enddo
         xx(1:mn,k+1)=x(1:mn)
         call apply_scattering_operator (x,y,operator_data, &
         & channel_eigenvalues,channel_preconditioner_factors,kinetic_matrix,options)
         y(1:mn)=xx(1:mn,1)
         do i = 1,channel_count
            s(i) = y(n_z_points*i)
         enddo
         do j = 1,k
            y(1:mn)=xx(1:mn,j+1)
            h(j,k) = (0.0d0,0.0d0)
            do i = 1,mn
               h(j,k) = h(j,k)+conjg(y(i))*x(i)
            enddo
            do i = 1,mn
               x(i) = x(i)-y(i)*h(j,k)
            enddo
            if (j .lt. k) then
               do i = 1,channel_count
                  s(i) = s(i)+y(n_z_points*i)*z(j)
               enddo
            endif
         enddo
         do i = 1,channel_count
            s(i) = (0.0d0,2.0d0)*b(i)*s(i)
         enddo
         s(specular_channel_index) = a(specular_channel_index)+s(specular_channel_index)
         xnorm = 0.0d0
         do i = 1,mn
            xnorm = xnorm + real(conjg(x(i))*x(i),kind=dp)
         enddo
         xnorm = sqrt(xnorm)
         h(k+1,k) = xnorm
         do j = 1,k-1
            temp = co(j)*h(j,k)+conjg(si(j))*h(j+1,k)
            h(j+1,k) = conjg(co(j))*h(j+1,k)-si(j)*h(j,k)
            h(j,k) = temp
         enddo
         call compute_complex_givens_rotation (h(k,k),h(k+1,k),co(k),si(k))
         g(k+1) = -si(k)*g(k)
         g(k) = co(k)*g(k)
         do j = 1,k
            z(j) = g(j)
         enddo
         do j = k,1,-1
            z(j) = z(j)/h(j,j)
            do i = 1,j-1
               z(i) = z(i)-h(i,j)*z(j)
            enddo
         enddo

         unit = 0.0d0
         diff = 0.0d0
         do j = 1,channel_count
            pj = real(conjg(s(j))*s(j),kind=dp)
            unit = unit+pj
            diff = max(diff,abs(pj-p(j)))
            p(j) = pj
         enddo
         diff = max(diff,abs(unit-1.0d0))
         if (diff .lt. options%rtol) then
            kconv = kconv+1
         else
            kconv = 0
         endif
         kk = k
         if (kconv.eq.options%convergence_window .or. xnorm.eq.0.0d0) exit
      enddo

      x(1:mn)=xx(1:mn,1)
      do j = 1,kk
         y(1:mn)=xx(1:mn,j+1)
         do i = 1,mn
            x(i) = x(i)+y(i)*z(j)
         enddo
      enddo

      result%iterations = kk
      if (kconv.lt.options%convergence_window .and. xnorm.gt.0.0d0) then
         result%info = 1
      endif

      if (allocated(h)) deallocate(h)
      if (allocated(g)) deallocate(g)
      if (allocated(z)) deallocate(z)
      if (allocated(co)) deallocate(co)
      if (allocated(si)) deallocate(si)
   end subroutine solve_gmres_system

   subroutine apply_upper_block (state_vector,operator_data)
      implicit none
      type(ScatteringOperator), intent(in) :: operator_data
      complex(dp), intent(inout) :: state_vector(operator_data%n_z_points,size(operator_data%channel_idx%index_x))
      integer :: i, j, k, ix, iy, channel_count

      channel_count = size(operator_data%channel_idx%index_x)
      do j = 1,channel_count
         do k = 1,operator_data%n_z_points
            state_vector(k,j) = (0.0d0,0.0d0)
         enddo
         do i = j+1,channel_count
            ix = mode_to_storage(operator_data%channel_idx%index_x(j)-operator_data%channel_idx%index_x(i), operator_data%nkx)
            iy = mode_to_storage(operator_data%channel_idx%index_y(j)-operator_data%channel_idx%index_y(i), operator_data%nky)
            do k = 1,operator_data%n_z_points
               state_vector(k,j) = state_vector(k,j) + operator_data%potential_values(ix,iy,k)*state_vector(k,i)
            enddo
         enddo
      enddo
   end subroutine apply_upper_block

   subroutine solve_lower_block (state_vector,operator_data,eigenvalues,preconditioner_factors,kinetic_matrix)
      implicit none
      type(ScatteringOperator), intent(in) :: operator_data
      complex(dp), intent(inout) :: state_vector(operator_data%n_z_points,size(operator_data%channel_idx%index_x))
      real(dp), intent(in) :: eigenvalues(operator_data%n_z_points)
      real(dp), intent(in) :: preconditioner_factors(operator_data%n_z_points,size(operator_data%channel_idx%index_x))
      real(dp), intent(in) :: kinetic_matrix(operator_data%n_z_points,operator_data%n_z_points)
      integer :: i, j, k, l, ix, iy, channel_count
      real(dp) :: channel_energy
      complex(dp) :: y(operator_data%n_z_points), fac

      channel_count = size(operator_data%channel_idx%index_x)
      do j = 1,channel_count
         do i = 1,j-1
            ix = mode_to_storage(operator_data%channel_idx%index_x(j)-operator_data%channel_idx%index_x(i), operator_data%nkx)
            iy = mode_to_storage(operator_data%channel_idx%index_y(j)-operator_data%channel_idx%index_y(i), operator_data%nky)
            do k = 1,operator_data%n_z_points
               state_vector(k,j) = state_vector(k,j) - operator_data%potential_values(ix,iy,k)*state_vector(k,i)
            enddo
         enddo

         ix = mode_to_storage(operator_data%channel_idx%index_x(j), operator_data%nkx)
         iy = mode_to_storage(operator_data%channel_idx%index_y(j), operator_data%nky)
         channel_energy = operator_data%perpendicular_kinetic_difference(ix, iy)

         do k = 1,operator_data%n_z_points
            y(k) = (0.0d0,0.0d0)
            do l = 1,operator_data%n_z_points
               y(k) = y(k)+state_vector(l,j)*kinetic_matrix(l,k)
            enddo
            y(k) = y(k)/(channel_energy+eigenvalues(k))
         enddo
         do k = 1,operator_data%n_z_points
            state_vector(k,j) = (0.0d0,0.0d0)
         enddo
         do l = 1,operator_data%n_z_points
            do k = 1,operator_data%n_z_points
               state_vector(k,j) = state_vector(k,j)+kinetic_matrix(k,l)*y(l)
            enddo
         enddo
         fac = state_vector(operator_data%n_z_points,j)*operator_data%wave_c(j)/ &
         & (1.0d0-preconditioner_factors(operator_data%n_z_points,j)*operator_data%wave_c(j))
         do k = 1,operator_data%n_z_points
            state_vector(k,j) = state_vector(k,j) + fac*preconditioner_factors(k,j)
         enddo
      enddo
   end subroutine solve_lower_block

   subroutine compute_complex_givens_rotation (input_a,input_b,c,s)
      implicit none
      complex(dp), intent(inout) :: input_a, input_b
      complex(dp), intent(out) :: c, s
      real(dp) :: scale, r
      complex(dp) :: rho, z

      rho = input_b
      if (abs(input_a) .gt. abs(input_b)) rho = input_a
      scale = abs(input_a) + abs(input_b)
      if (scale .eq. 0.0d0) then
         c = 1.0d0
         s = 0.0d0
         r = 0.0d0
      else
         r = real((input_a/scale)*conjg(input_a/scale) + &
         & (input_b/scale)*conjg(input_b/scale),kind=dp)
         r = scale*sqrt(r)
         if (real(rho,kind=dp) .lt. 0.0d0) r = -r
         c = conjg(input_a/r)
         s = input_b/r
      end if
      z = 1.0d0
      if (abs(input_a) .gt. abs(input_b)) z = s
      if (abs(input_b) .ge. abs(input_a) .and. abs(c) .ne. 0.0d0) z = 1.0d0/c
      input_a = r
      input_b = z
   end subroutine compute_complex_givens_rotation

end module multiscat_gmres
