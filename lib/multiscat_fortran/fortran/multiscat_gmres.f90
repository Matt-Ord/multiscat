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
      integer :: preconditioner_flag = 0
      type(ChannelIdxData), pointer :: channel_idx => null()
      complex(dp), pointer :: potential_values(:,:,:) => null()
      complex(dp), pointer :: wave_c(:) => null()
      real(dp), pointer :: perpendicular_kinetic_difference(:,:) => null()
      real(dp), pointer :: eigenvalues(:) => null()
      real(dp), pointer :: preconditioner_factors(:,:) => null()
      real(dp), pointer :: kinetic_matrix(:,:) => null()
   end type ScatteringOperator

   public :: ChannelIdxData
   public :: GmresOptions
   public :: GmresResult
   public :: ScatteringOperator
   public :: get_channel_index
   public :: solve_scattering_gmres
   public :: run_preconditioned_gmres

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
   & a,b,c,perpendicular_kinetic_difference,p,t,convergence_eps, &
   & preconditioner_flag)
      implicit none
      integer, intent(in) :: n_z_points
      integer, intent(in) :: nkx, nky, preconditioner_flag
      complex(dp), intent(in) :: potential_values(nkx,nky,n_z_points)
      complex(dp), intent(in) :: a(:), b(:), c(:)
      real(dp), intent(in) :: perpendicular_kinetic_difference(nkx,nky)
      real(dp), intent(out) :: p(:)
      real(dp), intent(in) :: t(n_z_points,n_z_points)
      real(dp), intent(in) :: convergence_eps
      type(GmresOptions) :: options
      type(GmresResult) :: result

      options%rtol = convergence_eps
      options%preconditioner_flag = preconditioner_flag
      options%maxiter = l
      options%convergence_window = 3

      call solve_scattering_gmres ( &
         n_z_points, potential_values, nkx, nky, a, b, c, &
         perpendicular_kinetic_difference, p, t, options, result &
         )

      if (result%info /= 0) p = -1.0_dp
   end subroutine run_preconditioned_gmres

   subroutine solve_scattering_gmres (n_z_points, &
   & potential_values,nkx,nky, &
   & a,b,c,perpendicular_kinetic_difference,p,t,options,result)
      implicit none
      integer, intent(in) :: n_z_points
      integer, intent(in) :: nkx, nky
      complex(dp), intent(in) :: potential_values(nkx,nky,n_z_points)
      complex(dp), intent(in) :: a(:), b(:), c(:)
      real(dp), intent(in) :: perpendicular_kinetic_difference(nkx,nky)
      real(dp), intent(out) :: p(:)
      real(dp), intent(in) :: t(n_z_points,n_z_points)
      type(GmresOptions), intent(in) :: options
      type(GmresResult), intent(out) :: result

      integer :: alloc_status
      integer :: mn, channel_count
      type(ChannelIdxData) :: channel_idx
      complex(dp), allocatable :: x(:), xx(:,:), y(:), s(:)
      real(dp), allocatable :: e(:), f(:,:), t_work(:,:)

      channel_count = size(a)
      if (size(b) /= channel_count .or. size(c) /= channel_count .or. size(p) /= channel_count) then
         error stop 'ERROR: inconsistent channel vector sizes.'
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
      allocate(e(n_z_points), f(n_z_points,channel_count), t_work(n_z_points,n_z_points), stat=alloc_status)
      if (alloc_status /= 0) error stop 'ERROR: allocation failure (e, f, t_work).'

      t_work = t

      call build_preconditioner ( &
         n_z_points, channel_idx, potential_values, nkx, nky, &
         perpendicular_kinetic_difference, e, f, t_work &
         )

      call solve_gmres_system ( &
         x, xx, y, n_z_points, channel_idx, potential_values, nkx, nky, &
         a, b, c, perpendicular_kinetic_difference, e, f, p, s, t_work, options, result &
         )

      if (allocated(channel_idx%index_x)) deallocate(channel_idx%index_x)
      if (allocated(channel_idx%index_y)) deallocate(channel_idx%index_y)
      if (allocated(x)) deallocate(x)
      if (allocated(xx)) deallocate(xx)
      if (allocated(y)) deallocate(y)
      if (allocated(s)) deallocate(s)
      if (allocated(e)) deallocate(e)
      if (allocated(f)) deallocate(f)
      if (allocated(t_work)) deallocate(t_work)
   end subroutine solve_scattering_gmres

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

   subroutine init_scattering_operator (operator_data,n_z_points,channel_idx, &
   & potential_values,nkx,nky,c,perpendicular_kinetic_difference,e,f,t,preconditioner_flag)
      implicit none
      type(ScatteringOperator), intent(out) :: operator_data
      integer, intent(in) :: n_z_points, nkx, nky, preconditioner_flag
      type(ChannelIdxData), target, intent(in) :: channel_idx
      complex(dp), target, intent(in) :: potential_values(nkx,nky,n_z_points)
      complex(dp), target, intent(in) :: c(:)
      real(dp), target, intent(in) :: perpendicular_kinetic_difference(nkx,nky), e(n_z_points)
      real(dp), target, intent(in) :: f(n_z_points,size(channel_idx%index_x))
      real(dp), target, intent(in) :: t(n_z_points,n_z_points)

      operator_data%n_z_points = n_z_points
      operator_data%nkx = nkx
      operator_data%nky = nky
      operator_data%preconditioner_flag = preconditioner_flag
      operator_data%channel_idx => channel_idx
      operator_data%potential_values => potential_values
      operator_data%wave_c => c
      operator_data%perpendicular_kinetic_difference => perpendicular_kinetic_difference
      operator_data%eigenvalues => e
      operator_data%preconditioner_factors => f
      operator_data%kinetic_matrix => t
   end subroutine init_scattering_operator

   subroutine apply_scattering_operator (x,y,operator_data)
      implicit none
      type(ScatteringOperator), intent(in) :: operator_data
      complex(dp), intent(inout) :: x(operator_data%n_z_points*size(operator_data%channel_idx%index_x))
      complex(dp), intent(inout) :: y(operator_data%n_z_points*size(operator_data%channel_idx%index_x))
      integer :: mn

      mn = operator_data%n_z_points*size(operator_data%channel_idx%index_x)
      y(1:mn) = x(1:mn)
      call apply_upper_block (x,operator_data%n_z_points,operator_data%channel_idx, &
      & operator_data%potential_values,operator_data%nkx,operator_data%nky)
      call solve_lower_block (x,operator_data%n_z_points,operator_data%channel_idx, &
      & operator_data%potential_values,operator_data%nkx,operator_data%nky, &
      & operator_data%wave_c,operator_data%perpendicular_kinetic_difference, &
      & operator_data%eigenvalues,operator_data%preconditioner_factors,operator_data%kinetic_matrix)
      x(1:mn) = y(1:mn) + x(1:mn)

      if (operator_data%preconditioner_flag .eq. 1) then
         y(1:mn) = x(1:mn)
         call apply_upper_block (x,operator_data%n_z_points,operator_data%channel_idx, &
         & operator_data%potential_values,operator_data%nkx,operator_data%nky)
         call solve_lower_block (x,operator_data%n_z_points,operator_data%channel_idx, &
         & operator_data%potential_values,operator_data%nkx,operator_data%nky, &
         & operator_data%wave_c,operator_data%perpendicular_kinetic_difference, &
         & operator_data%eigenvalues,operator_data%preconditioner_factors,operator_data%kinetic_matrix)
         x(1:mn) = y(1:mn) - x(1:mn)
      end if
   end subroutine apply_scattering_operator

   subroutine solve_gmres_system (x,xx,y,n_z_points, &
   & channel_idx, potential_values, nkx, nky, &
   & a,b,c,perpendicular_kinetic_difference,e,f,p,s,t,options,result)
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
      real(dp), target, intent(in) :: perpendicular_kinetic_difference(nkx,nky), e(n_z_points)
      real(dp), target, intent(in) :: f(n_z_points,size(channel_idx%index_x))
      real(dp), intent(out) :: p(size(channel_idx%index_x))
      real(dp), target, intent(in) :: t(n_z_points,n_z_points)
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
      & perpendicular_kinetic_difference, e, f, t, options%preconditioner_flag &
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
      call apply_upper_block (x,n_z_points,channel_idx, potential_values,nkx,nky)
      do i = 1,mn
         x(i) = -x(i)
      enddo
      x(n_z_points*specular_channel_index) = b(specular_channel_index)+x(n_z_points*specular_channel_index)
      call solve_lower_block (x,n_z_points,channel_idx, potential_values,nkx,nky, &
      & c,perpendicular_kinetic_difference,e,f,t)
      x(1:mn) = x(1:mn)-y(1:mn)
      if (options%preconditioner_flag .eq. 1) then
         y(1:mn) = x(1:mn)
         call apply_upper_block (x,n_z_points,channel_idx,potential_values,nkx,nky)
         call solve_lower_block (x,n_z_points,channel_idx,potential_values,nkx,nky, &
         & c,perpendicular_kinetic_difference,e,f,t)
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
         call apply_scattering_operator (x,y,operator_data)
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

   subroutine apply_upper_block (state_vector,n_z_points, channel_idx, &
   & potential_values,nkx,nky)
      implicit none
      integer, intent(in) :: n_z_points, nkx, nky
      type(ChannelIdxData), intent(in) :: channel_idx
      complex(dp), intent(inout) :: state_vector(n_z_points,size(channel_idx%index_x))
      complex(dp), intent(in) :: potential_values(nkx,nky,n_z_points)
      integer :: i, j, k, ix, iy, channel_count

      channel_count = size(channel_idx%index_x)
      do j = 1,channel_count
         do k = 1,n_z_points
            state_vector(k,j) = (0.0d0,0.0d0)
         enddo
         do i = j+1,channel_count
            ix = mode_to_storage(channel_idx%index_x(j)-channel_idx%index_x(i), nkx)
            iy = mode_to_storage(channel_idx%index_y(j)-channel_idx%index_y(i), nky)
            do k = 1,n_z_points
               state_vector(k,j) = state_vector(k,j) + potential_values(ix,iy,k)*state_vector(k,i)
            enddo
         enddo
      enddo
   end subroutine apply_upper_block

   subroutine solve_lower_block (state_vector,n_z_points, channel_idx, &
   & potential_values,nkx,nky,wave_c,perpendicular_kinetic_difference, &
   & eigenvalues,preconditioner_factors,kinetic_matrix)
      implicit none
      integer, intent(in) :: n_z_points, nkx, nky
      type(ChannelIdxData), intent(in) :: channel_idx
      complex(dp), intent(inout) :: state_vector(n_z_points,size(channel_idx%index_x))
      complex(dp), intent(in) :: potential_values(nkx,nky,n_z_points)
      complex(dp), intent(in) :: wave_c(size(channel_idx%index_x))
      real(dp), intent(in) :: perpendicular_kinetic_difference(nkx,nky), eigenvalues(n_z_points)
      real(dp), intent(in) :: preconditioner_factors(n_z_points,size(channel_idx%index_x))
      real(dp), intent(in) :: kinetic_matrix(n_z_points,n_z_points)
      integer :: i, j, k, l, ix, iy, channel_count
      real(dp) :: channel_energy
      complex(dp) :: y(n_z_points), fac

      channel_count = size(channel_idx%index_x)
      do j = 1,channel_count
         do i = 1,j-1
            ix = mode_to_storage(channel_idx%index_x(j)-channel_idx%index_x(i), nkx)
            iy = mode_to_storage(channel_idx%index_y(j)-channel_idx%index_y(i), nky)
            do k = 1,n_z_points
               state_vector(k,j) = state_vector(k,j) - potential_values(ix,iy,k)*state_vector(k,i)
            enddo
         enddo

         ix = mode_to_storage(channel_idx%index_x(j), nkx)
         iy = mode_to_storage(channel_idx%index_y(j), nky)
         channel_energy = perpendicular_kinetic_difference(ix, iy)

         do k = 1,n_z_points
            y(k) = (0.0d0,0.0d0)
            do l = 1,n_z_points
               y(k) = y(k)+state_vector(l,j)*kinetic_matrix(l,k)
            enddo
            y(k) = y(k)/(channel_energy+eigenvalues(k))
         enddo
         do k = 1,n_z_points
            state_vector(k,j) = (0.0d0,0.0d0)
         enddo
         do l = 1,n_z_points
            do k = 1,n_z_points
               state_vector(k,j) = state_vector(k,j)+kinetic_matrix(k,l)*y(l)
            enddo
         enddo
         fac = state_vector(n_z_points,j)*wave_c(j)/ &
         & (1.0d0-preconditioner_factors(n_z_points,j)*wave_c(j))
         do k = 1,n_z_points
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
