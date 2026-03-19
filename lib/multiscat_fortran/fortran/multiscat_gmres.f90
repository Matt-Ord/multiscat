subroutine run_preconditioned_gmres (n_z_points, &
& channel_index_x,channel_index_y, &
& channel_count,specular_channel_index, &
& fourier_values,fourier_index_x,fourier_index_y, &
& n_fourier_components,specular_fourier_component_index, &
& a,b,c,d,p,t,convergence_eps, &
& preconditioner_flag)
   use, intrinsic :: iso_fortran_env, only: real64
   implicit none
   integer, parameter :: dp = real64
   integer, parameter :: l = 2000
   integer, intent(in) :: n_z_points, channel_count, specular_channel_index
   integer, intent(in) :: n_fourier_components, preconditioner_flag
   integer, intent(in) :: specular_fourier_component_index
   integer, intent(in) :: channel_index_x(channel_count), channel_index_y(channel_count)
   integer, intent(in) :: fourier_index_x(n_fourier_components)
   integer, intent(in) :: fourier_index_y(n_fourier_components)
   complex(dp), intent(in) :: fourier_values(n_z_points,n_fourier_components)
   complex(dp), intent(in) :: a(channel_count), b(channel_count), c(channel_count)
   real(dp), intent(in) :: d(channel_count)
   real(dp), intent(out) :: p(channel_count)
   real(dp), intent(in) :: t(n_z_points,n_z_points)
   real(dp), intent(in) :: convergence_eps

   integer :: ifail
   integer :: alloc_status
   integer :: mn
   complex(dp), allocatable :: x(:), xx(:,:), y(:), s(:)
   real(dp), allocatable :: e(:), f(:,:), t_work(:,:)

   mn = n_z_points*channel_count
   allocate(x(mn), xx(mn,l+1), y(mn), s(channel_count), stat=alloc_status)
   if (alloc_status /= 0) error stop 'ERROR: allocation failure (x, xx, y, s).'
   allocate(e(n_z_points), f(n_z_points,channel_count), t_work(n_z_points,n_z_points), stat=alloc_status)
   if (alloc_status /= 0) error stop 'ERROR: allocation failure (e, f, t_work).'

   t_work = t

   call build_preconditioner ( &
      n_z_points, channel_count, fourier_values, n_fourier_components, &
      specular_fourier_component_index, d, e, f, t_work &
      )

   ifail = 0
   call solve_gmres_system ( &
      x, xx, y, n_z_points, channel_index_x, channel_index_y, &
      channel_count, specular_channel_index, fourier_values, &
      fourier_index_x, fourier_index_y, n_fourier_components, &
      a, b, c, d, e, f, p, s, t_work, convergence_eps, preconditioner_flag, ifail &
      )

   if (ifail == 1) p = -1.0_dp

   if (allocated(x)) deallocate(x)
   if (allocated(xx)) deallocate(xx)
   if (allocated(y)) deallocate(y)
   if (allocated(s)) deallocate(s)
   if (allocated(e)) deallocate(e)
   if (allocated(f)) deallocate(f)
   if (allocated(t_work)) deallocate(t_work)

   return
end subroutine run_preconditioned_gmres

subroutine build_preconditioner (n_z_points,channel_count, &
& fourier_values,n_fourier_components, &
& specular_fourier_component_index,channel_energy_z, &
& eigenvalues,preconditioner_factors,kinetic_matrix)
   use, intrinsic :: iso_fortran_env, only: real64
   implicit none
   integer, parameter :: dp = real64
   integer, parameter :: mmax = 550
   integer, intent(in) :: n_z_points, channel_count
   integer, intent(in) :: n_fourier_components, specular_fourier_component_index
!
! ------------------------------------------------------------------
! This subroutine constructs the matrix factors that are required
! for the block lower triangular preconditioner used in GMRES.
!
! t is the matrix from tshapes
! ------------------------------------------------------------------
!
   complex(dp), intent(in) :: fourier_values(n_z_points,n_fourier_components)
   real(dp), intent(in) :: channel_energy_z(channel_count)
   real(dp), intent(out) :: eigenvalues(n_z_points)
   real(dp), intent(out) :: preconditioner_factors(n_z_points,channel_count)
   real(dp), intent(inout) :: kinetic_matrix(n_z_points,n_z_points)
   real(dp) :: g(mmax)
   integer :: i, j, k, ierr
   external :: rs

! if m exceeds maximum size of m program terminates printing out precon 1
   if (n_z_points .gt. mmax) error stop 'precon 1'
!
   do k = 1,n_z_points
! real(x,kind=dp) transforms input into dp real
! for complex numbers it will return only real part
! This overwrites t to be H0 (as named in '90 Kohn paper)
      kinetic_matrix(k,k) = kinetic_matrix(k,k) + &
      & real(fourier_values(k, &
      & specular_fourier_component_index),kind=dp)
   enddo

! get eigenvalues [e] and eigenvectors [ overwrite them on t] of t,
! which is m x m symmetric real matrix. f is temporary storage array
   call rs (n_z_points,n_z_points,kinetic_matrix,eigenvalues, &
   & kinetic_matrix,preconditioner_factors,ierr)
! if ierr != 0 program terminates printing out precon 2, meaning rs failed
   if (ierr .ne. 0) error stop 'precon 2'

   do j = 1,channel_count

      do k = 1,n_z_points
         g(k) = kinetic_matrix(n_z_points,k) &
         & /(channel_energy_z(j)+eigenvalues(k))
         preconditioner_factors(k,j) = 0.0d0
      enddo

      do i = 1,n_z_points
         do k = 1,n_z_points
            preconditioner_factors(k,j) = &
            & preconditioner_factors(k,j) + kinetic_matrix(k,i)*g(i)
         enddo
      enddo

   enddo
   return
end subroutine build_preconditioner

subroutine solve_gmres_system (x,xx,y,n_z_points, &
& channel_index_x,channel_index_y, &
& channel_count,specular_channel_index, &
& fourier_values,fourier_index_x,fourier_index_y, &
& n_fourier_components, &
& a,b,c,d,e,f,p,s,t,convergence_eps, &
& preconditioner_flag,ifail)
   use, intrinsic :: iso_fortran_env, only: real64
   implicit none
!
! -----------------------------------------------------------------
! Complex Generalised Minimal Residual Algorithm (GMRES)
! This version written by DEM, 6/12/94
! -----------------------------------------------------------------
!
   integer, parameter :: dp = real64
   integer, intent(in) :: n_z_points, channel_count, specular_channel_index
   integer, intent(in) :: n_fourier_components, preconditioner_flag
   integer, intent(out) :: ifail
   integer, parameter :: l = 2000
   integer, intent(in) :: channel_index_x(channel_count), channel_index_y(channel_count)
   integer, intent(in) :: fourier_index_x(n_fourier_components)
   integer, intent(in) :: fourier_index_y(n_fourier_components)
   complex(dp), intent(out) :: x(n_z_points*channel_count)
   complex(dp), intent(out) :: xx(n_z_points*channel_count,l+1)
   complex(dp), intent(inout) :: y(n_z_points*channel_count)
   complex(dp), intent(in) :: fourier_values(n_z_points,n_fourier_components)
   complex(dp), intent(in) :: a(channel_count), b(channel_count), c(channel_count)
   complex(dp), intent(out) :: s(channel_count)
   real(dp), intent(in) :: d(channel_count), e(n_z_points)
   real(dp), intent(in) :: f(n_z_points,channel_count)
   real(dp), intent(out) :: p(channel_count)
   real(dp), intent(in) :: t(n_z_points,n_z_points)
   real(dp), intent(in) :: convergence_eps

   integer :: alloc_status
   integer :: mn, i, j, k, kount, kconv, kk
   real(dp) :: xnorm, unit, diff, pj
!
! NB:
! This subroutine implements a preconditioned version of GMRES(l).
! The rate of convergence can be improved (at the expense of a
! greater disk space requirement and more cpu time per iteration)
! by increasing the following parameter:
!
! parameter (l = 100)  increased l to 200, 22/4/1999, APJ
   complex(dp), allocatable :: h(:,:), g(:), z(:)
   complex(dp), allocatable :: co(:), si(:)
   complex(dp) temp
!
! store x matrices in xx rather than write to disk.
!
   allocate(h(l+1,l+1), g(l+1), z(l+1), co(l+1), si(l+1), stat=alloc_status)
   if (alloc_status /= 0) error stop 'ERROR: allocation failure (h, g, z, co, si).'
!
! Setup for GMRES(l):
!
   ifail = 0
   mn = n_z_points*channel_count
   do i = 1,mn
      x(i) = (0.0d0,0.0d0)
   enddo

!
! Initial step:
!
   kount = 0
   xx(1:mn,1)=x(1:mn)
   do i = 1,mn
      y(i) = x(i)
   enddo
   call apply_upper_block (x,n_z_points,channel_index_x, &
   & channel_index_y,channel_count,fourier_values, &
   & fourier_index_x,fourier_index_y,n_fourier_components)
   do i = 1,mn
      x(i) = -x(i)
   enddo
   x(n_z_points*specular_channel_index) = &
   & b(specular_channel_index)+x(n_z_points*specular_channel_index)
   call solve_lower_block (x,n_z_points,channel_index_x, &
   & channel_index_y,channel_count,fourier_values, &
   & fourier_index_x,fourier_index_y,n_fourier_components, &
   & c,d,e,f,t)
   do i = 1,mn
      x(i) = x(i)-y(i)
   enddo
   if (preconditioner_flag .eq. 1) then
      do i = 1,mn
         y(i) = x(i)
      enddo
      call apply_upper_block (x,n_z_points,channel_index_x, &
      & channel_index_y,channel_count,fourier_values, &
      & fourier_index_x,fourier_index_y,n_fourier_components)
      call solve_lower_block (x,n_z_points,channel_index_x, &
      & channel_index_y,channel_count,fourier_values, &
      & fourier_index_x,fourier_index_y,n_fourier_components, &
      & c,d,e,f,t)
      do i = 1,mn
         x(i) = y(i)-x(i)
      enddo
   endif
   xnorm = 0.0d0
   do i = 1,mn
      xnorm = xnorm + real(conjg(x(i))*x(i),kind=dp)
   enddo
   xnorm = sqrt(xnorm)
   g(1) = xnorm
!
! Generic recursion: (you mean iteration?)
!
   kconv = 0
   do j = 1,channel_count
      p(j) = 0.0d0
   enddo
   do k = 1,l
      kount = kount+1
      do i = 1,mn
         x(i) = x(i)/xnorm
      enddo
      xx(1:mn,k+1)=x(1:mn)
      do i = 1,mn
         y(i) = x(i)
      enddo
      call apply_upper_block (x,n_z_points,channel_index_x, &
      & channel_index_y,channel_count,fourier_values, &
      & fourier_index_x,fourier_index_y,n_fourier_components)
      call solve_lower_block (x,n_z_points,channel_index_x, &
      & channel_index_y,channel_count,fourier_values, &
      & fourier_index_x,fourier_index_y,n_fourier_components, &
      & c,d,e,f,t)
      do i = 1,mn
         x(i) = y(i)+x(i)
      enddo
      if (preconditioner_flag .eq. 1) then
         do i = 1,mn
            y(i) = x(i)
         enddo
         call apply_upper_block (x,n_z_points,channel_index_x, &
         & channel_index_y,channel_count,fourier_values, &
         & fourier_index_x,fourier_index_y,n_fourier_components)
         call solve_lower_block (x,n_z_points,channel_index_x, &
         & channel_index_y,channel_count,fourier_values, &
         & fourier_index_x,fourier_index_y,n_fourier_components, &
         & c,d,e,f,t)
         do i = 1,mn
            x(i) = y(i)-x(i)
         enddo
      endif
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
      s(specular_channel_index) = &
      & a(specular_channel_index)+s(specular_channel_index)
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
!
! Convergence test:
!
      unit = 0.0d0
      diff = 0.0d0
      do j = 1,channel_count
         pj = real(conjg(s(j))*s(j),kind=dp)
         unit = unit+pj
         diff = max(diff,abs(pj-p(j)))
         p(j) = pj
      enddo
      diff = max(diff,abs(unit-1.0d0))
      if (diff .lt. convergence_eps) then
         kconv = kconv+1
      else
         kconv = 0
      endif
      kk = k
      if (kconv.eq.3 .or. xnorm.eq.0.0d0) exit
   enddo
!
! back substitution for x:
!
   x(1:mn)=xx(1:mn,1)
   do j = 1,kk
      y(1:mn)=xx(1:mn,j+1)
      do i = 1,mn
         x(i) = x(i)+y(i)*z(j)
      enddo
   enddo
!
! all done?
!
   if (kconv.lt.3 .and. xnorm.gt.0.0d0) then
      ifail=1
   endif
!
! yes!
!
   if (allocated(h)) deallocate(h)
   if (allocated(g)) deallocate(g)
   if (allocated(z)) deallocate(z)
   if (allocated(co)) deallocate(co)
   if (allocated(si)) deallocate(si)
   return
end subroutine solve_gmres_system

subroutine apply_upper_block (state_vector,n_z_points, &
& channel_index_x,channel_index_y,channel_count, &
& fourier_values,fourier_index_x,fourier_index_y, &
& n_fourier_components)
   use, intrinsic :: iso_fortran_env, only: real64
   implicit none
!
! ----------------------------------------------------------
! This subroutine performs the block upper triangular
! matrix multiplication y = U*x, where A = L+U.
! The result y is overwritten on x on return.
! ----------------------------------------------------------
!
   integer, parameter :: dp = real64
   integer, intent(in) :: n_z_points, channel_count, n_fourier_components
   complex(dp), intent(inout) :: state_vector(n_z_points,channel_count)
   complex(dp), intent(in) :: fourier_values(n_z_points,n_fourier_components)
   integer, intent(in) :: channel_index_x(channel_count), channel_index_y(channel_count)
   integer, intent(in) :: fourier_index_x(n_fourier_components)
   integer, intent(in) :: fourier_index_y(n_fourier_components)
   integer :: i, j, k, l
!
   do j = 1,channel_count
      do k = 1,n_z_points
         state_vector(k,j) = (0.0d0,0.0d0)
      enddo
      do i = j+1,channel_count
         do l = 1,n_fourier_components
            if (channel_index_x(i) + fourier_index_x(l) .eq. channel_index_x(j) .and. &
            & channel_index_y(i) + fourier_index_y(l) .eq. channel_index_y(j)) then
               do k = 1,n_z_points
                  state_vector(k,j) = state_vector(k,j) + &
                  & fourier_values(k,l)*state_vector(k,i)
               enddo
            end if
         enddo
      enddo
   enddo
   return
end subroutine apply_upper_block

subroutine solve_lower_block (state_vector,n_z_points, &
& channel_index_x,channel_index_y,channel_count, &
& fourier_values,fourier_index_x,fourier_index_y, &
& n_fourier_components,wave_c,channel_energy_z, &
& eigenvalues,preconditioner_factors,kinetic_matrix)
   use, intrinsic :: iso_fortran_env, only: real64
   implicit none
   integer, parameter :: dp = real64
   integer, parameter :: mmax = 550
!
! ----------------------------------------------------------
! This subroutine solves the block lower triangular
! linear equation L*y = x, where A = L+U.
! The result y is overwritten on x on return.
! ----------------------------------------------------------
!
   integer, intent(in) :: n_z_points, channel_count, n_fourier_components
   complex(dp), intent(inout) :: state_vector(n_z_points,channel_count)
   complex(dp), intent(in) :: fourier_values(n_z_points,n_fourier_components)
   complex(dp), intent(in) :: wave_c(channel_count)
   integer, intent(in) :: channel_index_x(channel_count), channel_index_y(channel_count)
   integer, intent(in) :: fourier_index_x(n_fourier_components)
   integer, intent(in) :: fourier_index_y(n_fourier_components)
   real(dp), intent(in) :: channel_energy_z(channel_count), eigenvalues(n_z_points)
   real(dp), intent(in) :: preconditioner_factors(n_z_points,channel_count)
   real(dp), intent(in) :: kinetic_matrix(n_z_points,n_z_points)
   integer :: i, j, k, l
!

!parameter (mmax = 200)
   complex(dp) y(mmax), fac
   if (n_z_points .gt. mmax) error stop 'lower 1'
!
   do j = 1,channel_count
      do i = 1,j-1
         do l = 1,n_fourier_components
            if (channel_index_x(i) + fourier_index_x(l) .eq. channel_index_x(j) .and. &
            & channel_index_y(i) + fourier_index_y(l) .eq. channel_index_y(j)) then
               do k = 1,n_z_points
                  state_vector(k,j) = state_vector(k,j) &
                  & - fourier_values(k,l)*state_vector(k,i)
               enddo
            end if
         enddo
      enddo
      do k = 1,n_z_points
         y(k) = (0.0d0,0.0d0)
         do l = 1,n_z_points
            y(k) = y(k)+state_vector(l,j)*kinetic_matrix(l,k)
         enddo
         y(k) = y(k)/(channel_energy_z(j)+eigenvalues(k))
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
         state_vector(k,j) = state_vector(k,j) + &
         & fac*preconditioner_factors(k,j)
      enddo
   enddo
   return
end subroutine solve_lower_block

subroutine compute_complex_givens_rotation (input_a,input_b,c,s)
   use, intrinsic :: iso_fortran_env, only: real64
   implicit none
   integer, parameter :: dp = real64
   complex(dp), intent(inout) :: input_a, input_b
   complex(dp), intent(out) :: c, s
   real(dp) :: scale, r
   complex(dp) :: rho, z
!
! -----------------------------------------------------------------
! This subroutine constructs and performs a complex Givens rotation
! -----------------------------------------------------------------
!
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
   if (abs(input_b) .ge. abs(input_a) .and. abs(c) .ne. 0.0d0) &
   & z = 1.0d0/c
   input_a = r
   input_b = z
   return
end subroutine compute_complex_givens_rotation
