!!------------Nelder_Mead.f95-------------------------------------------------!!
!
!             Nelder Mead -- Downhill simplex in fortran!
!
! Purpose: To implement a simple optimization scheme in Fortran!
!
!   Notes: the value array has 3 accessory elements for different things:
!              1. Centroid position
!              2. Contraction / Expansion
!              3. Reflection
!
!!----------------------------------------------------------------------------!!

program nelder

!!----------------------------------------------------------------------------!!
!  DEFINITIONS
!!----------------------------------------------------------------------------!!

      integer, parameter :: dim = 6
      integer :: min, max, i
      real*8, dimension(2, dim)  :: pos
      real*8, dimension(dim)     :: value
      real*8  :: x, y, alpha, beta, gamma

      call downhill

end program

!!----------------------------------------------------------------------------!!
! SUBROUTINES
!!----------------------------------------------------------------------------!!

!! The actual method
!! The nelder mead method 
subroutine downhill
      implicit none
      integer, parameter         :: dim = 6
      integer :: min, max, i, minsave, maxsave
      real*8, dimension(2, dim)  :: pos
      real*8, dimension(dim)     :: value
      real*8  :: x, y, alpha = 0.5, beta = 0.5, gamma = 0.5, dist, cutoff = 0.1
      real*8  :: xsave, ysave

      interface
          subroutine findval(pos, value, dim)
              real*8, dimension(:,:) :: pos
              real*8, dimension(:)   :: value
              integer                :: dim
          end subroutine findval
      end interface

      interface
          subroutine populate(pos, dim)
              real*8, dimension(:,:) :: pos
              integer                :: dim
          end subroutine populate
      end interface

      interface
          subroutine minmax(value, min, max)
              real*8, intent(in)   :: value(:)
              integer, intent(out) :: min, max
          end subroutine minmax
      end interface

      interface
          subroutine centroid(pos, min, max, dim)
              real*8, intent(inout)  :: pos(:,:)
              integer, intent(in)    :: min, max, dim
          end subroutine centroid
      end interface

      interface
          subroutine reflect(pos, min, dim, alpha)
              real*8, intent(inout)  :: pos(:,:)
              integer, intent(in)    :: min, dim
              real*8, intent(in)     :: alpha
          end subroutine reflect
      end interface

      interface
          subroutine contract(pos, min, dim, beta)
              real*8, intent(inout)  :: pos(:,:)
              integer, intent(in)    :: min, dim
              real*8, intent(in)     :: beta
          end subroutine contract
      end interface

      interface
          subroutine expand(pos, min, dim, gamma)
              real*8, intent(inout)  :: pos(:,:)
              integer, intent(in)    :: min, dim
              real*8, intent(in)     :: gamma
          end subroutine expand
      end interface

      interface
          subroutine contractall(pos, min, dim)
              real*8, intent(inout)  :: pos(:,:)
              integer, intent(in)    :: min, dim
          end subroutine contractall
      end interface


      !! initialize everything
      call populate(pos, dim)
      call findval(pos, value, dim)
      call minmax(value, min, max)
      call centroid(pos, min, max, dim)

      dist = sqrt((pos(1, min) - pos(1, max)) * (pos(1, min) - pos(1, max)) &
                  + (pos(2, min) - pos(2, max)) * (pos(2, min) - pos(2, max)))
      i = 0

      do while (dist > cutoff)
          write(*,*) i
          minsave = min
          xsave = pos(1, min)
          ysave = pos(2, min)
          call reflect(pos, min, dim, alpha)
          call findval(pos, value, dim)
          call minmax(value, min, max)

          if (minsave.EQ.max) then
              write(*,*) "expanding..."
              maxsave = max
              xsave = pos(1, max)
              ysave = pos(2, max)
              call expand(pos, max, dim, gamma)
              call findval(pos, value, dim)
              call minmax(value, min, max)

              if (maxsave.NE.max) then
                  pos(1, max) = xsave
                  pos(2, max) = ysave
              end if

          else if (minsave.EQ.min) then
              write(*,*) "contracting..."
              xsave = pos(1, min)
              ysave = pos(2, min)

              pos(1, min) = xsave
              pos(2, min) = ysave
              call contract(pos, min, dim, beta)

              if (minsave.EQ.min) then
                  call contractall(pos, max, dim)
              end if

          end if
          i = i + 1
      end do

end subroutine

!! Total contraction
pure subroutine contractall(pos, max, dim)
      implicit none
      real*8, dimension(:,:), intent(inout) :: pos
      integer, intent(in)                   :: max, dim
      integer                               :: i

      do i = 1, dim - 3
          if (i.NE.max) then
              pos(1,i) = (pos(1, i) + pos(1, max)) * 0.5
              pos(2,i) = (pos(2, i) + pos(2, max)) * 0.5
          end if
      end do

end subroutine

!! for Nelder Mead reflections
pure subroutine reflect(pos, min, dim, alpha)
      implicit none
      real*8, dimension(:,:), intent(inout) :: pos
      integer, intent(in)                   :: min, dim
      real*8, intent(in)                    :: alpha

      pos(1,min) = (1 + alpha) * pos(1,dim - 2) - alpha * pos(1, min)
      pos(2,min) = (1 + alpha) * pos(2,dim - 2) - alpha * pos(2, min)

end subroutine

!! for Nelder Mead contractions
pure subroutine contract(pos, min, dim, beta)
      implicit none
      real*8, dimension(:,:), intent(inout) :: pos
      integer, intent(in)                   :: min, dim
      real*8, intent(in)                    :: beta

      pos(1,min) = (1 - beta) * pos(1,dim - 2) + beta * pos(1, min)
      pos(2,min) = (1 - beta) * pos(2,dim - 2) + beta * pos(2, min)

end subroutine

!! for Nelder Mead expansions
pure subroutine expand(pos, min, dim, gamma)
      implicit none
      real*8, dimension(:,:), intent(inout) :: pos
      integer, intent(in)                   :: min, dim
      real*8, intent(in)                    :: gamma

      pos(1,min) = (1 - gamma) * pos(1,dim - 2) + gamma * pos(1, min)
      pos(2,min) = (1 - gamma) * pos(2,dim - 2) + gamma * pos(2, min)


end subroutine

!! Finds the minima and maxima
!! We need a better way to search through the initial elements in our array!
subroutine minmax(value, min, max)
      implicit none
      real*8, intent(in)   :: value(:)
      integer, intent(out) :: min, max
      integer              :: max_array(1), min_array(1)

      min_array = minloc(value)
      max_array = maxloc(value)

      min = min_array(1)
      max = max_array(1)
end subroutine

!! populates grid -- UNPURE, fix later!
subroutine populate(pos, dim)
      implicit none
      integer                :: dim
      real*8, dimension(:,:) :: pos
      integer                :: seed, i, x

      !!call init_random_seed()

      do i = 1,dim - 3
          call random_number(pos(1,i))
          call random_number(pos(2,i))
      end do
      
end subroutine

!! finds centroid position 
pure subroutine centroid(pos, min, max, dim)
      implicit none
      real*8, intent(inout)  :: pos(:,:)
      integer, intent(in)    :: min, max, dim
      integer                :: i

      pos(1, dim-2) = 0
      pos(2, dim-2) = 0

      do i = 1,dim - 2
          if (i.NE.max) then
             pos(1, dim-2) = pos(1, dim-2) + pos(1, i)
             pos(2, dim-2) = pos(2, dim-2) + pos(2, i)
          end if
      end do

      pos(1, dim - 2) = pos(1, dim - 2) / real(dim - 3)
      pos(1, dim - 2) = pos(2, dim - 2) / real(dim - 3)
      
end subroutine

!! minimum currently set to 0.5, should be found via Downhill Simplex!
subroutine findval(pos, value, dim)
      implicit none
      real*8,  dimension(:,:):: pos
      real*8,  dimension(:)  :: value
      real*8                 :: sourcex = 0.5, sourcey = 0.5
      integer                :: dim
      integer                :: i

      do i = 1, dim - 3
          value(i) = sqrt((pos(1,i) - sourcex) * (pos(1,i) - sourcex) &
                     + (pos(2,i) - sourcey) * (pos(2,i) - sourcey))
      end do

      do i = dim - 2, dim
          value(i) = 0
      end do

end subroutine
