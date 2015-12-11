!!------------Nelder_Mead.f95-------------------------------------------------!!
!
!             Nelder Mead -- Downhill simplex in fortran!
!
! Purpose: To implement a simple optimization scheme in Fortran!
!
!   Notes: the value array has 3 accessory elements for different things:
!              1. Centroid position
!
!   ERROR: All of the points become equal after a few interations.
!          This is probably due to an index mismatch
!!----------------------------------------------------------------------------!!

program nelder

!!----------------------------------------------------------------------------!!
!  DEFINITIONS
!!----------------------------------------------------------------------------!!

      integer, parameter :: dim = 4
      integer :: min, max, i
      real*8, dimension(2, dim)  :: pos
      real*8, dimension(dim - 1)     :: value
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
      integer, parameter         :: dim = 4
      integer :: min, max, i, minsave, maxsave
      real*8, dimension(2, dim)  :: pos
      real*8, dimension(dim - 1)     :: value
      real*8  :: x, y, alpha = 0.5, beta = 0.5, gamma = 0.5, dist, cutoff = 0.00001
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
              real*8, intent(in)     :: beta
              integer, intent(in)    :: dim, min
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

      !write(*,*) pos(1,1), pos(1,2), pos(1,3), pos(1,4)
      !write(*,*) pos(2,1), pos(2,2), pos(2,3), pos(2,4)
      !write(*,*) min, max, dim

      i = 0

      dist = sqrt((pos(1,min)-pos(1,max)) * (pos(1,min)-pos(1,max)) &
                  + (pos(2,min)-pos(2,max)) * (pos(2,min)-pos(2,max)))

      !write(*,*) dist

      !! Nelder-Mead optimal control
      do while (dist > cutoff)
      !do while (i < 10)
          !! Reflection first
          write(*,*) i
          minsave = min
          xsave = pos(1, min)
          ysave = pos(2, min)
          call reflect(pos, min, dim, alpha)
          call findval(pos, value, dim)
          call minmax(value, min, max)
          call centroid(pos, min, max, dim)

          !! Expansion if the minimum value becomes the maximum
          if (minsave.EQ.max) then
              write(*,*) "expanding...", dist
             
              maxsave = max
              xsave = pos(1, max)
              ysave = pos(2, max)
              call expand(pos, max, dim, gamma)
              call findval(pos, value, dim)
              call minmax(value, min, max)
              call centroid(pos, min, max, dim)

              !! Setting things back, if expansion is worse than 
              !! standard reflection
              if (maxsave.NE.max) then
                  pos(1, maxsave) = xsave
                  pos(2, maxsave) = ysave
                  call findval(pos, value, dim)
                  call minmax(value, min, max)
              end if

          !! Contract from old position if minima value is still minima value
          else if (minsave.EQ.min) then
              !write(*,*) "contracting...", dist
              xsave = pos(1, min)
              ysave = pos(2, min)

              pos(1, min) = xsave
              pos(2, min) = ysave
              call contract(pos, min, dim, beta)
              call findval(pos, value, dim)
              call minmax(value, min, max)
              call centroid(pos, min, max, dim)

              !! if minima is still minima, contract everything towards max
              if (minsave.EQ.min) then
                  !write(*,*) "Contract all...", dist, min, max
                  !write(*,*) pos(1,1), pos(1,2), pos(1,3), pos(1,4)
                  !write(*,*) pos(2,:)
                  !write(*,*) pos
                  call contractall(pos, max, dim)
                  call findval(pos, value, dim)
                  call minmax(value, min, max)
                  call centroid(pos, min, max, dim)

              end if

          end if
          i = i + 1
          dist = sqrt((pos(1,min)-pos(1,max)) * (pos(1,min)-pos(1,max)) &
                      + (pos(2,min)-pos(2,max)) * (pos(2,min)-pos(2,max)))

      end do

      write(*,*) pos(1,1), pos(1,2), pos(1,3), pos(1,4)
      write(*,*) pos(2,1), pos(2,2), pos(2,3), pos(2,4)
      write(*,*) min, max, dim

end subroutine

!! Total contraction
pure subroutine contractall(pos, max, dim)
      implicit none
      real*8, dimension(:,:), intent(inout) :: pos
      integer, intent(in)                   :: max, dim
      integer                               :: i
      real*8                                :: xsum, ysum

      do i = 1, dim - 1
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

      pos(1,min) = (1 + alpha) * pos(1,dim) - alpha * pos(1, min)
      pos(2,min) = (1 + alpha) * pos(2,dim) - alpha * pos(2, min)

end subroutine

!! for Nelder Mead contractions
pure subroutine contract(pos, min, dim, beta)
      implicit none
      real*8, dimension(:,:), intent(inout) :: pos
      integer, intent(in)                   :: min, dim
      real*8, intent(in)                    :: beta

      pos(1,min) = (1 - beta) * pos(1,dim) + beta * pos(1, min)
      pos(2,min) = (1 - beta) * pos(2,dim) + beta * pos(2, min)

end subroutine

!! for Nelder Mead expansions
pure subroutine expand(pos, min, dim, gamma)
      implicit none
      real*8, dimension(:,:), intent(inout) :: pos
      integer, intent(in)                   :: min, dim
      real*8, intent(in)                    :: gamma

      pos(1,min) = (1 - gamma) * pos(1,dim) + gamma * pos(1, min)
      pos(2,min) = (1 - gamma) * pos(2,dim) + gamma * pos(2, min)


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

      do i = 1,dim -1
          call random_number(pos(1,i))
          pos(1,i) = (pos(1,i) - 0.5)* 10
          call random_number(pos(2,i))
          pos(2,i) = (pos(2,i) - 0.5) * 10
      end do
      
end subroutine

!! finds centroid position 
subroutine centroid(pos, min, max, dim)
      implicit none
      real*8, intent(inout)  :: pos(:,:)
      integer, intent(in)    :: min, max, dim
      integer                :: i
      real*8                 :: xsum, ysum

      pos(1, dim) = 0
      pos(2, dim) = 0
      xsum = 0
      ysum = 0

      do i = 1,dim - 1
          if (i.NE.min) then
             xsum = xsum + pos(1, i)
             ysum = ysum + pos(2, i)
          end if
      end do

      pos(1, dim) = xsum / real(dim - 2)
      pos(2, dim) = ysum / real(dim - 2)
      
end subroutine

!! minimum currently set to 0.5, should be found via Downhill Simplex!
subroutine findval(pos, value, dim)
      implicit none
      real*8,  dimension(:,:):: pos
      real*8,  dimension(:)  :: value
      real*8                 :: sourcex = 0.5, sourcey = 0.5
      integer                :: dim
      integer                :: i

      do i = 1, dim - 1
          value(i) = sqrt((pos(1,i) - sourcex) * (pos(1,i) - sourcex) &
                     + (pos(2,i) - sourcey) * (pos(2,i) - sourcey))
      end do

end subroutine
