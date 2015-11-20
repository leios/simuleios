!!------------Nelder_Mead.f95-------------------------------------------------!!
!
!             Nelder Mead -- Downhill simplex in fortran!
!
! Purpose: To implement a simple optimization scheme in Fortran!
!
!!----------------------------------------------------------------------------!!

program nelder

!!----------------------------------------------------------------------------!!
!  DEFINITIONS
!!----------------------------------------------------------------------------!!

      integer, parameter :: dim = 6
      integer :: min, max
      real*8  :: posx(dim), posy(dim), alpha, beta, gamma, value
      real*8  :: x, y

      call findval(0.5d+0, 0.5d+0, value)
      write(*,*) value

end program

!!----------------------------------------------------------------------------!!
! SUBROUTINES
!!----------------------------------------------------------------------------!!

!! for Nelder Mead reflections
pure subroutine reflect
end subroutine

!! for Nelder Mead contractions
pure subroutine contract
end subroutine

!! for Nelder Mead expansions
pure subroutine expand
end subroutine

!! populates grid -- UNPURE, fix later!
subroutine populate(posx, posy, dim)
      implicit none
      integer                :: dim
      real*8                 :: posx(:), posy(:)
      integer                :: seed, i, x

      !!call init_random_seed()

      do i = 1,dim - 3
          call random_number(posx(i))
          call random_number(posy(i))
      end do

      
end subroutine

!! finds centroid position
pure subroutine centroid(posx, posy, min, max, dim)
      implicit none
      real*8, intent(inout)  :: posx(:), posy(:)
      integer, intent(in)    :: min, max, dim
      integer                :: i

      posx(dim-2) = 0
      posy(dim-2) = 0

      do i = 1,dim - 2
          if (i.NE.max) then
             posx(dim-2) = posx(dim-2) + posx(i)
             posy(dim-2) = posy(dim-2) + posy(i)
          end if
      end do

      posx(dim - 2) = posx(dim - 2) / real(dim - 3)
      posy(dim - 2) = posy(dim - 2) / real(dim - 3)
      
end subroutine

!! minimum currently set to 0.5, should be found via Downhill Simplex!
subroutine findval(xval, yval, value)
      implicit none
      real*8, intent(in)  :: xval, yval
      real*8, intent(out) :: value
      real*8              :: sourcex = 0.5, sourcey = 0.5

      value = sqrt((xval - sourcex) * (xval - sourcex) &
              + (yval - sourcey) * (yval - sourcey))

end subroutine
