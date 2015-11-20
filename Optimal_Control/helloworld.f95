!!-------------helloworld.f95-------------------------------------------------!!
!
!              hello world.f59
!
! Purpose: To check out some fortran!
!
!!----------------------------------------------------------------------------!!

program helloworld
      !! defninitons
      character :: tab = char(9)
      integer   :: i = 10
      real*8    :: variable

      !! formatting / opeining
  300 format(A, A, F4.2)
      open(100, file = "output.dat")

      !! writing
      write(100,300) "hello world!", tab, 1.00

      !! executing functions (subroutines)
      call loop(i, variable)
      write(*,*) variable, i
end program

!! subroutines -- pure!
pure subroutine loop(i, variable)
      implicit none
      integer, intent(inout)  :: i
      integer              :: j
      real*8, intent(out)  :: variable

      !! looping
      do j=1,10
         variable = variable + real(j) * real(i) * 0.001
         i = i + 1
      end do

end subroutine
