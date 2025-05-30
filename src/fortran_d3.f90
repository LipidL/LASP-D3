program test_d3
    use, intrinsic :: iso_c_binding
    implicit none
    
    ! External function interface declarations
    interface
        subroutine init_params() bind(c, name='init_params')
            use, intrinsic :: iso_c_binding
            implicit none
        end subroutine init_params

        subroutine compute_dispersion_energy(atoms, elements, num_atoms, cell, &
                cutoff_radius, CN_cutoff_radius, max_neighbors, &
                energy, force, &
                stress) bind(c, name='compute_dispersion_energy')
            use, intrinsic :: iso_c_binding
            implicit none
            
            real(c_float), intent(in) :: atoms(3,*)
            integer(c_int16_t), intent(in) :: elements(*)
            integer(c_int64_t), value :: num_atoms
            real(c_float), intent(in) :: cell(3,3)
            real(c_float), value :: cutoff_radius
            real(c_float), value :: CN_cutoff_radius
            integer(c_int64_t), value :: max_neighbors
            real(c_float), intent(out) :: energy
            real(c_float), intent(out) :: force(*)
            real(c_float), intent(out) :: stress(9)
        end subroutine compute_dispersion_energy

        function init_d3_handle(elements, max_length, cutoff_radius, &
            coordination_number_cutoff, max_neighbors) bind(c, name='init_d3_handle')
            use, intrinsic :: iso_c_binding
            implicit none
            
            integer(c_int16_t), intent(in) :: elements(*)
            integer(c_int64_t), value :: max_length
            real(c_float), value :: cutoff_radius
            real(c_float), value :: coordination_number_cutoff
            integer(c_int64_t), value :: max_neighbors
            type(c_ptr) :: init_d3_handle
        end function init_d3_handle

        subroutine set_atoms(handle, coords, elements, length) bind(c, name='set_atoms')
            use, intrinsic :: iso_c_binding
            implicit none
            
            type(c_ptr), value :: handle
            real(c_float), intent(in) :: coords(*)
            integer(c_int16_t), intent(in) :: elements(*)
            integer(c_int64_t), value :: length
        end subroutine set_atoms

        subroutine set_cell(handle, cell) bind(c, name='set_cell')
            use, intrinsic :: iso_c_binding
            implicit none
            
            type(c_ptr), value :: handle
            real(c_float), intent(in) :: cell(3,3)
        end subroutine set_cell

        subroutine free_d3_handle(handle) bind(c, name='free_d3_handle')
            use, intrinsic :: iso_c_binding
            implicit none
            
            type(c_ptr), value :: handle
        end subroutine free_d3_handle

        subroutine clear_d3_handle(handle) bind(c, name='clear_d3_handle')
            use, intrinsic :: iso_c_binding
            implicit none
            
            type(c_ptr), value :: handle
        end subroutine clear_d3_handle

        subroutine compute_dispersion_energy_from_handle(handle, energy, force, stress) &
            bind(c, name='compute_dispersion_energy_from_handle')
            use, intrinsic :: iso_c_binding
            implicit none
            
            type(c_ptr), value :: handle
            real(c_float), intent(out) :: energy
            real(c_float), intent(out) :: force(*)
            real(c_float), intent(out) :: stress(9)
        end subroutine compute_dispersion_energy_from_handle
    end interface

    ! Declarations
    integer(c_int16_t), parameter :: num_atoms = 10
    real(c_float) :: atoms(3, num_atoms)
    integer(c_int16_t) :: elements(num_atoms)
    real(c_float) :: cell(3, 3)
    real(c_float) :: energy
    real(c_float), allocatable :: force(:)
    real(c_float) :: stress(9)
    real(c_float) :: cutoff_radius, CN_cutoff_radius
    real(c_float) :: angstrom_to_bohr
    real(c_float) :: force_sum(3)
    integer :: i, j
    
    ! Convert Angstrom to Bohr
    angstrom_to_bohr = 1.0_c_float/0.52917726_c_float
    
    ! Initialize atom positions (stored in column-major order in Fortran)
    atoms(:, 1) = [5.1372_c_float, 5.5512_c_float, 10.1047_c_float]
    atoms(:, 2) = [4.5169_c_float, 6.1365_c_float, 11.3604_c_float]
    atoms(:, 3) = [6.1937_c_float, 4.4752_c_float, 10.2703_c_float]
    atoms(:, 4) = [4.7872_c_float, 5.9358_c_float, 8.9937_c_float]
    atoms(:, 5) = [6.7474_c_float, 4.3475_c_float, 9.3339_c_float]
    atoms(:, 6) = [5.6975_c_float, 3.5214_c_float, 10.5181_c_float]
    atoms(:, 7) = [6.8870_c_float, 4.7006_c_float, 11.0939_c_float]
    atoms(:, 8) = [4.8579_c_float, 5.6442_c_float, 12.2774_c_float]
    atoms(:, 9) = [3.4204_c_float, 6.0677_c_float, 11.2935_c_float]
    atoms(:, 10) = [4.7678_c_float, 7.2075_c_float, 11.4098_c_float]
    
    ! Convert positions from Angstrom to Bohr
    do i = 1, num_atoms
        atoms(:, i) = atoms(:, i) * angstrom_to_bohr
    end do
    
    ! Define atomic numbers
    elements = [6, 6, 6, 8, 1, 1, 1, 1, 1, 1]
    
    ! Set up cell
    cell = 0.0_c_float
    cell(1, 1) = 20.0_c_float * angstrom_to_bohr
    cell(2, 2) = 20.0_c_float * angstrom_to_bohr
    cell(3, 3) = 20.0_c_float * angstrom_to_bohr
    
    ! Set cutoff radii
    CN_cutoff_radius = 40.0_c_float
    cutoff_radius = 94.8683_c_float
    
    ! Allocate force array (3 components per atom)
    allocate(force(3 * num_atoms))
    
    ! Initialize parameters
    print *, "Computing dispersion energy for", num_atoms, "atoms..."
    call init_params()
    print *, "Computing dispersion energy..."
    
    ! Call the compute_dispersion_energy function
    call compute_dispersion_energy(atoms, elements, int(num_atoms, c_int64_t), cell, &
                                  cutoff_radius, CN_cutoff_radius, int(5000, c_int64_t), &
                                  energy, force, stress)
    
    ! Print results
    print '(a,f12.6,a)', "energy: ", energy, " eV"
    
    ! Initialize force sum
    force_sum = 0.0_c_float
    
    ! Print forces and calculate sum
    do i = 1, num_atoms
        print '(a,i0,a,3f15.13)', "force[", i-1, "]: ", &
              force(3*(i-1)+1), force(3*(i-1)+2), force(3*(i-1)+3)
        
        force_sum(1) = force_sum(1) + force(3*(i-1)+1)
        force_sum(2) = force_sum(2) + force(3*(i-1)+2)
        force_sum(3) = force_sum(3) + force(3*(i-1)+3)
    end do
    
    ! Print force sum
    print '(a,3f15.13)', "force sum: ", force_sum
    
    ! Print stress tensor
    do i = 1, 3
        do j = 1, 3
            print '(a,i0,a,i0,a,f15.13)', "stress[", i-1, "][", j-1, "]: ", stress((i-1)*3 + j)
        end do
    end do
    
    ! Clean up
    deallocate(force)
    
end program test_d3