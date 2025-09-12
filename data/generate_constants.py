#!/usr/bin/env python3
"""
Generate C header file for DFT-D3 reference data

This script reads the Fortran reference.f90 file and extracts the c6ab reference data
and reference_cn data to create a C header file with the c6ab_ref array 
in the format: c6ab_ref[103][103][7][7][3]

Each element c6ab_ref[i][j][iref][jref] contains:
- [0]: c6ab value 
- [1]: reference_cn value for element i, reference iref
- [2]: reference_cn value for element j, reference jref
"""

import re
import sys

ANGSTROM_TO_BOHR = 1.8897261249935897

def parse_fortran_reference_file(filename):
    """Parse the Fortran reference file to extract data"""
    
    # First, extract the reference_cn matrix (103 x 7)
    reference_cn = []
    
    # Extract the number_of_references array (103 elements)
    number_of_references = []
    
    # Extract c6ab data
    c6ab_data = []
    
    # Extract additional arrays
    r2r4_data = []  # sqrt_z_r4_over_r2
    r0ab_data = []  # vdwrad (triangular matrix)
    rcov_data = []  # covalent_rad_d3
    
    with open(filename, 'r') as f:
        content = f.read()
    
    # Extract number_of_references
    ref_pattern = r'number_of_references\(max_elem\)\s*=\s*\[\s*&\s*(.*?)\]'
    ref_match = re.search(ref_pattern, content, re.DOTALL)
    if ref_match:
        ref_data = ref_match.group(1)
        # Parse the numbers
        numbers = re.findall(r'(\d+)', ref_data)
        number_of_references = [int(n) for n in numbers]
        print(f"Extracted {len(number_of_references)} number_of_references values")
        if len(number_of_references) != 103:
            print(f"Warning: Expected 103 values, got {len(number_of_references)}")
    else:
        print("Warning: Could not find number_of_references data, using default values")
        number_of_references = [1] * 103
    
    # Extract reference_cn matrix
    cn_pattern = r'reference_cn\(max_ref,\s*max_elem\)\s*=\s*reshape\(\[\s*&\s*(.*?)\],&?\s*\[max_ref,\s*max_elem\]\)'
    cn_match = re.search(cn_pattern, content, re.DOTALL)
    if not cn_match:
        # Try alternative pattern with different spacing
        cn_pattern = r'reference_cn\(max_ref,\s*max_elem\)\s*=\s*reshape\(\[\s*&\s*(.*?)\s*\[max_ref,\s*max_elem\]\)'
        cn_match = re.search(cn_pattern, content, re.DOTALL)
    
    if cn_match:
        cn_data = cn_match.group(1)
        # Parse the floating point numbers
        numbers = re.findall(r'([+-]?\d+\.?\d*_wp)', cn_data)
        # Convert to float, removing _wp suffix
        cn_values = [float(n.replace('_wp', '')) for n in numbers]
        
        # The Fortran data is stored sequentially: 7 values per element, 103 elements
        # Each line in the Fortran file represents one element with its 7 reference CN values
        reference_cn = []
        for elem in range(103):  # max_elem
            elem_refs = []
            for ref in range(7):  # max_ref
                idx = elem * 7 + ref  # Element-major order as written in the file
                if idx < len(cn_values):
                    elem_refs.append(cn_values[idx])
                else:
                    elem_refs.append(-1.0)  # Default for missing values
            reference_cn.append(elem_refs)
        print(f"Extracted reference_cn matrix: {len(reference_cn)} x {len(reference_cn[0]) if reference_cn else 0} from {len(cn_values)} values")
    else:
        print("Warning: Could not find reference_cn data, using default values")
        reference_cn = [[-1.0] * 7 for _ in range(103)]
    
    # Extract r4_over_r2 data
    r4r2_pattern = r'r4_over_r2\(max_elem\)\s*=\s*\[\s*&\s*(.*?)\s*&\s*\]'
    r4r2_match = re.search(r4r2_pattern, content, re.DOTALL)
    if r4r2_match:
        r4r2_data_str = r4r2_match.group(1)
        numbers = re.findall(r'([+-]?\d+\.?\d*_wp)', r4r2_data_str)
        r4_over_r2_values = [float(n.replace('_wp', '')) for n in numbers]
        
        # Calculate sqrt_z_r4_over_r2 values (r2r4 in the C code)
        import math
        r2r4_data = []
        for i, r4r2_val in enumerate(r4_over_r2_values):
            if i < 118:  # max_elem for this array
                z = i + 1  # atomic number (1-based)
                sqrt_z = math.sqrt(z)
                r2r4_val = math.sqrt(0.5 * r4r2_val * sqrt_z)
                r2r4_data.append(r2r4_val)
        print(f"Extracted {len(r2r4_data)} r2r4 values")
    else:
        print("Warning: Could not find r4_over_r2 data")
        r2r4_data = [0.0] * 118
    
    # Extract vdwrad data (triangular matrix)
    vdw_pattern = r'vdwrad\(max_elem\*\(1\+max_elem\)/2\)\s*=\s*aatoau\s*\*\s*\[\s*&\s*(.*?)\s*&\s*\]'
    vdw_match = re.search(vdw_pattern, content, re.DOTALL)
    if vdw_match:
        vdw_data_str = vdw_match.group(1)
        numbers = re.findall(r'([+-]?\d+\.?\d*_wp)', vdw_data_str)
        r0ab_data = [float(n.replace('_wp', '')) * ANGSTROM_TO_BOHR for n in numbers]
        print(f"Extracted {len(r0ab_data)} vdwrad values")
    else:
        print("Warning: Could not find vdwrad data")
        r0ab_data = [0.0] * (103 * 104 // 2)
    
    # Extract covalent_rad_2009 data to calculate covalent_rad_d3
    cov_pattern = r'covalent_rad_2009\(max_elem\)\s*=\s*aatoau\s*\*\s*\[\s*&\s*(.*?)\s*\]'
    cov_match = re.search(cov_pattern, content, re.DOTALL)
    if cov_match:
        cov_data_str = cov_match.group(1)
        numbers = re.findall(r'([+-]?\d+\.?\d*_wp)', cov_data_str)
        covalent_rad_2009_values = [float(n.replace('_wp', '')) for n in numbers]
        
        # Calculate covalent_rad_d3 = 4/3 * covalent_rad_2009 (first 103 elements)
        rcov_data = [val * 4.0 / 3.0 * ANGSTROM_TO_BOHR for val in covalent_rad_2009_values[:103]]
        print(f"Extracted {len(rcov_data)} rcov values")
    else:
        print("Warning: Could not find covalent_rad_2009 data")
        rcov_data = [0.0] * 103
    
    # Extract c6ab data from all c6ab_view assignments
    c6ab_patterns = re.finditer(r'c6ab_view\((\d+):(\d+)\)\s*=\s*\[\s*&\s*(.*?)\s*&\s*\]', content, re.DOTALL)
    
    all_c6ab_values = [0.0] * 262444  # Total size
    
    for match in c6ab_patterns:
        start_idx = int(match.group(1)) - 1  # Convert to 0-based indexing
        end_idx = int(match.group(2)) - 1
        data_str = match.group(3)
        
        # Extract floating point numbers
        numbers = re.findall(r'([+-]?\d+\.?\d*_wp)', data_str)
        values = [float(n.replace('_wp', '')) for n in numbers]
        assert (end_idx - start_idx + 1) == len(values), f"Expected {end_idx - start_idx + 1} values, got {len(values)}"
        
        # Fill the array
        for i, val in enumerate(values):
            if start_idx + i < len(all_c6ab_values):
                all_c6ab_values[start_idx + i] = val
    
    print(f"Extracted {len(all_c6ab_values)} c6ab values")
    
    return number_of_references, reference_cn, all_c6ab_values, r2r4_data, r0ab_data, rcov_data

def get_c6_value(c6ab_data, iref, jref, ati, atj):
    """
    Get C6 value from flattened array using the same logic as Fortran get_c6 function
    
    Fortran indexing:
    if (ati > atj) then
        ic = atj + ati*(ati-1)/2
        c6 = reference_c6(iref, jref, ic)
    else
        ic = ati + atj*(atj-1)/2
        c6 = reference_c6(jref, iref, ic)
    """
    # Convert to 0-based indexing for Python
    ati_0 = ati - 1
    atj_0 = atj - 1
    iref_0 = iref - 1  
    jref_0 = jref - 1
    
    if ati > atj:
        ic = atj_0 + ati_0 * (ati_0 + 1) // 2  # Adjusted for 0-based
        # Array layout: reference_c6(max_ref, max_ref, npairs)
        # Fortran column-major: idx = iref + jref*max_ref + ic*max_ref*max_ref
        idx = iref_0 + jref_0 * 7 + ic * 7 * 7
    else:
        ic = ati_0 + atj_0 * (atj_0 + 1) // 2  # Adjusted for 0-based  
        # Swapped iref and jref
        idx = jref_0 + iref_0 * 7 + ic * 7 * 7
    assert idx >= 0 and idx < len(c6ab_data), f"Index {idx} out of bounds for c6ab_data of length {len(c6ab_data)}"

    if idx < len(c6ab_data):
        return c6ab_data[idx]
    else:
        return 0.0

def generate_c_header(number_of_references, reference_cn, c6ab_data, r2r4_data, r0ab_data, rcov_data, output_filename):
    """Generate the C header file"""
    
    with open(output_filename, 'w') as f:
        f.write("""/*
 * DFT-D3 Reference Data for C
 * 
 * This file contains the reference data for DFT-D3 dispersion corrections
 * converted from the Fortran source in simple-dftd3.
 * 
 * Data structure:
 * - c6ab_ref[i][j][iref][jref][3] where:
 *   - i, j: atomic numbers (0-based, 0-102 for H-Lr) 
 *   - iref, jref: reference state indices (0-6)
 *   - [0]: C6 coefficient value
 *   - [1]: reference coordination number for atom i in reference state iref
 *   - [2]: reference coordination number for atom j in reference state jref
 * 
 * Additional arrays:
 * - r2r4[118]: r4/r2 expectation values 
 * - r0ab[103][103]: van der Waals radii (symmetric matrix)
 * - rcov[103]: covalent radii 
 * 
 * Generated automatically from parameters.f90
 */

#ifndef DFTD3_REFERENCE_H
#define DFTD3_REFERENCE_H


#define NUM_ELEMENTS_R2R4 118
#define NUM_C6AB_ENTRIES 3

/* Number of reference states for each element (0-based indexing) */
static const int number_of_references[NUM_ELEMENTS] = {
""")
        
        # Write number_of_references array
        for i in range(0, len(number_of_references), 8):
            line_vals = number_of_references[i:i+8]
            line_str = "    " + ", ".join(f"{val:2d}" for val in line_vals)
            if i + 8 < len(number_of_references):
                line_str += ","
            f.write(line_str + "\n")
        
        f.write("};\n\n")
        
        f.write("""/* Reference coordination numbers [element][reference_state] */
static const double reference_cn[NUM_ELEMENTS][NUM_REF_C6] = {
""")
        
        # Write reference_cn matrix  
        for i, elem_refs in enumerate(reference_cn):
            f.write("    {")
            vals_str = ", ".join(f"{val:10.8f}" for val in elem_refs)
            f.write(vals_str)
            f.write("}")
            if i < len(reference_cn) - 1:
                f.write(",")
            f.write("\n")
            
        f.write("};\n\n")
        
        # Write r2r4 array
        f.write("""/* R4/R2 expectation values for elements 1-118 */
static const double r2r4[NUM_ELEMENTS_R2R4] = {
""")
        for i in range(0, len(r2r4_data), 4):
            line_vals = r2r4_data[i:i+4]
            line_str = "    " + ", ".join(f"{val:10.8f}" for val in line_vals)
            if i + 4 < len(r2r4_data):
                line_str += ","
            f.write(line_str + "\n")
        f.write("};\n\n")
        
        # Write rcov array
        f.write("""/* Covalent radii for elements 1-103 */
static const double rcov[NUM_ELEMENTS] = {
""")
        for i in range(0, len(rcov_data), 4):
            line_vals = rcov_data[i:i+4]
            line_str = "    " + ", ".join(f"{val:10.8f}" for val in line_vals)
            if i + 4 < len(rcov_data):
                line_str += ","
            f.write(line_str + "\n")
        f.write("};\n\n")
        
        # Convert triangular r0ab_data to symmetric matrix
        f.write("""/* Van der Waals radii [element1][element2] (symmetric matrix) */
static const double r0ab[NUM_ELEMENTS][NUM_ELEMENTS] = {
""")
        for i in range(103):
            f.write("    {")
            row_vals = []
            for j in range(103):
                # Get value from triangular matrix
                if i <= j:
                    # Upper triangle: index = i + j*(j+1)/2
                    idx = i + j * (j + 1) // 2
                else:
                    # Lower triangle: index = j + i*(i+1)/2 (symmetric)
                    idx = j + i * (i + 1) // 2
                
                if idx < len(r0ab_data):
                    row_vals.append(r0ab_data[idx])
                else:
                    row_vals.append(0.0)

            vals_str = ", ".join(f"{val:10.8f}" for val in row_vals)
            f.write(vals_str)
            f.write("}")
            if i < 102:
                f.write(",")
            f.write("\n")
        f.write("};\n\n")
        
        f.write("""/* C6 reference data [elem1][elem2][ref1][ref2][3] 
 * [0] = C6 value, [1] = CN for elem1, [2] = CN for elem2 */
static const double c6ab_ref[NUM_ELEMENTS][NUM_ELEMENTS][NUM_REF_C6][NUM_REF_C6][NUM_C6AB_ENTRIES] = {
""")
        
        # Generate the 5D array
        for i in range(103):  # elem1
            f.write("  {\n")
            for j in range(103):  # elem2  
                f.write("    {\n")
                for iref in range(7):  # ref1
                    f.write("      {\n")
                    for jref in range(7):  # ref2
                        # Get C6 value using symmetry logic
                        c6_val = get_c6_value(c6ab_data, iref+1, jref+1, i+1, j+1)
                        
                        # Get reference CN values - these should always correspond to 
                        # the element and reference indices in the array position,
                        # regardless of any C6 symmetry considerations
                        cn1 = -1.0  # CN for element i, reference iref
                        cn2 = -1.0  # CN for element j, reference jref
                        
                        if i < len(reference_cn) and iref < len(reference_cn[i]):
                            cn1 = reference_cn[i][iref] if iref < number_of_references[i] else -1.0
                        if j < len(reference_cn) and jref < len(reference_cn[j]):
                            cn2 = reference_cn[j][jref] if jref < number_of_references[j] else -1.0
                        if (cn1 == -1.0 or cn2 == -1.0):
                            cn1 = -1.0
                            cn2 = -1.0
                            c6_val = -1.0
                        f.write(f"        {{{c6_val:12.8f}, {cn1:10.8f}, {cn2:10.8f}}}")
                        if jref < 6:
                            f.write(",")
                        f.write("\n")
                    f.write("      }")
                    if iref < 6:
                        f.write(",")
                    f.write("\n")
                f.write("    }")
                if j < 102:
                    f.write(",")
                f.write("\n")
            f.write("  }")
            if i < 102:
                f.write(",")
            f.write("\n")
            
            # Progress indicator
            if (i + 1) % 10 == 0:
                print(f"Generated data for {i+1}/103 elements")
        
        f.write("};\n\n")
        f.write("inline void init_params(){}\n\n")
        f.write("#endif /* DFTD3_REFERENCE_H */\n")

def main():
    if len(sys.argv) != 2:
        print("Usage: python generate_c6ab_header.py <path_to_parameters.f90>")
        sys.exit(1)
    
    fortran_file = sys.argv[1]
    output_file = "../src/gen_constants.h"
    
    print(f"Parsing Fortran file: {fortran_file}")
    number_of_references, reference_cn, c6ab_data, r2r4_data, r0ab_data, rcov_data = parse_fortran_reference_file(fortran_file)
    
    print(f"Generating C header file: {output_file}")
    generate_c_header(number_of_references, reference_cn, c6ab_data, r2r4_data, r0ab_data, rcov_data, output_file)
    
    print(f"Successfully generated {output_file}")
    print(f"Array sizes:")
    print(f"  c6ab_ref[{len(reference_cn)}][{len(reference_cn)}][{len(reference_cn[0])}][{len(reference_cn[0])}][3]")
    print(f"  r2r4[{len(r2r4_data)}]")
    print(f"  r0ab[{len(rcov_data)}][{len(rcov_data)}]")
    print(f"  rcov[{len(rcov_data)}]")

if __name__ == "__main__":
    main()
