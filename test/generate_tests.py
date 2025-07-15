import os
import glob
import re
import numpy as np

ELEMENT_Z = {
    'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10,
    'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18,
    'K': 19, 'Ca': 20, 'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30,
    'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35, 'Kr': 36,
    'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40, 'Nb': 41, 'Mo': 42, 'Tc': 43, 'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48,
    'In': 49, 'Sn': 50, 'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54,
    'Cs': 55, 'Ba': 56, 'La': 57, 'Ce': 58, 'Pr': 59, 'Nd': 60, 'Pm': 61, 'Sm': 62, 'Eu': 63, 'Gd': 64, 'Tb': 65, 'Dy': 66, 'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70, 'Lu': 71,
    'Hf': 72, 'Ta': 73, 'W': 74, 'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78, 'Au': 79, 'Hg': 80,
    'Tl': 81, 'Pb': 82, 'Bi': 83, 'Po': 84, 'At': 85, 'Rn': 86,
    'Fr': 87, 'Ra': 88, 'Ac': 89, 'Th': 90, 'Pa': 91, 'U': 92, 'Np': 93, 'Pu': 94, 'Am': 95, 'Cm': 96, 'Bk': 97, 'Cf': 98, 'Es': 99, 'Fm': 100, 'Md': 101, 'No': 102, 'Lr': 103,
    'Rf': 104, 'Db': 105, 'Sg': 106, 'Bh': 107, 'Hs': 108, 'Mt': 109, 'Ds': 110, 'Rg': 111, 'Cn': 112, 'Nh': 113, 'Fl': 114, 'Mc': 115, 'Lv': 116, 'Ts': 117, 'Og': 118
}

def read_file_content(filepath) -> str:
    """Read file content as string"""
    with open(filepath, 'r') as f:
        return f.read().strip()

def parse_energy_data(content) -> float:
    """Parse energy content into a float"""
    regex = r"^\s*Edisp /kcal,au,eV:\s*[-+]?\d*\.?\d+\s+([-+]?\d*\.?\d+)"
    match = re.search(regex, content, re.MULTILINE)
    if match:
        return float(match.group(1))
    return 0.0

def parse_numeric_data(content, is_vector=False):
    """Parse numeric content into C++ compatible format"""
    lines = content.strip().split('\n')
    
    # Format for C++
    if is_vector:
        # Handle multi-line vector data (like forces or stress)
        vectors = []
        for line in lines:
            if not line.strip():
                continue
            values = line.split()
            vectors.append("{" + ", ".join(values) + "}")
        return "{" + ",\n          ".join(vectors) + "}"
    else:
        # Handle scalar data (like energy)
        try:
            return float(content.strip())
        except ValueError:
            # If not a simple number, return as raw string
            return f'R"({content})"'

def parse_poscar_data(content):
    """Parse POSCAR data into C++ compatible format"""
    lines = content.strip().split('\n')
    
    # lines[0] is comment, lines[1] is scaling factor (usually 1.0)
    
    # Cell vectors
    cell_lines = lines[2:5]
    cell = np.array([[float(x) for x in line.split()] for line in cell_lines])
    cell_cpp = "{{" + "}, {".join([", ".join(map(str, row)) for row in cell]) + "}}"

    # Elements and atom counts
    element_symbols = lines[5].split()
    atom_counts = [int(x) for x in lines[6].split()]
    
    elements_list = []
    for el, count in zip(element_symbols, atom_counts):
        elements_list.extend([ELEMENT_Z[el]] * count)
    elements_cpp = "{" + ", ".join(map(str, elements_list)) + "}"
    
    # Direct or Cartesian
    coord_type_line = lines[7]
    is_direct = coord_type_line.strip().lower().startswith('d')
    
    # Atom coordinates
    atom_lines = lines[8:8+sum(atom_counts)]
    atoms = np.array([[float(x) for x in line.split()[:3]] for line in atom_lines])
        
    if is_direct:
        atoms = np.dot(atoms, cell)

    atoms_cpp = "{\n          " + ",\n          ".join(["{" + ", ".join(map(str, row)) + "}" for row in atoms]) + "}"
    
    return {
        'cell': cell_cpp,
        'atoms': atoms_cpp,
        'elements': elements_cpp
    }


def generate_cpp_code(test_data):
    cpp_code = """// Auto-generated test configurations
#include <vector>
#include <string>
#include <array>

// Assuming real_t is defined as double or float
using real_t = double;

struct TestConfig {
    std::string test_name;
    real_t cell[3][3];
    std::vector<uint16_t> elements;
    std::vector<std::array<real_t, 3>> atoms;
    real_t energy;
    std::vector<std::array<real_t, 3>> force;
    std::vector<std::array<real_t, 3>> stress;
    std::string arc_content;
    std::string poscar_content;
};

std::vector<TestConfig> test_configs = {
"""
    
    for name, data in test_data.items():
        cpp_code += f"""    {{
        "{name}",  // test_name
        {data.get('cell', '{{{{0.0, 0.0, 0.0}}, {{0.0, 0.0, 0.0}}, {{0.0, 0.0, 0.0}}}}')}, // cell
        {data.get('elements', '{{}}')},  // elements
        {data.get('atoms', '{{}}')}, // atoms
        {data.get('energy', '0.0')},  // energy
        {data.get('force', '{{}}')},  // force
        {data.get('stress', '{{}}')},  // stress
        R"({data.get('arc', '')})",  // arc_content
        R"({data.get('poscar_content', '')})" // poscar_content
    }},
"""
    
    cpp_code += "};\n"
    return cpp_code

def main():
    # Find all base files
    arc_files = glob.glob("./standard_results/*.arc")
    
    # Group files by base name
    test_data = {}
    
    for arc_file in arc_files:
        base_name = os.path.basename(arc_file).replace(".arc", "")
        test_data[base_name] = {}
        
        # Read base arc file
        test_data[base_name]['arc'] = read_file_content(arc_file)
        
        # Read associated files
        if os.path.exists(f"{arc_file}.energy"):
            test_data[base_name]['energy'] = parse_energy_data(read_file_content(f"{arc_file}.energy"))
            
        if os.path.exists(f"{arc_file}.force"):
            test_data[base_name]['force'] = parse_numeric_data(read_file_content(f"{arc_file}.force"), is_vector=True)
            
        if os.path.exists(f"{arc_file}.poscar"):
            poscar_content = read_file_content(f"{arc_file}.poscar")
            test_data[base_name]['poscar_content'] = poscar_content
            poscar_parsed = parse_poscar_data(poscar_content)
            test_data[base_name].update(poscar_parsed)
            
        if os.path.exists(f"{arc_file}.stress"):
            test_data[base_name]['stress'] = parse_numeric_data(read_file_content(f"{arc_file}.stress"), is_vector=True)
    
    # Generate C++ file
    cpp_code = generate_cpp_code(test_data)
    
    # Write to file
    with open("./grand_test.h", 'w') as f:
        f.write(cpp_code)
    
    print(f"Generated grand_test.cpp with {len(test_data)} test configurations")

if __name__ == "__main__":
    main()