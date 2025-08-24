#include <cmath>
#include <string>
#include <vector>

namespace periodic_table {
/**
 * @brief Represents a chemical element in the periodic table.
 */
template <typename T>
class Element {
   public:
    std::string name;            // Name of the element
    std::string symbol;          // Symbol of the element
    uint16_t atomic_number;      // Atomic number of the element
    T mass;                      // Atomic weight of the element
    T valence_radius;            // Valence radius of the element
    T covalent_radius;           // Covalent radius of the element
    uint16_t valence_electrons;  // Number of valence electrons
    uint16_t group;   // Group number of the element in the periodic table
    uint16_t period;  // Period number of the element in the periodic table
};

/**
 * @brief Represents the periodic table of elements.
 */
template <typename T>
class PeriodicTable {
   public:
   /**
    * @brief Constructs the periodic table and initializes it with elements.
    */
    PeriodicTable() {
        // Initialize the periodic table with some elements
        elements.push_back(Element<T>{"Hydrogen", "H", 1, 1.008, 0.53, 0.37, 1, 1, 1});
        elements.push_back(Element<T>{"Helium", "He", 2, 4.002602, 0.31, 0.28, 2, 18, 1});
        elements.push_back(Element<T>{"Lithium", "Li", 3, 6.94, 1.52, 1.23, 1, 1, 2});
        elements.push_back(Element<T>{"Beryllium", "Be", 4, 9.0122, 1.12, 0.90, 2, 2, 2});
        elements.push_back(Element<T>{"Boron", "B", 5, 10.81, 0.87, 0.82, 3, 13, 2});
        elements.push_back(Element<T>{"Carbon", "C", 6, 12.011, 0.77, 0.77, 4, 14, 2});
        elements.push_back(Element<T>{"Nitrogen", "N", 7, 14.007, 0.75, 0.70, 5, 15, 2});
        elements.push_back(Element<T>{"Oxygen", "O", 8, 15.999, 0.73, 0.66, 6, 16, 2});
        elements.push_back(Element<T>{"Fluorine", "F", 9, 18.998, 0.72, 0.64, 7, 17, 2});
        elements.push_back(Element<T>{"Neon", "Ne", 10, 20.180, 0.71, 0.62, 8, 18, 2});
        elements.push_back(Element<T>{"Sodium", "Na", 11, 22.990, 1.86, 1.54, 1, 1, 3});
        elements.push_back(Element<T>{"Magnesium", "Mg", 12, 24.305, 1.60, 1.36, 2, 2, 3});
        elements.push_back(Element<T>{"Aluminum", "Al", 13, 26.982, 1.43, 1.18, 3, 13, 3});
        elements.push_back(Element<T>{"Silicon", "Si", 14, 28.085, 1.18, 1.11, 4, 14, 3});
        elements.push_back(Element<T>{"Phosphorus", "P", 15, 30.974, 1.10, 1.07, 5, 15, 3});
        elements.push_back(Element<T>{"Sulfur", "S", 16, 32.06, 1.02, 1.02, 6, 16, 3});
        elements.push_back(Element<T>{"Chlorine", "Cl", 17, 35.45, 0.99, 0.99, 7, 17, 3});
        elements.push_back(Element<T>{"Argon", "Ar", 18, 39.948, 0.97, 0.96, 8, 18, 3});
        elements.push_back(Element<T>{"Potassium", "K", 19, 39.098, 2.75, 2.03, 1, 1, 4});
        elements.push_back(Element<T>{"Calcium", "Ca", 20, 40.078, 2.31, 1.74, 2, 2, 4});
        elements.push_back(Element<T>{"Scandium", "Sc", 21, 44.956, 2.00, 1.44, 3, 3, 4});
        elements.push_back(Element<T>{"Titanium", "Ti", 22, 47.867, 1.76, 1.32, 4, 4, 4});
        elements.push_back(Element<T>{"Vanadium", "V", 23, 50.941, 1.53, 1.22, 5, 5, 4});
        elements.push_back(Element<T>{"Chromium", "Cr", 24, 51.996, 1.39, 1.18, 6, 6, 4});
        elements.push_back(Element<T>{"Manganese", "Mn", 25, 54.938, 1.27, 1.15, 7, 7, 4});
        elements.push_back(Element<T>{"Iron", "Fe", 26, 55.845, 1.25, 1.17, 8, 8, 4});
        elements.push_back(Element<T>{"Cobalt", "Co", 27, 58.933, 1.24, 1.16, 9, 9, 4});
        elements.push_back(Element<T>{"Nickel", "Ni", 28, 58.693, 1.24, 1.15, 10, 10, 4});
        elements.push_back(Element<T>{"Copper", "Cu", 29, 63.546, 1.28, 1.17, 11, 11, 4});
        elements.push_back(Element<T>{"Zinc", "Zn", 30, 65.38, 1.31, 1.22, 12, 12, 4});
        elements.push_back(Element<T>{"Gallium", "Ga", 31, 69.723, 1.36, 1.22, 3, 13, 4});
        elements.push_back(Element<T>{"Germanium", "Ge", 32, 72.63, 1.40, 1.20, 4, 14, 4});
        elements.push_back(Element<T>{"Arsenic", "As", 33, 74.922, 1.38, 1.19, 5, 15, 4});
        elements.push_back(Element<T>{"Selenium", "Se", 34, 78.971, 1.40, 1.20, 6, 16, 4});
        elements.push_back(Element<T>{"Bromine", "Br", 35, 79.904, 1.38, 1.19, 7, 17, 4});
        elements.push_back(Element<T>{"Krypton", "Kr", 36, 83.798, 1.37, 1.18, 8, 18, 4});
        elements.push_back(Element<T>{"Rubidium", "Rb", 37, 85.467, 2.97, 2.20, 1, 1, 5});
        elements.push_back(Element<T>{"Strontium", "Sr", 38, 87.62, 2.49, 1.95, 2, 2, 5});
        elements.push_back(Element<T>{"Yttrium", "Y", 39, 88.906, 2.25, 1.62, 3, 3, 5});
        elements.push_back(Element<T>{"Zirconium", "Zr", 40, 91.224, 2.16, 1.45, 4, 4, 5});
        elements.push_back(Element<T>{"Niobium", "Nb", 41, 92.906, 2.15, 1.34, 5, 5, 5});
        elements.push_back(Element<T>{"Molybdenum", "Mo", 42, 95.95, 2.16, 1.30, 6, 6, 5});
        elements.push_back(Element<T>{"Technetium", "Tc", 43, 98.0, 2.17, 1.27, 7, 7, 5});
        elements.push_back(Element<T>{"Ruthenium", "Ru", 44, 101.07, 2.20, 1.26, 8, 8, 5});
        elements.push_back(Element<T>{"Rhodium", "Rh", 45, 102.905, 2.24, 1.25, 9, 9, 5});
        elements.push_back(Element<T>{"Palladium", "Pd", 46, 106.42, 2.28, 1.27, 10, 10, 5});
        elements.push_back(Element<T>{"Silver", "Ag", 47, 107.868, 2.44, 1.45, 11, 11, 5});
        elements.push_back(Element<T>{"Cadmium", "Cd", 48, 112.414, 2.69, 1.69, 12, 12, 5});
        elements.push_back(Element<T>{"Indium", "In", 49, 114.818, 2.93, 1.66, 3, 13, 5});
        elements.push_back(Element<T>{"Tin", "Sn", 50, 118.710, 3.00, 1.40, 4, 14, 5});
        elements.push_back(Element<T>{"Antimony", "Sb", 51, 121.760, 3.05, 1.39, 5, 15, 5});
        elements.push_back(Element<T>{"Tellurium", "Te", 52, 127.60, 3.05, 1.38, 6, 16, 5});
        elements.push_back(Element<T>{"Iodine", "I", 53, 126.904, 3.04, 1.37, 7, 17, 5});
        elements.push_back(Element<T>{"Xenon", "Xe", 54, 131.293, 3.00, 1.36, 8, 18, 5});
        elements.push_back(Element<T>{"Cesium", "Cs", 55, 132.905, 3.43, 2.32, 1, 1, 6});
        elements.push_back(Element<T>{"Barium", "Ba", 56, 137.327, 2.68, 1.96, 2, 2, 6});
        elements.push_back(Element<T>{"Lanthanum", "La", 57, 138.905, 2.40, 1.80, 3, 3, 6});
        elements.push_back(Element<T>{"Cerium", "Ce", 58, 140.116, 2.35, 1.63, 4, 3, 6});
        elements.push_back(Element<T>{"Praseodymium", "Pr", 59, 140.908, 2.32, 1.76, 5, 3, 6});
        elements.push_back(Element<T>{"Neodymium", "Nd", 60, 144.242, 2.31, 1.74, 6, 3, 6});
        elements.push_back(Element<T>{"Promethium", "Pm", 61, 145.0, 2.29, 1.73, 7, 3, 6});
        elements.push_back(Element<T>{"Samarium", "Sm", 62, 150.36, 2.27, 1.72, 8, 3, 6});
        elements.push_back(Element<T>{"Europium", "Eu", 63, 151.964, 2.25, 1.68, 9, 3, 6});
        elements.push_back(Element<T>{"Gadolinium", "Gd", 64, 157.25, 2.24, 1.69, 10, 3, 6});
        elements.push_back(Element<T>{"Terbium", "Tb", 65, 158.925, 2.23, 1.68, 11, 3, 6});
        elements.push_back(Element<T>{"Dysprosium", "Dy", 66, 162.500, 2.21, 1.67, 12, 3, 6});
        elements.push_back(Element<T>{"Holmium", "Ho", 67, 164.930, 2.20, 1.66, 13, 3, 6});
        elements.push_back(Element<T>{"Erbium", "Er", 68, 167.259, 2.19, 1.65, 14, 3, 6});
        elements.push_back(Element<T>{"Thulium", "Tm", 69, 168.934, 2.18, 1.64, 15, 3, 6});
        elements.push_back(Element<T>{"Ytterbium", "Yb", 70, 173.045, 2.22, 1.70, 16, 3, 6});
        elements.push_back(Element<T>{"Lutetium", "Lu", 71, 174.967, 2.17, 1.62, 17, 3, 6});
        elements.push_back(Element<T>{"Hafnium", "Hf", 72, 178.49, 2.08, 1.52, 4, 4, 6});
        elements.push_back(Element<T>{"Tantalum", "Ta", 73, 180.948, 2.00, 1.46, 5, 5, 6});
        elements.push_back(Element<T>{"Tungsten", "W", 74, 183.84, 1.93, 1.37, 6, 6, 6});
        elements.push_back(Element<T>{"Rhenium", "Re", 75, 186.207, 1.88, 1.31, 7, 7, 6});
        elements.push_back(Element<T>{"Osmium", "Os", 76, 190.23, 1.85, 1.29, 8, 8, 6});
        elements.push_back(Element<T>{"Iridium", "Ir", 77, 192.217, 1.80, 1.27, 9, 9, 6});
        elements.push_back(Element<T>{"Platinum", "Pt", 78, 195.084, 1.77, 1.30, 10, 10, 6});
        elements.push_back(Element<T>{"Gold", "Au", 79, 196.967, 1.74, 1.34, 11, 11, 6});
        elements.push_back(Element<T>{"Mercury", "Hg", 80, 200.592, 1.71, 1.49, 12, 12, 6});
        elements.push_back(Element<T>{"Thallium", "Tl", 81, 204.38, 1.56, 1.48, 3, 13, 6});
        elements.push_back(Element<T>{"Lead", "Pb", 82, 207.2, 1.54, 1.47, 4, 14, 6});
        elements.push_back(Element<T>{"Bismuth", "Bi", 83, 208.980, 1.52, 1.46, 5, 15, 6});
        elements.push_back(Element<T>{"Polonium", "Po", 84, 209.0, 1.53, 1.46, 6, 16, 6});
        elements.push_back(Element<T>{"Astatine", "At", 85, 210.0, 1.50, 1.45, 7, 17, 6});
        elements.push_back(Element<T>{"Radon", "Rn", 86, 222.0, 1.50, 1.45, 8, 18, 6});
        elements.push_back(Element<T>{"Francium", "Fr", 87, 223.0, 3.48, 2.60, 1, 1, 7});
        elements.push_back(Element<T>{"Radium", "Ra", 88, 226.0, 2.83, 2.21, 2, 2, 7});
        elements.push_back(Element<T>{"Actinium", "Ac", 89, 227.0, 2.60, 2.15, 3, 3, 7});
        elements.push_back(Element<T>{"Thorium", "Th", 90, 232.038, 2.52, 2.06, 4, 3, 7});
        elements.push_back(Element<T>{"Protactinium", "Pa", 91, 231.036, 2.43, 2.00, 5, 3, 7});
        elements.push_back(Element<T>{"Uranium", "U", 92, 238.029, 2.41, 1.96, 6, 3, 7});
        elements.push_back(Element<T>{"Neptunium", "Np", 93, 237.0, 2.39, 1.90, 7, 3, 7});
        elements.push_back(Element<T>{"Plutonium", "Pu", 94, 244.0, 2.43, 1.87, 8, 3, 7});
        elements.push_back(Element<T>{"Americium", "Am", 95, 243.0, 2.44, 1.80, 9, 3, 7});
        elements.push_back(Element<T>{"Curium", "Cm", 96, 247.0, 2.45, 1.69, 10, 3, 7});
        elements.push_back(Element<T>{"Berkelium", "Bk", 97, 247.0, 2.44, 1.68, 11, 3, 7});
        elements.push_back(Element<T>{"Californium", "Cf", 98, 251.0, 2.45, 1.68, 12, 3, 7});
        elements.push_back(Element<T>{"Einsteinium", "Es", 99, 252.0, 2.45, 1.65, 13, 3, 7});
        elements.push_back(Element<T>{"Fermium", "Fm", 100, 257.0, 2.45, 1.67, 14, 3, 7});
        elements.push_back(Element<T>{"Mendelevium", "Md", 101, 258.0, 2.46, 1.73, 15, 3, 7});
        elements.push_back(Element<T>{"Nobelium", "No", 102, 259.0, 2.46, 1.76, 16, 3, 7});
        elements.push_back(Element<T>{"Lawrencium", "Lr", 103, 266.0, 2.46, 1.61, 17, 3, 7});
        elements.push_back(Element<T>{"Rutherfordium", "Rf", 104, 267.0, 2.40, 1.57, 4, 4, 7});
        elements.push_back(Element<T>{"Dubnium", "Db", 105, 268.0, 2.20, 1.49, 5, 5, 7});
        elements.push_back(Element<T>{"Seaborgium", "Sg", 106, 269.0, 2.15, 1.43, 6, 6, 7});
        elements.push_back(Element<T>{"Bohrium", "Bh", 107, 270.0, 2.10, 1.41, 7, 7, 7});
        elements.push_back(Element<T>{"Hassium", "Hs", 108, 277.0, 2.05, 1.34, 8, 8, 7});
        elements.push_back(Element<T>{"Meitnerium", "Mt", 109, 278.0, 2.00, 1.29, 9, 9, 7});
        elements.push_back(Element<T>{"Darmstadtium", "Ds", 110, 281.0, 1.95, 1.28, 10, 10, 7});
        elements.push_back(Element<T>{"Roentgenium", "Rg", 111, 282.0, 1.90, 1.21, 11, 11, 7});
        elements.push_back(Element<T>{"Copernicium", "Cn", 112, 285.0, 1.85, 1.22, 12, 12, 7});
        elements.push_back(Element<T>{"Nihonium", "Nh", 113, 286.0, 1.80, 1.36, 3, 13, 7});
        elements.push_back(Element<T>{"Flerovium", "Fl", 114, 289.0, 1.75, 1.43, 4, 14, 7});
        elements.push_back(Element<T>{"Moscovium", "Mc", 115, 289.0, 1.70, 1.62, 5, 15, 7});
        elements.push_back(Element<T>{"Livermorium", "Lv", 116, 293.0, 1.65, 1.75, 6, 16, 7});
        elements.push_back(Element<T>{"Tennessine", "Ts", 117, 294.0, 1.60, 1.65, 7, 17, 7});
        elements.push_back(Element<T>{"Oganesson", "Og", 118, 294.0, 1.55, 1.57, 8, 18, 7});
    }
    ~PeriodicTable() = default;  // Default destructor

    /**
     * @brief Adds a new element to the periodic table.
     */
    void addElement(const Element<T>& element) {
        elements.push_back(element);  // Add element to the vector
    }

    /**
     * @brief Retrieves an element by its atomic number.
     * @param atomic_number The atomic number of the element to retrieve.
     * @return The element with the specified atomic number.
     * @throws std::runtime_error if the element is not found.
     */
    Element<T> getElementByAtomicNumber(uint16_t atomic_number) const {
        for (const auto& element : elements) {
            if (element.atomic_number == atomic_number) {
                return element;  // Return the element with the specified atomic number
            }
        }
        throw std::runtime_error("Element not found");  // Throw an error if not found
    }

    /**
     * @brief Retrieves an element by its symbol.
     * @param symbol The symbol of the element to retrieve.
     * @return The element with the specified symbol.
     * @throws std::runtime_error if the element is not found.
     */
    Element<T> getElementBySymbol(const std::string& symbol) const {
        for (const auto& element : elements) {
            if (element.symbol == symbol) {
                return element;  // Return the element with the specified symbol
            }
        }
        throw std::runtime_error("Element not found");  // Throw an error if not found
    }

   private:
    std::vector<Element<T>> elements;  // Vector to store elements
};
}  // namespace periodic_table

namespace structures {
template <typename T>
struct Coordinate {
    T x;
    T y;
    T z;

    Coordinate(T x, T y, T z) : x(x), y(y), z(z) {}
};

/**
 * @brief Represents an atom in the periodic table.
 */
template <typename T>
class Atom {
   public:
    periodic_table::Element<T> element;  // Element of the atom
    Coordinate<T> position;

    /**
     * @brief Constructs an Atom with the specified element and position.
     */
    Atom(const periodic_table::Element<T>& element, const Coordinate<T>& position)
        : element(element), position(position) {}

    /**
     * @brief Sets the position of the atom.
     * @param newPosition The new position to set.
     */
    void setPosition(const Coordinate<T>& newPosition) {
        position = newPosition;
    }
};

/**
 * @brief Represents a cell in the periodic table.
 */
template <typename T>
class Cell {
   public:
    T x;
    T y;
    T z;
    T alpha;
    T beta;
    T gamma;
    T cell[3][3];  // 3x3 matrix to store vectors of the cell

    /**
     * @brief Constructs a Cell with the specified dimensions and angles.
     * @param x The x dimension of the cell.
     * @param y The y dimension of the cell.
     * @param z The z dimension of the cell.
     * @param alpha The alpha angle of the cell.
     * @param beta The beta angle of the cell.
     * @param gamma The gamma angle of the cell.
     */
    Cell(T x, T y, T z, T alpha, T beta, T gamma)
        : x(x), y(y), z(z), alpha(alpha), beta(beta), gamma(gamma) {
        // Initialize the cell matrix
        // Convert angles from degrees to radians
        const T pi = std::acos(-1.0);
        const T alpha_rad = alpha * pi / 180.0;
        const T beta_rad = beta * pi / 180.0;
        const T gamma_rad = gamma * pi / 180.0;

        // First vector
        cell[0][0] = x;
        cell[0][1] = 0.0;
        cell[0][2] = 0.0;

        // Second vector
        cell[1][0] = y * std::cos(gamma_rad);
        cell[1][1] = y * std::sin(gamma_rad);
        cell[1][2] = 0.0;

        // Third vector
        cell[2][0] = z * std::cos(beta_rad);
        cell[2][1] =
            z *
            (std::cos(alpha_rad) - std::cos(beta_rad) * std::cos(gamma_rad)) /
            std::sin(gamma_rad);
        T tmp =
            1.0 - std::pow(std::cos(alpha_rad), 2) -
            std::pow(std::cos(beta_rad), 2) - std::pow(std::cos(gamma_rad), 2) +
            2 * std::cos(alpha_rad) * std::cos(beta_rad) * std::cos(gamma_rad);
        cell[2][2] = z * std::sqrt(std::max(tmp, T(0.0))) / std::sin(gamma_rad);
    }
};

/**
 * @brief Represents a structure block in the periodic table.
 */
template <typename T>
class StructureBlock {
   public:
    T energy;
    std::string symmetry;
    Cell<T> cell;
    std::vector<Atom<T>> atoms;  // Vector to store atoms

    /**
     * @brief Constructs a StructureBlock with the specified energy and symmetry.
     * @param energy The energy of the structure block.
     * @param symmetry The symmetry of the structure block.
     */
    StructureBlock(T energy, std::string symmetry)
        : energy(energy), symmetry(symmetry), cell(0, 0, 0, 90, 90, 90) {}

    /**
     * @brief Adds an atom to the structure block.
     * @param atom The atom to add.
     */
    void addatom(const Atom<T>& atom) {
        atoms.push_back(atom);  // Add atom to the vector
    }

    /**
     * @brief Sets the cell of the structure block.
     * @param newCell The new cell to set.
     */
    void set_cell(const Cell<T>& newCell) {
        cell = newCell;  // Set the cell
    }
};
}  // namespace structures