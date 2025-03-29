#include <concepts.h>
#include <string>
#include <vector>

namespace periodic_table
{
    template <typename T>
    requires Real<T> // Ensure T is a real number type
    class Element
    {
        public:
        std::string name; // Name of the element
        std::string symbol; // Symbol of the element
        uint16_t atomic_number; // Atomic number of the element
        T mass; // Atomic weight of the element
        T valence_radius; // Valence radius of the element
        T covalent_radius; // Covalent radius of the element
        uint16_t valence_electrons; // Number of valence electrons
        uint16_t group; // Group number of the element in the periodic table
        uint16_t period; // Period number of the element in the periodic table
    };

    template <typename T>
    requires Real<T> // Ensure T is a real number type
    class PeriodicTable
    {
        public:
        std::vector<Element<T>> elements; // Vector to store elements

        PeriodicTable() {
        }

        void addElement(const Element<T>& element) {
            elements.push_back(element); // Add element to the vector
        }
        Element<T> getElementByAtomicNumber(uint16_t atomic_number) const {
            for (const auto& element : elements) {
                if (element.atomic_number == atomic_number) {
                    return element; // Return the element with the specified atomic number
                }
            }
            throw std::runtime_error("Element not found"); // Throw an error if not found
        }
        Element<T> getElementBySymbol(const std::string& symbol) const {
            for (const auto& element : elements) {
                if (element.symbol == symbol) {
                    return element; // Return the element with the specified symbol
                }
            }
            throw std::runtime_error("Element not found"); // Throw an error if not found
        }
    };
} // namespace periodic_table

namespace structures
{
    template <typename T>
    requires Real<T> // Ensure T is a real number type
    struct Coordinate
    {
        T x;
        T y;
        T z;

        Coordinate(T x, T y, T z) : x(x), y(y), z(z) {}
    };

    template <typename T>
    requires Real<T> // Ensure T is a real number type
    class Atom {
        public:
        periodic_table::Element<T> element; // Element of the atom
        Coordinate<T> position;
        
        Atom(const periodic_table::Element<T>& element, const Coordinate<T>& position)
            : element(element), position(position) {}

        void setPosition(const Coordinate<T>& newPosition) {
            position = newPosition;
        }

        Coordinate<T> getPosition() const {
            return position;
        }
    };

    template <typename T>
    requires Real<T> // Ensure T is a real number type
    class Cell {
        public:
        T x;
        T y;
        T z;
        T alpha;
        T beta;
        T gamma;
    };

    template <typename T>
    requires Real<T> // Ensure T is a real number type
    class StructureBlock {
        public:
        T energy;
        std::string symmetry;
        Cell<T> cell;
        std::vector<Atom<T>> atoms; // Vector to store atoms

        void addatom(const Atom<T>& atom) {
            atoms.push_back(atom); // Add atom to the vector
        }

        void set_cell(const Cell<T>& newCell) {
            cell = newCell; // Set the cell
        }
    };
} // namespace structure