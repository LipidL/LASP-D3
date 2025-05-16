#include <string>
#include <optional>
#include "modules.h"
#include <regex>
#include <fstream>
#include <sstream>

namespace parser
{
    template <typename T>
    requires std::floating_point<T> // Ensure T is a real number type
    class structureIO
    {
    public:
        virtual std::optional<structures::Atom<T>> parse_atom(std::string input){
            return std::nullopt; // Placeholder for actual parsing logic
        }
        virtual std::optional<structures::Cell<T>> parse_cell(std::string input){
            return std::nullopt; // Placeholder for actual parsing logic
        }
        virtual std::optional<structures::StructureBlock<T>> parse_header(std::string input){
            return std::nullopt; // Placeholder for actual parsing logic
        }
        virtual std::optional<std::vector<structures::StructureBlock<T>>> parse_file(std::string filename){
            // open the file
            std::ifstream file(filename);
            if (!file.is_open()) {
                return std::nullopt; // File could not be opened
            }
            // read the file line by line
            std::vector<structures::StructureBlock<T>> structures;
            std::optional<structures::StructureBlock<T>> current_structure = std::nullopt;
            while (true) {
                std::string line;
                if (!std::getline(file, line)) {
                    break; // End of file reached
                }
                // parse the line
                auto atom = parse_atom(line);
                if (atom) {
                    if (current_structure) {
                        current_structure->addatom(*atom); // Add atom to the current structure
                    }
                } else {
                    auto cell = parse_cell(line);
                    if (cell) {
                        if (current_structure) {
                            current_structure->cell = *cell; // Set the cell for the current structure
                        }
                    } else {
                        auto structure = parse_header(line);
                        if (structure) {
                            if (current_structure) {
                                structures.push_back(*current_structure); // Add the structure to the vector
                            }
                            current_structure = *structure; // Start a new structure
                        } else {
                            // Handle invalid line
                            std::cerr << "Invalid line: " << line << std::endl;
                        }
                    }
                }
            }
            // file will be closed automatically when ifstream goes out of scope
            if (current_structure) {
                structures.push_back(*current_structure); // Add the last structure to the vector
            }
            if (structures.empty()) {
                return std::nullopt; // No structures found
            }
            return structures; // Return the vector of structures
        }
    };

    template <typename T>
    requires std::floating_point<T> // Ensure T is a real number type
    class ArcParser : public structureIO<T>
    {
    public:
        ArcParser() {
            this->periodic_table = periodic_table::PeriodicTable<T>(); // Initialize the periodic table
        }
        std::optional<structures::Atom<T>> parse_atom(std::string input) override {
            /* split the string by space */
            std::istringstream iss(input);
            std::vector<std::string> tokens;
            std::string token;
            while (iss >> token) {
                tokens.push_back(token);
            }
            // Check the size of tokens
            if (tokens.size() != 10) {  // atom line should have 10 entries
                return std::nullopt;
            }

            try {
                // Format: element_symbol x y z [other_data...]
                std::string element = tokens[0];
                T x = static_cast<T>(std::stod(tokens[1]));
                T y = static_cast<T>(std::stod(tokens[2]));
                T z = static_cast<T>(std::stod(tokens[3]));
                periodic_table::Element<T> element_data = periodic_table.getElementBySymbol(element);
                // Create and return an Atom object
                return structures::Atom<T>(periodic_table.getElementBySymbol(element), structures::Coordinate<T>(x, y, z));
            } catch (const std::exception& e) {
                // Handle conversion errors
                if (std::string(e.what()) == "Element not found") {
                    std::cerr << "Element not found: " << tokens[0] << std::endl;
                }
                return std::nullopt;
            }
        }

        std::optional<structures::Cell<T>> parse_cell(std::string input) override {
            /* split the string by space */
            std::istringstream iss(input);
            std::vector<std::string> tokens;
            std::string token;
            while (iss >> token) {
                tokens.push_back(token);
            }
            /* check the size of tokens */
            if (tokens.size() != 7) {  // cell line should have 7 entries
                return std::nullopt;
            }
            try{
                /* Format: PBC a b c alpha beta gamma*/
                if (tokens[0] != "PBC") {
                    return std::nullopt; // Not a valid cell line
                }
                T a = static_cast<T>(std::stod(tokens[1]));
                T b = static_cast<T>(std::stod(tokens[2]));
                T c = static_cast<T>(std::stod(tokens[3]));
                T alpha = static_cast<T>(std::stod(tokens[4]));
                T beta = static_cast<T>(std::stod(tokens[5]));
                T gamma = static_cast<T>(std::stod(tokens[6]));
                /* Create and return a cell object */
                return structures::Cell<T>(a, b, c, alpha, beta, gamma);
            } catch(const std::exception& e) {
                std::cerr << e.what() << '\n';
                return std::nullopt; // Handle conversion errors
            }
            
        }

        std::optional<structures::StructureBlock<T>> parse_header(std::string input) override {
            /* split the string by space */
            std::istringstream iss(input);
            std::vector<std::string> tokens;
            std::string token;
            while (iss >> token) {
                tokens.push_back(token);
            }
            /* check the size of tokens */
            if (tokens.size() != 4 && tokens.size() != 5) {  // header line should have 4 or 5 entries
                return std::nullopt;
            }
            try {
                /* Format: Energy number force energy symmetry */
                if (tokens[0] != "Energy") {
                    return std::nullopt; // Not a valid header line
                }
                T energy = static_cast<T>(std::stod(tokens[3]));
                std::string symmetry;
                if (tokens.size() == 5) {
                    symmetry = tokens[4];
                } else {
                    symmetry = "C1"; // Default symmetry
                }
                /* Create and return a structure block object */
                return structures::StructureBlock<T>(energy, symmetry);
            } catch (const std::exception& e) {
                // Handle conversion errors
                std::cerr << e.what() << '\n';
                return std::nullopt;
            }
        }
    private:
        periodic_table::PeriodicTable<T> periodic_table; // Periodic table instance
    }; // ArcParser
} // namespace parser