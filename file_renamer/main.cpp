#include <iostream>
#include <filesystem>
#include <string>

namespace fs = std::filesystem;

int main() {
    fs::path folder = R"(C:\Users\culya\Desktop\binary_masks)";

    for (const auto& entry : fs::directory_iterator(folder)) {

        if (!entry.is_regular_file())
            continue;

        std::string filename = entry.path().filename().string();

        // Check if filename ends with "_mask.png"
        std::string suffix = "_mask.png";

        if (filename.size() >= suffix.size() &&
            filename.compare(filename.size() - suffix.size(),
                             suffix.size(), suffix) == 0)
        {
            // remove "_mask"
            std::string new_name =
                filename.substr(0, filename.size() - suffix.size())
                + ".png";

            fs::path new_path = folder / new_name;

            std::cout << "Renaming: "
                      << filename << " -> "
                      << new_name << std::endl;

            fs::rename(entry.path(), new_path);
        }
    }

    return 0;
}