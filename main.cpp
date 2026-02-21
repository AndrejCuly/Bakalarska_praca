#include <iostream>
#include <filesystem>
#include <string>
#include <sys/stat.h>

#define SOURCE_FOLDER R"(Insert\absolute\path)"
#define TARGET_FOLDER R"(Insert\absolute\path)"

int main() {
    std::filesystem::path source_folder = SOURCE_FOLDER;
    std::filesystem::path target_folder = TARGET_FOLDER;

    for (const auto& entry : std::filesystem::recursive_directory_iterator(source_folder)) {
        std::filesystem::path new_folder = target_folder;
        new_folder /= entry.path().filename().string().substr(0,2);
        std::cout << new_folder << std::endl;
        std::filesystem::path destination = new_folder / entry.path().filename();

        if (!mkdir(new_folder.string().c_str())) std::cout << "Folder created: " << target_folder.string() << std::endl;
        else std::cout << "Folder exists: " << target_folder.string() << std::endl;

        try {
            std::filesystem::rename(entry.path(), destination);
            std::cout << "File " << entry.path().string() << " moved to: " << destination << std::endl;
        }
        catch (const std::filesystem::filesystem_error& e) { std::cout << "Error: " << e.what() << std::endl; }
    }
    return 0;
}