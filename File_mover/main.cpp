#include <iostream>
#include <filesystem>
#include <string>
#include <fstream>
#include <vector>
#include <unordered_map>
#include <sys/stat.h>

using namespace std;

#define SOURCE_FOLDER R"(C:\Users\culya\Desktop\data_bakalarka\data\patches\crops_cancer)"
#define TARGET_FOLDER R"(C:\Users\culya\Desktop\gec)"
#define CSV_PATH R"(C:\Users\culya\Desktop\data_bakalarka\data_info\patient_stats.csv)"

typedef struct {
    int id;
    int no_files;
    int cancerous;
}PatientAttr;

unordered_map<int, int> patientNoFiles;
unordered_map<int, int> patientCancerous;


void crtFolder(filesystem::path folder_path) {
    if (!mkdir(folder_path.string().c_str())) cout << "Folder created: " << folder_path.string() << endl;
    else cout << "Folder exists: " << folder_path.string() << endl;
}

string extractView(const string& filename)
{
    if (filename.find("L_CC") != string::npos) return "L_CC";
    if (filename.find("R_CC") != string::npos) return "R_CC";
    if (filename.find("L_MLO") != string::npos) return "L_MLO";
    if (filename.find("R_MLO") != string::npos) return "R_MLO";

    return "UNKNOWN";
}


int main() {
    filesystem::path source_folder = SOURCE_FOLDER;
    filesystem::path target_folder = TARGET_FOLDER;
    ifstream file(CSV_PATH);

    string line;
    getline(file, line);

    while (getline(file, line)) {
        stringstream ss(line);
        string csvCell;
        vector<string> csvRow;
        while (getline(ss, csvCell, ',')) csvRow.push_back(csvCell);

        PatientAttr entry;
        entry.id = stoi(csvRow[0]);
        entry.no_files = stoi(csvRow[1]);
        entry.cancerous = stoi(csvRow[2]);
        patientNoFiles[entry.id] = entry.no_files;
        patientCancerous[entry.id] = entry.cancerous;
    }

    for (const auto& entry : filesystem::recursive_directory_iterator(source_folder)) {
        if (!entry.is_regular_file())continue;

        string filename = entry.path().filename().string();

        string patientID = entry.path().filename().string().substr(0,5);
        int patientIDnum = stoi(patientID);

        /*filesystem::path cancer_folder = target_folder;
        cancer_folder /= to_string(patientCancerous[patientIDnum]);
        crtFolder(cancer_folder);*/

        /*filesystem::path no_exams_folder = target_folder;
        no_exams_folder /= to_string(patientNoFiles[patientIDnum]);
        crtFolder(no_exams_folder);*/

        /*filesystem::path id_sub_folder = no_exams_folder;
        id_sub_folder /= patientID.substr(0,2);
        crtFolder(id_sub_folder);*/

        filesystem::path patient_folder = target_folder;
        patient_folder /= patientID;
        crtFolder(patient_folder);

        filesystem::path images_folder = patient_folder / "images";
        filesystem::path masks_folder = patient_folder / "masks";
        crtFolder(images_folder);
        crtFolder(masks_folder);

        string view = extractView(filename);
        filesystem::path view_folder = images_folder / view;
        crtFolder(view_folder);

        filesystem::path destination = view_folder / filename;

        try {
            filesystem::rename(entry.path(), destination);
            cout << "File " << entry.path().string() << " moved to: " << destination << endl;
        }
        catch (const filesystem::filesystem_error& e) { cout << "Error: " << e.what() << endl; }
    }
    return 0;
}