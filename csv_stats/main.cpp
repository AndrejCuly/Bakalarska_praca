#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

#define CSV_PATH R"(C:\Users\culya\Desktop\data_bakalarka\data_info\CSAW_CaseControl_WITHOUTHIDDEN_for_anon_dataset_210902.csv)"

typedef struct {
    int patient_id;
    int patient_no_files;
    bool cancerous;
}PatientHelpStruct;

std::vector<PatientHelpStruct> patientStats;

int main() {
    std::ifstream file(CSV_PATH);
    if (!file.is_open()) {
        std::cerr << "Error opening CSV file\n";
        return 1;
    }

    std::string line;

    // skip header
    std::getline(file, line);

    int last_patient_id = -1;

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string cell;
        std::vector<std::string> tokens;

        while (std::getline(ss, cell, ',')) {
            tokens.push_back(cell);
        }

        if (tokens.size() < 4)
            continue;

        int patient_id = std::stoi(tokens[0]);
        bool cancerous = (tokens[3] != "NA");

        if (patient_id != last_patient_id) {
            patientStats.push_back({patient_id, 1, cancerous});
            last_patient_id = patient_id;
        } else {
            patientStats.back().patient_no_files++;
        }
    }

    std::ofstream output_file(R"(C:\Users\culya\Desktop\patient_stats.csv)");
    if (!output_file.is_open()) {
        std::cerr << "Error opening CSV file\n";
        return 1;
    }
    output_file << "patient_id, patient_no_files, cancerous\n";

    for (const PatientHelpStruct& p : patientStats) {
        output_file << p.patient_id << ", " << p.patient_no_files/4 << ", " << p.cancerous << "\n";
    }
    return 0;
}