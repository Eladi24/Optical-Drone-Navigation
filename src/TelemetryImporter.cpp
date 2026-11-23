#include <cctype>
#include "TelemetryImporter.hpp"

#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <sys/stat.h>
#include <sys/types.h>
#include <vector>

namespace {

struct Row {
    int    num;
    int    video_idx;  // 0-based index in the assembled video = filename number - 1
    std::string filename;
    double time_sec;   // relative to the first row, filled in after parsing
    double lat, lon, height, phi1;
};

// Parses "YYYY-MM-DDTHH:MM:SS" into seconds since epoch (UTC). Only relative
// deltas between rows are ever used, so timezone handling doesn't matter as
// long as it's consistent -- timegm (UTC, no DST) is the simplest correct choice.
double parseIsoTimestamp(const std::string& iso) {
    std::tm tm{};
    std::istringstream ss(iso);
    ss >> std::get_time(&tm, "%Y-%m-%dT%H:%M:%S");
    if (ss.fail()) return -1.0;
    // Optional fractional seconds (".084"): AerialVL frames are 0.5 s apart, and dropping the
    // fraction would quantise Time_Sec / the Kalman dt to 0 or 1 s. UAV-VisLoc's whole-second
    // timestamps parse exactly as before (2026-11-23, tests/test_telemetry_importer.cpp).
    double frac = 0.0;
    if (ss.peek() == '.') {
        ss.get();
        std::string digits;
        while (std::isdigit(ss.peek())) digits += static_cast<char>(ss.get());
        if (!digits.empty()) frac = std::stod("0." + digits);
    }
    return static_cast<double>(timegm(&tm)) + frac;
}

// "08_0221.JPG" -> 220. DatasetVideoAssembler stitches <prefix>%04d.JPG for i = 1..frame_count,
// so the FILENAME number, not the CSV's `num`, is what fixes a still's video index. They differ
// on flight 08: `num` jumps 220 -> 261 while filenames and positions run on unbroken.
// Returns -1 if the filename has no trailing number.
int videoIndexFromFilename(const std::string& filename) {
    size_t dot = filename.find_last_of('.');
    size_t end = (dot == std::string::npos) ? filename.size() : dot;
    size_t start = end;
    while (start > 0 && std::isdigit(static_cast<unsigned char>(filename[start - 1]))) --start;
    if (start == end) return -1;
    return std::stoi(filename.substr(start, end - start)) - 1;
}

std::vector<std::string> splitCsvLine(const std::string& line) {
    std::vector<std::string> fields;
    std::istringstream ss(line);
    std::string field;
    while (std::getline(ss, field, ',')) fields.push_back(field);
    return fields;
}

} // namespace

TelemetryImportResult importUavVisLocTelemetry(const std::string& csv_path,
                                                const std::string& sample_name) {
    std::ifstream in(csv_path);
    if (!in.is_open()) {
        std::cerr << "importUavVisLocTelemetry: could not open " << csv_path << std::endl;
        return {false, 0, -1.0};
    }

    std::string header;
    std::getline(in, header);   // num,filename,date,lat,lon,height,Omega,Kappa,Phi1,Phi2

    std::vector<Row> rows;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        std::vector<std::string> f = splitCsvLine(line);
        if (f.size() < 9) continue;   // malformed row -- skip rather than crash

        Row row;
        row.num      = std::stoi(f[0]);
        row.filename = f[1];
        row.video_idx = videoIndexFromFilename(row.filename);
        if (row.video_idx < 0) row.video_idx = row.num - 1;
        double abs_time = parseIsoTimestamp(f[2]);
        row.lat    = std::stod(f[3]);
        row.lon    = std::stod(f[4]);
        row.height = std::stod(f[5]);
        row.phi1   = std::stod(f[8]);   // Phi1: higher-confidence yaw estimate
        row.time_sec = abs_time;        // absolute for now; rebased below
        rows.push_back(row);
    }

    if (rows.empty()) {
        std::cerr << "importUavVisLocTelemetry: no rows parsed from " << csv_path << std::endl;
        return {false, 0, -1.0};
    }

    double t0 = rows.front().time_sec;
    double altitude_sum = 0.0;
    for (auto& r : rows) {
        r.time_sec = (r.time_sec >= 0 && t0 >= 0) ? (r.time_sec - t0) : 0.0;
        altitude_sum += r.height;
    }
    double mean_altitude_m = altitude_sum / static_cast<double>(rows.size());

    mkdir("CSV Files", 0755);

    std::string gt_path = "CSV Files/ground_truth_" + sample_name + ".csv";
    std::ofstream gt(gt_path);
    // Default stream precision is 6 SIGNIFICANT digits -- that truncated lng 115.9747973 to
    // 115.975 (~48 m at 30N) and cost 15-26 m mean / 50 m max of ground-truth error.
    gt << std::setprecision(12);
    // Frame = 0-based video index (filename number - 1), the same index VideoProcessing's frame_idx
    // and the video_telemetry CSV use. Until 2026-11-17 it was the 1-based `num`: every Python-side
    // comparison was one still (~80 m) off, and on flight 08 (num gap 220->261) frames >= 220 were
    // compared against a position 40 stills (~4 km) earlier, in C++ and Python alike.
    gt << "Frame,Time_Sec,Lat,Lng,Label\n";
    for (auto& r : rows)
        gt << r.video_idx << "," << r.time_sec << "," << r.lat << "," << r.lon << "," << r.filename << "\n";

    std::string telem_path = "CSV Files/telemetry_" + sample_name + ".csv";
    std::ofstream telem(telem_path);
    telem << std::setprecision(12);
    telem << "Frame,Time_Sec,Lat,Lng,Alt,Heading\n";
    for (auto& r : rows)
        telem << r.video_idx << "," << r.time_sec << "," << r.lat << "," << r.lon << "," << r.height << "," << r.phi1 << "\n";

    std::cout << "importUavVisLocTelemetry: wrote " << rows.size() << " rows -> "
              << gt_path << " and " << telem_path
              << " (mean altitude " << mean_altitude_m << " m)" << std::endl;

    return {true, static_cast<int>(rows.size()), mean_altitude_m};
}

std::vector<double> loadTelemetryFrameTimes(const std::string& sample_name) {
    std::vector<double> times;
    std::ifstream in("CSV Files/telemetry_" + sample_name + ".csv");
    if (!in.is_open()) return times;

    std::string header;
    std::getline(in, header);   // Frame,Time_Sec,Lat,Lng,Alt,Heading

    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        std::vector<std::string> f = splitCsvLine(line);
        if (f.size() < 2) continue;
        int frame = std::stoi(f[0]);
        double time_sec = std::stod(f[1]);
        int idx = frame;       // Frame is the 0-based video index
        if (idx < 0) continue;
        if (static_cast<int>(times.size()) <= idx) times.resize(idx + 1, 0.0);
        times[idx] = time_sec;
    }
    return times;
}

std::vector<double> loadTelemetryHeadings(const std::string& sample_name) {
    std::vector<double> headings;
    std::ifstream in("CSV Files/telemetry_" + sample_name + ".csv");
    if (!in.is_open()) return headings;

    std::string header;
    std::getline(in, header);   // Frame,Time_Sec,Lat,Lng,Alt,Heading

    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        std::vector<std::string> f = splitCsvLine(line);
        if (f.size() < 6) continue;
        int frame = std::stoi(f[0]);
        double heading_deg = std::stod(f[5]);
        int idx = frame;       // Frame is the 0-based video index
        if (idx < 0) continue;
        if (static_cast<int>(headings.size()) <= idx)
            headings.resize(idx + 1, std::numeric_limits<double>::quiet_NaN());
        headings[idx] = heading_deg;
    }
    return headings;
}

SatelliteBounds readUavVisLocSatelliteBounds(const std::string& coords_csv_path,
                                             const std::string& map_name) {
    std::ifstream in(coords_csv_path);
    if (!in.is_open()) return {false, 0, 0, 0, 0};

    std::string header;
    std::getline(in, header);   // mapname,LT_lat_map,LT_lon_map,RB_lat_map,RB_lon_map,region

    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        std::vector<std::string> f = splitCsvLine(line);
        if (f.size() < 5) continue;
        if (f[0] != map_name) continue;
        return {true, std::stod(f[1]), std::stod(f[2]), std::stod(f[3]), std::stod(f[4])};
    }
    return {false, 0, 0, 0, 0};
}
