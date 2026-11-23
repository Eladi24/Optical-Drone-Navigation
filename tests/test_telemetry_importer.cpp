// TelemetryImporter. Regressions for two silent data bugs:
//  - frame numbering (2026-11-17): Frame must be the image FILENAME number - 1, never the CSV's
//    `num`; on flight 08 `num` jumps 220 -> 261 while the files don't, which put 79% of the
//    flight against ground truth ~4 km away.
//  - GT precision (2026-11-16): lat/lng were written with 6 significant digits (15-26 m error).
#include "TelemetryImporter.hpp"
#include "test_helpers.hpp"
#include <gtest/gtest.h>
#include <cmath>
#include <fstream>
#include <sstream>

namespace {
void writeFile(const std::string& path, const std::string& text)
{
    std::ofstream f(path);
    f << text;
}

std::vector<std::vector<std::string>> readCsv(const std::string& path)
{
    std::vector<std::vector<std::string>> rows;
    std::ifstream in(path);
    std::string line;
    std::getline(in, line);  // header
    while (std::getline(in, line)) {
        std::vector<std::string> f;
        std::stringstream ss(line);
        std::string field;
        while (std::getline(ss, field, ',')) f.push_back(field);
        rows.push_back(f);
    }
    return rows;
}

// Mimics flight 08 around the `num` jump, including a 568 s capture pause.
const char* kFlightCsv =
    "num,filename,date,lat,lon,height,Omega,Kappa,Phi1,Phi2\n"
    "219,08_0219.JPG,2019-06-14T16:07:31,30.945347381234,120.144597212345,551.9,-5.3,1.1,98.336746,103.6\n"
    "220,08_0220.JPG,2019-06-14T16:07:33,30.945244251234,120.145101412345,552.1,-8.6,-0.2,97.259586,101.1\n"
    "261,08_0221.JPG,2019-06-14T16:17:01,30.946001234567,120.146002345678,553.0,-2.0,0.5,-81.5,-80.0\n";
}  // namespace

TEST(TelemetryImporter, FrameIsFilenameNumberMinusOneNotNum)
{
    testutil::ScopedTempCwd tmp;
    writeFile("flight.csv", kFlightCsv);
    TelemetryImportResult r = importUavVisLocTelemetry("flight.csv", "t08");
    ASSERT_TRUE(r.success);
    EXPECT_EQ(r.frame_count, 3);
    EXPECT_NEAR(r.mean_altitude_m, (551.9 + 552.1 + 553.0) / 3.0, 1e-9);

    auto gt = readCsv("CSV Files/ground_truth_t08.csv");
    ASSERT_EQ(gt.size(), 3u);
    EXPECT_EQ(gt[0][0], "218");
    EXPECT_EQ(gt[1][0], "219");
    EXPECT_EQ(gt[2][0], "220");  // num says 261 -> 260 would be the bug
    EXPECT_EQ(gt[2][4], "08_0221.JPG");
}

TEST(TelemetryImporter, TimeIsRealElapsedSeconds)
{
    testutil::ScopedTempCwd tmp;
    writeFile("flight.csv", kFlightCsv);
    ASSERT_TRUE(importUavVisLocTelemetry("flight.csv", "t08").success);
    std::vector<double> times = loadTelemetryFrameTimes("t08");
    ASSERT_EQ(times.size(), 221u);  // indexed by Frame
    EXPECT_DOUBLE_EQ(times[218], 0.0);
    EXPECT_DOUBLE_EQ(times[219], 2.0);
    EXPECT_DOUBLE_EQ(times[220], 570.0);  // the capture pause is preserved, not a nominal fps
}

// AerialVL (via scripts/prepare_aerialvl.py) has 2 Hz frames with millisecond timestamps.
TEST(TelemetryImporter, FractionalSecondTimestampsArePreserved)
{
    testutil::ScopedTempCwd tmp;
    writeFile("seq.csv",
              "num,filename,date,lat,lon,height,Omega,Kappa,Phi1,Phi2\n"
              "1,av2_0001.JPG,2023-03-18T09:02:43.084,36.5976033333,120.4294850000,277.0,0,0,90,90\n"
              "2,av2_0002.JPG,2023-03-18T09:02:43.584,36.5976100000,120.4295216667,277.0,0,0,90,90\n"
              "3,av2_0003.JPG,2023-03-18T09:02:44.0849,36.5976166667,120.4295583333,277.0,0,0,90,90\n"
              "4,av2_0004.JPG,2023-03-18T09:02:45,36.5976233333,120.4295950000,277.0,0,0,90,90\n");
    ASSERT_TRUE(importUavVisLocTelemetry("seq.csv", "av").success);
    std::vector<double> t = loadTelemetryFrameTimes("av");
    ASSERT_EQ(t.size(), 4u);
    EXPECT_NEAR(t[0], 0.0, 1e-6);
    EXPECT_NEAR(t[1], 0.5, 1e-6);    // was 0.0 before the fix (fraction dropped)
    EXPECT_NEAR(t[2], 1.0009, 1e-6); // any number of fraction digits
    EXPECT_NEAR(t[3], 1.916, 1e-6);  // a whole-second stamp mixed in still works
}

TEST(TelemetryImporter, LatLngKeepSubMetrePrecision)
{
    testutil::ScopedTempCwd tmp;
    writeFile("flight.csv", kFlightCsv);
    ASSERT_TRUE(importUavVisLocTelemetry("flight.csv", "t08").success);
    auto gt = readCsv("CSV Files/ground_truth_t08.csv");
    ASSERT_EQ(gt.size(), 3u);
    // 1e-7 deg ~ 1 cm. Six significant digits (the old bug) would be off by up to ~5e-5 deg (~5 m).
    EXPECT_NEAR(std::stod(gt[0][2]), 30.945347381234, 1e-7);
    EXPECT_NEAR(std::stod(gt[0][3]), 120.144597212345, 1e-7);
    EXPECT_NEAR(std::stod(gt[2][3]), 120.146002345678, 1e-7);
}

TEST(TelemetryImporter, HeadingsAreIndexedByFrame)
{
    testutil::ScopedTempCwd tmp;
    writeFile("flight.csv", kFlightCsv);
    ASSERT_TRUE(importUavVisLocTelemetry("flight.csv", "t08").success);
    std::vector<double> h = loadTelemetryHeadings("t08");
    ASSERT_EQ(h.size(), 221u);
    EXPECT_TRUE(std::isnan(h[0]));  // frames with no row are NaN, not 0 (0 would mean "north-up")
    EXPECT_NEAR(h[218], 98.336746, 1e-9);
    EXPECT_NEAR(h[220], -81.5, 1e-9);
}

TEST(TelemetryImporter, MissingFileFailsCleanly)
{
    testutil::ScopedTempCwd tmp;
    EXPECT_FALSE(importUavVisLocTelemetry("does_not_exist.csv", "x").success);
    EXPECT_TRUE(loadTelemetryFrameTimes("x").empty());
}

TEST(TelemetryImporter, SatelliteBoundsLookupByMapName)
{
    testutil::ScopedTempCwd tmp;
    writeFile("coords.csv",
              "mapname,LT_lat_map,LT_lon_map,RB_lat_map,RB_lon_map,region\n"
              "satellite01.tif,29.774065,115.970635,29.702283,115.996851,Changjiang-20\n"
              "satellite03.tif,32.355491,119.805926,32.290290,119.900052,Taizhou-1\n");
    SatelliteBounds b = readUavVisLocSatelliteBounds("coords.csv", "satellite03.tif");
    ASSERT_TRUE(b.success);
    EXPECT_DOUBLE_EQ(b.lt_lat, 32.355491);
    EXPECT_DOUBLE_EQ(b.lt_lon, 119.805926);
    EXPECT_DOUBLE_EQ(b.rb_lat, 32.290290);
    EXPECT_DOUBLE_EQ(b.rb_lon, 119.900052);
    EXPECT_FALSE(readUavVisLocSatelliteBounds("coords.csv", "satellite99.tif").success);
}
