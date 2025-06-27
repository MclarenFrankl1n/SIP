#define _USE_MATH_DEFINES
#include <iostream>
#include <cmath>
#include <vector>
#include <map>
#include <string>
#include <fstream>
#include <sstream>
#include <algorithm>

constexpr double PI = M_PI;

// This function calculates the maximum velocity for a circular motion based on the provided parameters.
// It returns the maximum velocity in meters per second.
double calculateMaximumVelocity(double radius, double velocity, double cycleTimeInMs)
{
	double circumference = 2.0 * PI * radius; // Calculate the circumference of the Circular Motion
	double V_max = circumference / (cycleTimeInMs * 0.001);
	return V_max;
}

// This function calculates the scan time for a circular motion based on the radius.
// This includes the ramp-up, ramp-down time, and the constant velocity time.
// It returns the scan time in seconds.
double getCircularMotionCycleTime(double radius, double velocity, double acceleration, double CycleTimeInMs)
{
	double peakVelocity = calculateMaximumVelocity(radius, velocity, CycleTimeInMs); // Calculate the maximum velocity in counts per second
	double peakAcceleration = acceleration;
	double parabolicRatio = 2.0 / 3;

	double rampTime = peakVelocity / (peakAcceleration * parabolicRatio);
	double halfRampTime = rampTime / 2;

	double rampDistance = (std::pow(peakVelocity, 2) / (2 * peakAcceleration * parabolicRatio));

	double cruiseDistance = 2.0 * PI * radius;
	double cruiseTime = cruiseDistance / peakVelocity;

	double totalScanTime = (2 * rampTime) + cruiseTime;

	return totalScanTime;


}

// This function calculates the time taken for a point-to-point movement.
double calculateP2PTime(double distance, double velocity, double acceleration)
{
	double parabolicRatio = 2.0 / 3;

	double rampTime = velocity / (acceleration * parabolicRatio);

	double rampDistance = (std::pow(velocity, 2) / (2 * acceleration * parabolicRatio));

	double cruiseDistance = distance - (2 * rampDistance);
	double cruiseTime = cruiseDistance / velocity;

	double totalRampDistance = 2 * rampDistance; // Total distance covered during ramp-up and ramp-down


	//If the distance is less than or equal to the total ramp distance, the cruise phase is not needed.
	if(distance <= totalRampDistance)
	{
		velocity = std::sqrt(distance * acceleration * parabolicRatio);
		double rampTime = velocity / (acceleration * parabolicRatio);

		double totalTime = 2 * rampTime;
		return totalTime;
	}
	else
	{
		double totalTime = (2 * rampTime) + cruiseTime;
		return totalTime;
	}
}


// Returns the max time (sync time) between two axes, in seconds
double synchronizeMultiAxisMotion(
	const std::vector<double>& p1,
	const std::vector<double>& p2,
	double v_max,
	double a_max
)
{
	std::vector<double> times;
	for(size_t i = 0; i < p1.size(); ++i)
	{
		double dist = std::abs(p2[i] - p1[i]);
		double t = calculateP2PTime(dist, v_max, a_max);
		times.push_back(t);
	}
	return *std::max_element(times.begin(), times.end());
}

std::vector<std::map<std::string, double>> loadFOVFromCSV(const std::string& filename)
{
	std::vector<std::map<std::string, double>> rectangles;
	std::ifstream file(filename);
	if(!file.is_open())
	{
		std::cerr << "Could not open file: " << filename << std::endl;
		return rectangles;
	}

	std::string line;
	std::vector<std::string> headers;

	// Read header
	if(std::getline(file, line))
	{
		std::stringstream ss(line);
		std::string col;
		while(std::getline(ss, col, ','))
		{
			headers.push_back(col);
		}
	}

	// Read data rows
	while(std::getline(file, line))
	{
		std::stringstream ss(line);
		std::string cell;
		std::map<std::string, double> rect;
		size_t idx = 0;
		while(std::getline(ss, cell, ','))
		{
			if(idx < headers.size())
			{
				try
				{
					rect[headers[idx]] = std::stod(cell) / 1e9; // Convert nm to m
				}
				catch(...)
				{
					rect[headers[idx]] = 0.0;
				}
			}
			++idx;
		}
		rectangles.push_back(rect);
	}
	return rectangles;
}

double getP2PCycleTime(const std::string& filename, double velocity, double acceleration)
{
	auto FOV = loadFOVFromCSV(filename);
	std::vector<std::string> axis_keys;
	for(const auto& kv : FOV[0])
	{
		if(kv.first.size() > 0 && kv.first[0] == 'c')
		{
			axis_keys.push_back(kv.first);
		}
	}

	for(size_t i = 1; i < FOV.size(); ++i)
	{
		std::vector<double> prev, curr;
		for(const auto& k : axis_keys)
		{
			prev.push_back(FOV[i - 1].at(k));
			curr.push_back(FOV[i].at(k));
		}
		double syncTime = synchronizeMultiAxisMotion(prev, curr, velocity, acceleration);


		return syncTime; // Return the sync time in seconds
	}
}

int main()
{
	double P2P = getP2PCycleTime("D:\\SIP\\SIP\\ScanTests.csv", 1, 5);
	std::cout << "P2P Cycle Time: " << P2P << " seconds" << std::endl;

	double circularMotionCycleTime = getCircularMotionCycleTime(0.1, 1, 5, 1600);
	std::cout << "Circular Motion Cycle Time: " << circularMotionCycleTime << " seconds" << std::endl;
	return 0;
}