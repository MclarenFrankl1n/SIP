#include "MotionTimeEstimator.h"
#include <iostream>
#include <cmath>
#include <vector>
#include <map>
#include <string>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <array>

#define PI 3.14159265358979323846

// Calculates the maximum velocity the cruise phase in a circular motion.
double MotionTimeEstimator::calculateMaximumVelocity(double radius, double velocity, double cycleTimeInMs)
{
	double circumference = 2.0 * PI * radius;
	double V_max = circumference / (cycleTimeInMs * 0.001);
	return V_max;
}

// Calculates the total time required for a circular motion cycle. Ramp-up + Cruise + Ramp-down
double MotionTimeEstimator::getCircularMotionCycleTime(double radius, double velocity, double acceleration, double CycleTimeInMs)
{
	double peakVelocity = calculateMaximumVelocity(radius, velocity, CycleTimeInMs);
	double peakAcceleration = acceleration;
	double parabolicRatio = 2.0 / 3;

	double rampTime = peakVelocity / (peakAcceleration * parabolicRatio);
	double rampDistance = (std::pow(peakVelocity, 2) / (2 * peakAcceleration * parabolicRatio));
	double cruiseDistance = 2.0 * PI * radius;
	double cruiseTime = cruiseDistance / peakVelocity;
	double totalScanTime = (2 * rampTime) + cruiseTime;
	return totalScanTime;
}

// Calculates the time required for a P2P movement.
double MotionTimeEstimator::calculateP2PTime(double distance, double velocity, double acceleration)
{
	double parabolicRatio = 2.0 / 3;
	double rampTime = velocity / (acceleration * parabolicRatio);
	double rampDistance = (std::pow(velocity, 2) / (2 * acceleration * parabolicRatio));
	double cruiseDistance = distance - (2 * rampDistance);
	double cruiseTime = cruiseDistance / velocity;
	double totalRampDistance = 2 * rampDistance;

	if(distance <= totalRampDistance)
	{
		velocity = std::sqrt(distance * acceleration * parabolicRatio);
		rampTime = velocity / (acceleration * parabolicRatio);
		return 2 * rampTime;
	}
	else
	{
		return (2 * rampTime) + cruiseTime;
	}
}

// Calculates the maximum time required for each axis to complete its P2P movement.
// It returns the maximum time among all axes.
double MotionTimeEstimator::synchronizeMultiAxisMotion(
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


// Calculates the starting coordinates for a circular motion (before the ramp-up phase).
// It returns the coordinates as an array of doubles.
std::array<double, 3> MotionTimeEstimator::getCircularMotionStartingPosition(
	double radius, double velocity, double acceleration, double CycleTimeInMs, double cx, double cy, double cz)
{
	double peakVelocity = calculateMaximumVelocity(radius, velocity, CycleTimeInMs);
	double parabolicRatio = 2.0 / 3.0;

	double rampDistance = (std::pow(peakVelocity, 2) / (2 * acceleration * parabolicRatio));
	double theta0 = -rampDistance / radius;
	double x0 = cx + radius * std::cos(theta0);
	double y0 = cy + radius * std::sin(theta0);
	double z0 = cz;
	return { x0, y0 , z0 };
}

// Calculates the final coordinates for a circular motion (after the ramp-down phase).
// It returns the coordinates as an array of doubles.
std::array<double, 3> MotionTimeEstimator::getCircularMotionFinalPosition(
	double radius, double velocity, double acceleration, double CycleTimeInMs, double cx, double cy, double cz)
{
	double peakVelocity = calculateMaximumVelocity(radius, velocity, CycleTimeInMs);
	double parabolicRatio = 2.0 / 3.0;

	double rampDistance = (std::pow(peakVelocity, 2) / (2 * acceleration * parabolicRatio));
	double theta1 = (2.0 * PI * radius + rampDistance) / radius;
	double x1 = cx + radius * std::cos(theta1);
	double y1 = cy + radius * std::sin(theta1);
	double z1 = cz;
	return { x1, y1 , z1 };
}

// Creates a vector of P2PCoordinates in between each circular motion
std::vector<MotionTimeEstimator::P2PCoordinates> MotionTimeEstimator::createP2PCoordinates(
    const std::vector<MotionTimeEstimator::FOVScanPoint>& fovPoints,
    double velocity,
    double acceleration
)
{
    std::vector<std::array<double, 3>> starts, finals;
    for(const auto& fov : fovPoints)
    {
        starts.push_back(getCircularMotionStartingPosition(
            fov.radius, velocity, acceleration, fov.cycleTimeInMs, fov.cx, fov.cy, fov.cz));
        finals.push_back(getCircularMotionFinalPosition(
            fov.radius, velocity, acceleration, fov.cycleTimeInMs, fov.cx, fov.cy, fov.cz));
    }

    std::vector<P2PCoordinates> segments;
    if(!fovPoints.empty())
    {
        segments.push_back({ {fovPoints[0].cx, fovPoints[0].cy, fovPoints[0].cz}, starts[0], fovPoints[0].radius, fovPoints[0].cycleTimeInMs });
        std::cout << "Segment 0: from (" << fovPoints[0].cx << "," << fovPoints[0].cy << "," << fovPoints[0].cz
            << ") to (" << starts[0][0] << "," << starts[0][1] << "," << starts[0][2]
            << "), radius=" << fovPoints[0].radius << ", cycleTime=" << fovPoints[0].cycleTimeInMs << std::endl;
    }
    for(size_t i = 0; i + 1 < fovPoints.size(); ++i)
    {
        segments.push_back({ finals[i], starts[i + 1], fovPoints[i + 1].radius, fovPoints[i + 1].cycleTimeInMs });
        std::cout << "Segment " << (i + 1) << ": from (" << finals[i][0] << "," << finals[i][1] << "," << finals[i][2]
            << ") to (" << starts[i + 1][0] << "," << starts[i + 1][1] << "," << starts[i + 1][2]
            << "), radius=" << fovPoints[i + 1].radius << ", cycleTime=" << fovPoints[i + 1].cycleTimeInMs << std::endl;
    }
    return segments;
}

// Computes the P2P sync time for a single P2P movement
double MotionTimeEstimator::getP2PCycleTime(
	const std::array<double, 3>& from,
	const std::array<double, 3>& to,
	double velocity,
	double acceleration
)
{
	std::vector<double> v_from(from.begin(), from.end());
	std::vector<double> v_to(to.begin(), to.end());
	double p2pTime = synchronizeMultiAxisMotion(v_from, v_to, velocity, acceleration);
	return p2pTime;
}

// This function calculates the total time for all segments, including P2P and circular motion
double MotionTimeEstimator::getEstimatedOTFCycleTime(
	const std::vector<MotionTimeEstimator::FOVScanPoint>& fovPoints,
	double velocity,
	double acceleration
)
{
	std::vector<MotionTimeEstimator::P2PCoordinates> segments = createP2PCoordinates(fovPoints, velocity, acceleration);
	double totalTime = 0.0;
	for(size_t i = 0; i < segments.size(); ++i)
	{
		const auto& seg = segments[i];
		double p2pTime = getP2PCycleTime(seg.from, seg.to, velocity, acceleration);
		double scanTime = getCircularMotionCycleTime(
			seg.radius, velocity, acceleration, seg.cycleTimeInMs);

		double fovCycleTime = p2pTime + scanTime;
		std::cout << "FOV " << i << " P2P time: " << p2pTime
			<< ", scan time: " << scanTime
			<< ", FOV cycle time: " << fovCycleTime << " seconds" << std::endl;
		totalTime += fovCycleTime;
	}
	std::cout << "Total time: " << totalTime << " seconds" << std::endl;
	return totalTime;
}
