// Markus Buchholz
// g++ social_group_optimization_robot.cpp -o t -I/usr/include/python3.8 -lpython3.8
#include <iostream>
#include <vector>
#include <tuple>
#include <algorithm>
#include <math.h>
#include <random>

#include "matplotlibcpp.h"

namespace plt = matplotlibcpp;

//--------Path Planner--------------------------------------------------------------

float xmin = 0.0;
float xmax = 50.0;
float ymin = 0.0;
float ymax = 50.0;

float obsX = 25.0;
float obsY = 25.0;
float obsR = 3.0;

float goalX = 45.0;
float goalY = 45.0;

float startX = 2.0;
float startY = 2.0;

float K1 = 0.8;    // / obsR; // fitting parameter table 1
float K2 = 0.0001; // fitting parameter table 2

//--------------------------------------------------------------------------------
int EVOLUTIONS = 2;
int PERSONS = 1000;
float C = 0.1; // (0 - 1)

//--------------------------------------------------------------------------------

struct Pos
{

    float x;
    float y;
};

//--------------------------------------------------------------------------------

float euclid(Pos a, Pos b)
{

    return std::sqrt(std::pow(a.x - b.x, 2) + std::pow(a.y - b.y, 2));
}
//--------------------------------------------------------------------------------

float generateRandom()
{

    std::random_device engine;
    std::uniform_real_distribution<float> distrib(0, 1.0);
    return distrib(engine);
}

//--------------------------------------------------------------------------------

float valueGenerator(float low, float high)
{

    return low + generateRandom() * (high - low);
}

//--------------------------------------------------------------------------------

std::vector<float> function(std::vector<Pos> pos)
{

    std::vector<float> funcValue;
    Pos Obs{obsX, obsY};
    Pos Goal{goalX, goalY};

    for (auto &ii : pos)
    {

        funcValue.push_back(K1 * (1 / euclid(Obs, ii)) + K2 * euclid(Goal, ii));
    }

    return funcValue;
}

//--------------------------------------------------------------------------------

float func(Pos pos)
{
    Pos Obs{obsX, obsY};
    Pos Goal{goalX, goalY};

    return K1 * (1 / euclid(Obs, pos)) + K2 * euclid(Goal, pos);
}

//--------------------------------------------------------------------------------

Pos positionUpdateCheck(Pos actualPos)
{

    Pos Pnew = actualPos;

    if (Pnew.x < xmin)
    {
        Pnew.x = xmin;
    }

    if (Pnew.x > xmax)
    {
        Pnew.x = xmax;
    }

    if (Pnew.y < ymin)
    {
        Pnew.y = ymin;
    }

    if (Pnew.y > ymax)
    {
        Pnew.y = ymax;
    }

    return Pnew;
}

//--------------------------------------------------------------------------------

Pos posImproving(Pos act, Pos best)
{

    Pos Xnew;

    Xnew.x = C * act.x + generateRandom() * (best.x - act.x);
    Xnew.y = C * act.y + generateRandom() * (best.y - act.y);

    return positionUpdateCheck(Xnew);
}

//--------------------------------------------------------------------------------

Pos posAcquiring(Pos old, Pos imp, Pos imp_partner, Pos best, float actValue, float partnerValue)
{

    Pos newPos;

    float r1 = generateRandom();
    float r2 = generateRandom();

    if (actValue < partnerValue)
    {

        newPos.x = old.x + r1 * (imp.x - imp_partner.x) + r2 * (best.x - imp.x);
        newPos.y = old.y + r1 * (imp.y - imp_partner.y) + r2 * (best.y - imp.y);
    }

    else
    {
        newPos.x = old.x + r1 * (imp_partner.x - imp.x) + r2 * (best.x - imp.x);
        newPos.y = old.y + r1 * (imp_partner.y - imp.y) + r2 * (best.y - imp.y);
    }

    return positionUpdateCheck(newPos);
}

//--------------------------------------------------------------------------------

std::vector<Pos> initPosXY()
{

    std::vector<Pos> pos;

    for (int ii = 0; ii < PERSONS; ii++)
    {

        pos.push_back({valueGenerator(xmin, xmax), valueGenerator(ymin, ymax)});
    }

    return pos;
}

//-------------------------------------------------------------------------------
bool compareMin(std::pair<Pos, float> a, std::pair<Pos, float> b)
{

    return a.second < b.second;
}

//-------------------------------------------------------------------------------

// min
std::tuple<Pos, float> findBestPosFuncValue(std::vector<Pos> positions, std::vector<float> func)
{

    std::vector<std::pair<Pos, float>> best;

    for (int ii = 0; ii < func.size(); ii++)
    {

        best.push_back(std::pair<Pos, float>(positions[ii], func[ii]));
    }

    std::sort(best.begin(), best.end(), compareMin);

    return best[0];
}

//-------------------------------------------------------------------------------

int choosePartner(int actual)
{

    std::random_device engine;
    std::uniform_int_distribution<int> distribution(0, PERSONS);

    int r = -1;

    do
    {

        r = distribution(engine);

    } while (r == actual);

    return r;
}

//-------------------------------------------------------------------------------
std::tuple<std::vector<float>, std::vector<float>> gen_circle(float a, float b, float r)
{

    std::vector<float> xX;
    std::vector<float> yY;

    for (float dt = -M_PI; dt < M_PI; dt += 0.01)
    {

        xX.push_back(a + r * std::cos(dt));
        yY.push_back(b + r * std::sin(dt));
    }
    return std::make_tuple(xX, yY);
}

//-----------------------------------------------------------------------------------------

void plot2D(std::vector<float> xX, std::vector<float> yY)
{
    std::sort(xX.begin(), xX.end());
    std::sort(yY.begin(), yY.end());

    std::tuple<std::vector<float>, std::vector<float>> circle = gen_circle(obsX, obsY, obsR);

    std::vector<float> xObs = std::get<0>(circle);
    std::vector<float> yObs = std::get<1>(circle);

    plt::plot(xX, yY);
    plt::plot(xObs, yObs);
    plt::xlabel("X");
    plt::ylabel("Y");
    plt::show();
}

//------------------------------------------------------------------------------------------

std::vector<Pos> runSGO()
{

    std::vector<Pos> currentPositions = initPosXY();

    for (int ix = 0; ix < EVOLUTIONS; ix++)
    {
        std::vector<float> currentValueFunction = function(currentPositions);

        std::vector<Pos> improvingPositions(currentPositions.size(), {0.0, 0.0});

        // improving

        for (int ii = 0; ii < PERSONS - 1; ii++)
        {

            std::tuple<Pos, float> bestPosFuncValueIx = findBestPosFuncValue(currentPositions, currentValueFunction);

            Pos bestPosIx = std::get<0>(bestPosFuncValueIx);
            float bestFuncValueIx = std::get<1>(bestPosFuncValueIx);

            Pos newPosIx = posImproving(currentPositions[ii], bestPosIx);
            float newFuncValueIx = func(newPosIx);

            if (newFuncValueIx < currentValueFunction[ii])
            {

                improvingPositions[ii] = newPosIx;
                currentValueFunction[ii] = newFuncValueIx;
            }

            else
            {
                improvingPositions[ii] = currentPositions[ii];
            }
        }

        // acquiring
        std::vector<float> improvingValueFunction = function(improvingPositions);

        for (int jj = 0; jj < PERSONS - 1; jj++)
        {

            std::tuple<Pos, float> bestPosFuncValueAx = findBestPosFuncValue(improvingPositions, improvingValueFunction);

            Pos bestPosAx = std::get<0>(bestPosFuncValueAx);
            float bestFuncValueAx = std::get<1>(bestPosFuncValueAx);

            int partner = choosePartner(jj);
            Pos newPosAx = posAcquiring(currentPositions[jj], improvingPositions[jj], currentPositions[partner], bestPosAx, improvingValueFunction[jj], currentValueFunction[partner]);
            float newFuncValueAx = func(newPosAx);
            float partnerFunction = func(improvingPositions[partner]);
            if (newFuncValueAx < currentValueFunction[jj])
            {

                currentPositions[jj] = newPosAx;
            }
        }
    }

    return currentPositions;
}

//-------------------------------------------------------------------------------

int main()
{

    std::vector<Pos> path = runSGO();

    std::vector<float> xX;
    std::vector<float> yY;

    for (auto &ii : path)
    {
        xX.push_back(ii.x);
        yY.push_back(ii.y);

        std::cout << ii.x << " ," << ii.y << "\n";
    }

    plot2D(xX, yY);
}