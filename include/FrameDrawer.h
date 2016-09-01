/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef FRAMEDRAWER_H
#define FRAMEDRAWER_H

#include "Tracking.h"
#include "MapPoint.h"
#include "Map.h"

#include<opencv2/core/core.hpp>
#include<opencv2/features2d/features2d.hpp>

#include<mutex>


namespace ORB_SLAM2
{

class Tracking;
class Viewer;

class FrameDrawer
{
public:
    FrameDrawer(Map* pMap);

    // Update info from the last processed frame.
    void Update(Tracking *pTracker);

    // Draw last processed frame.
    void DrawFrame(cv::Mat& m1, cv::Mat& m2, cv::Mat& m3);

protected:

    void DrawTextInfo(cv::Mat &im, int nState, cv::Mat &imText);

    // Info of the frame to be drawn
    cv::Mat mIm;
    int N;
    vector<cv::KeyPoint> mvCurrentKeys;
    vector<int> mvbMap, mvbVO;
    bool mbOnlyTracking;
    int mnTracked, mnTrackedVO;
    vector<cv::KeyPoint> mvIniKeys;
    vector<int> mvIniMatches;
    int mState;

    Map* mpMap;

    std::mutex mMutex;

    cv::Rect mBenchmarkObjectBox;
    cv::Rect mtBenchmarkObjectBox;
    int mBenchmarkRadioMaxIndex;
    int mtBenchmarkRadioMaxIndex;
    float mBenchmarkRadioMax;
    float mtBenchmarkRadioMax;

    cv::Rect mNewAlgoTrackerObjectBox;
    cv::Rect mtNewAlgoTrackerObjectBox;
    int mNewAlgoTrackerRadioMaxIndex;
    int mtNewAlgoTrackerRadioMaxIndex;
    float mNewAlgoTrackerRadioMax;
    float mtNewAlgoTrackerRadioMax;

    float mAreaPoints;
    float mtAreaPoints;
    float mArea;
    float mtArea;


    cv::Mat mBefore;
    cv::Mat mtBefore;
    cv::Mat mAfter;
    cv::Mat mtAfter;

    int mKeyTrackerRadioMaxIndex;
    int mtKeyTrackerRadioMaxIndex;
    float mKeyTrackerRadioMax;
    float mtKeyTrackerRadioMax;
    cv::Rect mKeyTrackerObjectBox;
    cv::Rect mtKeyTrackerObjectBox;
    int mKeyTrackerIndex;
    int mtKeyTrackerIndex;

    //for debug info
    int mnMatchesByProjectionLastFrame;
    int mntMatchesByProjectionLastFrame;

    int mnMatchesByProjectionMapPointCovFrames;
    int mntMatchesByProjectionMapPointCovFrames;

    vector<bool> mvbMapPointsMatchFromLocalMap;
    vector<bool> mvbtMapPointsMatchFromLocalMap;
    
    vector<bool> mvbMapPointsMatchFromPreviousFrame;
    vector<bool> mvbtMapPointsMatchFromPreviousFrame;

    int mntMatchesBoth;
    int mnMatchesBoth;
};

} //namespace ORB_SLAM

#endif // FRAMEDRAWER_H
