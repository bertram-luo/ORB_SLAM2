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

#include "FrameDrawer.h"
#include "Tracking.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include<mutex>

namespace ORB_SLAM2
{

FrameDrawer::FrameDrawer(Map* pMap):
    mpMap(pMap),

    mNewAlgoTrackerObjectBox(cv::Rect(0, 0, 0, 0)),
    mtNewAlgoTrackerObjectBox(cv::Rect(0, 0, 0, 0)),
    mNewAlgoTrackerRadioMaxIndex(0),
    mtNewAlgoTrackerRadioMaxIndex(0),
    mNewAlgoTrackerRadioMax(0),
    mtNewAlgoTrackerRadioMax(0),
    mPointsArea(0),
    mtAreaPoints(0),
    mArea(0),
    mtArea(0),

    mKeyTrackerRadioMaxIndex(0),
    mtKeyTrackerRadioMaxIndex(0),
    mKeyTrackerRadioMax(0),
    mtKeyTrackerRadioMax(0),
    mKeyTrackerObjectBox(cv::Rect(0, 0, 0, 0)),
    mtKeyTrackerObjectBox(cv::Rect(0, 0, 0, 0)),
    mKeyTrackerIndex(0),
    mtKeyTrackerIndex(0),

    //for debug info
    mnMatchesByProjectionLastFrame(0),
    mntMatchesByProjectionLastFrame(0),

    mnMatchesByProjectionMapPointCovFrames(0),
    mntMatchesByProjectionMapPointCovFrames(0),

    //vector<bool> mvbMapPointsMatchFromLocalMap,
    //vector<bool> mvbtMapPointsMatchFromLocalMap,
    
    //vector<bool> mvbMapPointsMatchFromPreviousFrame,
    //vector<bool> mvbtMapPointsMatchFromPreviousFrame,

    mntMatchesBoth(0),
    mnMatchesBoth(0)
{
    mState=Tracking::SYSTEM_NOT_READY;
    mIm = cv::Mat(480,640,CV_8UC3, cv::Scalar(0,0,0));
}

void FrameDrawer::DrawFrame(cv::Mat& m1, cv::Mat& m2, cv::Mat& m3)
{
    cv::Mat im;
    vector<cv::KeyPoint> vIniKeys; // Initialization: KeyPoints in reference frame
    vector<int> vMatches; // Initialization: correspondeces with reference keypoints
    vector<cv::KeyPoint> vCurrentKeys; // KeyPoints in current frame
    vector<int> vbVO, vbMap; // Tracked MapPoints in current frame
    int state; // Tracking state


    //Copy variables within scoped mutex
    {
        unique_lock<mutex> lock(mMutex);

        mtBenchmarkObjectBox = mBenchmarkObjectBox;
        mtBenchmarkRadioMaxIndex = mBenchmarkRadioMaxIndex;
        mtBenchmarkRadioMax = mBenchmarkRadioMax;


        mtNewAlgoTrackerObjectBox = mNewAlgoTrackerObjectBox;
        mtNewAlgoTrackerRadioMaxIndex = mNewAlgoTrackerRadioMaxIndex;
        mtNewAlgoTrackerRadioMax = mNewAlgoTrackerRadioMax;

        mtAreaPoints = mPointsArea;
        mtArea = mArea;

        mtBefore = mBefore;
        mtAfter = mAfter;

        //mtKeyTrackerRadioMaxIndex = mKeyTrackerRadioMaxIndex;
        //mtKeyTrackerRadioMax = mKeyTrackerRadioMax;
        //mtKeyTrackerObjectBox = mKeyTrackerObjectBox;
        //mtKeyTrackerIndex = mKeyTrackerIndex;

        mntMatchesByProjectionLastFrame = mnMatchesByProjectionLastFrame;
        mntMatchesByProjectionMapPointCovFrames = mnMatchesByProjectionMapPointCovFrames;

        mvbtMapPointsMatchFromLocalMap = mvbMapPointsMatchFromLocalMap;
        mvbtMapPointsMatchFromPreviousFrame = mvbMapPointsMatchFromPreviousFrame ;

        state=mState;
        if(mState==Tracking::SYSTEM_NOT_READY)
            mState=Tracking::NO_IMAGES_YET;

        mIm.copyTo(im);

        if(mState==Tracking::NOT_INITIALIZED)
        {
            vCurrentKeys = mvCurrentKeys;
            vIniKeys = mvIniKeys;
            vMatches = mvIniMatches;
        }
        else if(mState==Tracking::OK)
        {
            vCurrentKeys = mvCurrentKeys;
            vbVO = mvbVO;
            vbMap = mvbMap;
        }
        else if(mState==Tracking::LOST)
        {
            vCurrentKeys = mvCurrentKeys;
        }
    } // destroy scoped mutex -> release mutex

    if(im.channels()<3) //this should be always true
        cvtColor(im,im,CV_GRAY2BGR);

    //Draw
    //
    mntMatchesBoth = 0;
        mnMatchesBoth = 0;
    for(int i = 0; i <mvbtMapPointsMatchFromPreviousFrame.size(); i++ ){
            bool m1 = mvbtMapPointsMatchFromPreviousFrame[i];
            bool m2 = mvbtMapPointsMatchFromLocalMap[i];
            if (m1 && m2){
                mntMatchesBoth++;
            }
    }
    if(state==Tracking::NOT_INITIALIZED) //INITIALIZING
    {
        for(unsigned int i=0; i<vMatches.size(); i++)
        {
            if(vMatches[i]>=0)
            {
                cv::line(im,vIniKeys[i].pt,vCurrentKeys[vMatches[i]].pt,
                        cv::Scalar(0,255,0));
            }
        }        
    }
    else if(state==Tracking::OK) //TRACKING
    {
        mnTracked=0;
        mnTrackedVO=0;
        const float r = 5;
        for(int i=0;i<N;i++)
        {


            if(vbVO[i] || vbMap[i])
            {
            bool m1 = mvbtMapPointsMatchFromPreviousFrame[i];
            bool m2 = mvbtMapPointsMatchFromLocalMap[i];
            if (m1 && m2){
                mnMatchesBoth++;
            }
                cv::Point2f pt1,pt2;
                pt1.x=vCurrentKeys[i].pt.x-r;
                pt1.y=vCurrentKeys[i].pt.y-r;
                pt2.x=vCurrentKeys[i].pt.x+r;
                pt2.y=vCurrentKeys[i].pt.y+r;

                // This is a match to a MapPoint in the map
                if(vbMap[i])
                {

                    if (mvbtMapPointsMatchFromLocalMap[i]){
                        if (mvbtMapPointsMatchFromPreviousFrame[i]){
                            mntMatchesBoth++;
                            cv::rectangle(im,pt1,pt2,cv::Scalar(0,0, 200));
                            cv::circle(im,vCurrentKeys[i].pt,2,cv::Scalar(0, 0, 200),-1);
                        } else {
                            cv::rectangle(im,pt1,pt2,cv::Scalar(200,0, 0));
                            cv::circle(im,vCurrentKeys[i].pt,2,cv::Scalar(200,0,0),-1);
                        }
                    
                    } else {
                        cv::rectangle(im,pt1,pt2,cv::Scalar(0,200,0));
                        cv::circle(im,vCurrentKeys[i].pt,2,cv::Scalar(0,200,0),-1);
                    
                    }
                    mnTracked++;
                }
                else // This is match to a "visual odometry" MapPoint created in the last frame
                {
                    cv::rectangle(im,pt1,pt2,cv::Scalar(255,0,0));
                    cv::circle(im,vCurrentKeys[i].pt,2,cv::Scalar(255,0,0),-1);
                    mnTrackedVO++;
                }
            }
        }
    }

    /*start*** support for tracking algorithm**/
    rectangle(im, mtBenchmarkObjectBox, Scalar(200, 0, 0), 2);
    rectangle(im, mtNewAlgoTrackerObjectBox, Scalar(0, 0, 200), 2);
    /*end**** support for tracking algorithm**/
    cv::Mat imWithInfo;
    DrawTextInfo(im,state, imWithInfo);


    imWithInfo.copyTo(m1);
    mtBefore.copyTo(m2);
    mtAfter.copyTo(m3);
    //return imWithInfo;
}


void FrameDrawer::DrawTextInfo(cv::Mat &im, int nState, cv::Mat &imText)
{
    stringstream s;
    if(nState==Tracking::NO_IMAGES_YET)
        s << " WAITING FOR IMAGES";
    else if(nState==Tracking::NOT_INITIALIZED)
        s << " TRYING TO INITIALIZE ";
    else if(nState==Tracking::OK)
    {
        if(!mbOnlyTracking)
            s << "SLAM MODE |  ";
        else
            s << "LOCALIZATION | ";
        int nKFs = mpMap->KeyFramesInMap();
        int nMPs = mpMap->MapPointsInMap();
        s << "KFs: " << nKFs << ", MPs: " << nMPs << ", Matches: " << mnTracked;
        if(mnTrackedVO>0)
            s << ", + VO matches: " << mnTrackedVO;
    }
    else if(nState==Tracking::LOST)
    {
        s << " TRACK LOST. TRYING TO RELOCALIZE ";
    }
    else if(nState==Tracking::SYSTEM_NOT_READY)
    {
        s << " LOADING ORB VOCABULARY. PLEASE WAIT...";
    }


    stringstream object_tracking_s;
    object_tracking_s << "Index" << mtBenchmarkRadioMaxIndex << "Max" << mBenchmarkRadioMax ;
    object_tracking_s << "|area points:" << mtAreaPoints <<"area"<<mtArea;
    //object_tracking_s << "| index " << mtKeyTrackerIndex << " Index " << mtKeyTrackerRadioMaxIndex << "Max" << mtKeyTrackerRadioMax;
    int baseline=0;
    cv::Size textSize = cv::getTextSize(s.str(),cv::FONT_HERSHEY_PLAIN,1,1,&baseline);

    stringstream match_info_s;
    match_info_s 
        <<"|LM nai:"<< (int)((float)mntMatchesByProjectionMapPointCovFrames/mntMatchesByProjectionLastFrame * 100) <<"%"
        << "|LM fin" <<(int)((float)mntMatchesByProjectionMapPointCovFrames/mnTracked * 100) << "%"
        <<"|BOTH fin" << (int)((float)mntMatchesBoth/mnTracked * 100) << "%"
        << "|BOTH" << mntMatchesBoth
        << ":"<<mnMatchesBoth
        << "|mth LM:"<< mntMatchesByProjectionMapPointCovFrames
        << "|mth LF:" << mntMatchesByProjectionLastFrame;


    int rowSize = textSize.height + 10;
    imText = cv::Mat(im.rows+3* rowSize,im.cols,im.type());
    im.copyTo(imText.rowRange(0,im.rows).colRange(0,im.cols));
    imText.rowRange(im.rows,im.rows + 3 * rowSize) = cv::Mat::zeros(3*(rowSize),im.cols,im.type());
    cv::putText(imText,s.str(),cv::Point(5,imText.rows-5 - 2 * rowSize),cv::FONT_HERSHEY_PLAIN,1,cv::Scalar(255,255,255),1,8);
    cv::putText(imText,object_tracking_s.str(),cv::Point(5,imText.rows-5 -rowSize),cv::FONT_HERSHEY_PLAIN,1,cv::Scalar(255,255,255),1,8);
    cv::putText(imText, match_info_s.str(),cv::Point(5,imText.rows-5),cv::FONT_HERSHEY_PLAIN,1,cv::Scalar(255,255,255),1,8);

}

void FrameDrawer::Update(Tracking *pTracker)
{
    unique_lock<mutex> lock(mMutex);
    pTracker->mImGray.copyTo(mIm);
    mvCurrentKeys=pTracker->mCurrentFrame.mvKeys;
    N = mvCurrentKeys.size();
    mvbVO = vector<int>(N,false);
    mvbMap = vector<int>(N,false);
    mbOnlyTracking = pTracker->mbOnlyTracking;


    // for trackers;
    mBenchmarkObjectBox = pTracker->mpObjectTracker->mBenchmarkObjectBox;
    mBenchmarkRadioMaxIndex = pTracker->mpObjectTracker->mBenchmarkRadioMaxIndex;
    mBenchmarkRadioMax = pTracker->mpObjectTracker->mBenchmarkRadioMax;

    mNewAlgoTrackerObjectBox = pTracker->mpObjectTracker->mNewAlgoTrackerObjectBox;
    mNewAlgoTrackerRadioMaxIndex = pTracker->mpObjectTracker->mNewAlgoTrackerRadioMaxIndex;
    mNewAlgoTrackerRadioMax = pTracker->mpObjectTracker->mNewAlgoTrackerRadioMax;

    mPointsArea = pTracker->mpObjectTracker->mPointsArea;
    mArea = pTracker->mpObjectTracker->mArea;


    mBefore = pTracker->mpObjectTracker->mBefore;
    mAfter= pTracker->mpObjectTracker->mAfter;

    //mKeyTrackerRadioMaxIndex = pTracker->mpObjectTracker->mKeyTrackerRadioMaxIndex;
    //mKeyTrackerRadioMax = pTracker->mpObjectTracker->mKeyTrackerRadioMax;
    //mKeyTrackerObjectBox = pTracker->mpObjectTracker->mKeyTrackerObjectBox;
    //mKeyTrackerIndex = pTracker->mpObjectTracker->mKeyTrackerIndex;


    // for debug;
    mnMatchesByProjectionLastFrame = pTracker->mnMatchesByProjectionLastFrame ;
    mnMatchesByProjectionMapPointCovFrames = pTracker->mnMatchesByProjectionMapPointCovFrames;

    mvbMapPointsMatchFromLocalMap = pTracker->mCurrentFrame.mvbMapPointsMatchFromLocalMap;
    mvbMapPointsMatchFromPreviousFrame = pTracker->mCurrentFrame.mvbMapPointsMatchFromPreviousFrame;

    if(pTracker->mLastProcessedState==Tracking::NOT_INITIALIZED)
    {
        mvIniKeys=pTracker->mInitialFrame.mvKeys;
        mvIniMatches=pTracker->mvIniMatches;
    }
    else if(pTracker->mLastProcessedState==Tracking::OK)
    {


        for(int i=0;i<N;i++)
        {


            MapPoint* pMP = pTracker->mCurrentFrame.mvpMapPoints[i];
            if(pMP)
            {
                if(!pTracker->mCurrentFrame.mvbOutlier[i])
                {


                    if(pMP->Observations()>0){
                        if(pMP->Observations()>=10){
                            mvbMap[i]=true;
                        }
                    }
                    //else
                    //    mvbVO[i]=true;
                }
            }
        }
    }
    mState=static_cast<int>(pTracker->mLastProcessedState);
}


} //namespace ORB_SLAM
