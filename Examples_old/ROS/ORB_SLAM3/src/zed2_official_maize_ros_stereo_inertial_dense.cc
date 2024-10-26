/**
* This file is part of ORB-SLAM3
*
* Copyright (C) 2017-2021 Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
* Copyright (C) 2014-2016 Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
*
* ORB-SLAM3 is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
* License as published by the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM3 is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
* the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License along with ORB-SLAM3.
* If not, see <http://www.gnu.org/licenses/>.
*/

#include<iostream>
#include<algorithm>
#include<fstream>
#include<chrono>
#include<vector>
#include<queue>
#include<thread>
#include<mutex>
#include <nav_msgs/Path.h>
#include <nav_msgs/Odometry.h>

#include<ros/ros.h>
#include<cv_bridge/cv_bridge.h>
#include<sensor_msgs/Imu.h>

#include<opencv2/core/core.hpp>

#include"../../../include/System.h"
#include"../include/ImuTypes.h"
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>

using namespace std;

class ImuGrabber {
public:
    ImuGrabber() {};

    void GrabImu(const sensor_msgs::ImuConstPtr &imu_msg);

    queue<sensor_msgs::ImuConstPtr> imuBuf;
    std::mutex mBufMutex;
};

class ImageGrabber {
public:
    ros::NodeHandle nh;
    ros::Publisher pubRgb, pubDepth, pubTcw, pubCameraPath, pubOdom;
    nav_msgs::Path cameraPath;

    ImageGrabber(ORB_SLAM3::System *pSLAM, ImuGrabber *pImuGb, const bool bRect, const bool bClahe) : mpSLAM(pSLAM),
                                                                                                      mpImuGb(pImuGb),
                                                                                                      do_rectify(bRect),
                                                                                                      mbClahe(bClahe) {
        pubRgb = nh.advertise<sensor_msgs::Image>("/ys/RGBImage", 10);
        pubDepth = nh.advertise<sensor_msgs::Image>("/ys/DepthImage", 10);
        pubTcw = nh.advertise<geometry_msgs::PoseStamped>("/ys/CameraPose", 10);
        pubOdom = nh.advertise<nav_msgs::Odometry>("/ys/Odometry", 10);
        pubCameraPath = nh.advertise<nav_msgs::Path>("/ys/Path", 10);
    }

    void GrabImageLeft(const sensor_msgs::ImageConstPtr &msg);

    void GrabImageRight(const sensor_msgs::ImageConstPtr &msg);

    cv::Mat GetImage(const sensor_msgs::ImageConstPtr &img_msg);

    void SyncWithImu();

    queue<sensor_msgs::ImageConstPtr> imgLeftBuf, imgRightBuf;
    std::mutex mBufMutexLeft, mBufMutexRight;

    ORB_SLAM3::System *mpSLAM;
    ImuGrabber *mpImuGb;

    const bool do_rectify;
    cv::Mat M1l, M2l, M1r, M2r;

    const bool mbClahe;
    cv::Ptr<cv::CLAHE> mClahe = cv::createCLAHE(3.0, cv::Size(8, 8));
};


int main(int argc, char **argv) {
    ros::init(argc, argv, "Stereo_Inertial");
    ros::NodeHandle n("~");
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);
    bool bEqual = false;
    if (argc < 4 || argc > 5) {
        cerr << endl
             << "Usage: rosrun ORB_SLAM3 Stereo_Inertial path_to_vocabulary path_to_settings do_rectify [do_equalize]"
             << endl;
        ros::shutdown();
        return 1;
    }

    std::string sbRect(argv[3]);
    if (argc == 5) {
        std::string sbEqual(argv[4]);
        if (sbEqual == "true")
            bEqual = true;
    }

    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    ORB_SLAM3::System SLAM(argv[1], argv[2], ORB_SLAM3::System::IMU_STEREO, true);

    ImuGrabber imugb;
    ImageGrabber igb(&SLAM, &imugb, sbRect == "true", bEqual);

    if (igb.do_rectify) {
        // Load settings related to stereo calibration
        cv::FileStorage fsSettings(argv[2], cv::FileStorage::READ);
        if (!fsSettings.isOpened()) {
            cerr << "ERROR: Wrong path to settings" << endl;
            return -1;
        }

        cv::Mat K_l, K_r, P_l, P_r, R_l, R_r, D_l, D_r;
        fsSettings["LEFT.K"] >> K_l;
        fsSettings["RIGHT.K"] >> K_r;

        fsSettings["LEFT.P"] >> P_l;
        fsSettings["RIGHT.P"] >> P_r;

        fsSettings["LEFT.R"] >> R_l;
        fsSettings["RIGHT.R"] >> R_r;

        fsSettings["LEFT.D"] >> D_l;
        fsSettings["RIGHT.D"] >> D_r;

        int rows_l = fsSettings["LEFT.height"];
        int cols_l = fsSettings["LEFT.width"];
        int rows_r = fsSettings["RIGHT.height"];
        int cols_r = fsSettings["RIGHT.width"];

        if (K_l.empty() || K_r.empty() || P_l.empty() || P_r.empty() || R_l.empty() || R_r.empty() || D_l.empty() ||
            D_r.empty() ||
            rows_l == 0 || rows_r == 0 || cols_l == 0 || cols_r == 0) {
            cerr << "ERROR: Calibration parameters to rectify stereo are missing!" << endl;
            return -1;
        }

        cv::initUndistortRectifyMap(K_l, D_l, R_l, P_l.rowRange(0, 3).colRange(0, 3), cv::Size(cols_l, rows_l), CV_32F,
                                    igb.M1l, igb.M2l);
        cv::initUndistortRectifyMap(K_r, D_r, R_r, P_r.rowRange(0, 3).colRange(0, 3), cv::Size(cols_r, rows_r), CV_32F,
                                    igb.M1r, igb.M2r);
    }

    // Maximum delay, 5 seconds
    ros::Subscriber sub_imu = n.subscribe("/terrasentia/zed2/zed_node/imu/data", 1000, &ImuGrabber::GrabImu, &imugb);
    ros::Subscriber sub_img_left = n.subscribe("/zed2_front/zed_node/left/image_raw_color", 100,
                                               &ImageGrabber::GrabImageLeft, &igb);
    ros::Subscriber sub_img_right = n.subscribe("/zed2_front/zed_node/right/image_raw_color", 100,
                                                &ImageGrabber::GrabImageRight, &igb);

    std::thread sync_thread(&ImageGrabber::SyncWithImu, &igb);

    ros::spin();

    // Stop all threads,jrcv
    SLAM.Shutdown();

    // Save camera trajectory
    SLAM.SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");
    SLAM.SaveTrajectoryTUM("orbslam3_trajectory.txt");
    ros::shutdown();//jrcv

    return 0;
}


void ImageGrabber::GrabImageLeft(const sensor_msgs::ImageConstPtr &img_msg) {
    mBufMutexLeft.lock();
    if (!imgLeftBuf.empty())
        imgLeftBuf.pop();
    imgLeftBuf.push(img_msg);
    mBufMutexLeft.unlock();
}

void ImageGrabber::GrabImageRight(const sensor_msgs::ImageConstPtr &img_msg) {
    mBufMutexRight.lock();
    if (!imgRightBuf.empty())
        imgRightBuf.pop();
    imgRightBuf.push(img_msg);
    mBufMutexRight.unlock();
}

cv::Mat ImageGrabber::GetImage(const sensor_msgs::ImageConstPtr &img_msg) {
    // Copy the ros image message to cv::Mat.
    cv_bridge::CvImageConstPtr cv_ptr;
    try {
        cv_ptr = cv_bridge::toCvShare(img_msg, sensor_msgs::image_encodings::MONO8);
    }
    catch (cv_bridge::Exception &e) {
        ROS_ERROR("cv_bridge exception: %s", e.what());
    }

    if (cv_ptr->image.type() == 0) {
        return cv_ptr->image.clone();
    } else {
        std::cout << "Error type" << std::endl;
        return cv_ptr->image.clone();
    }
}

void ImageGrabber::SyncWithImu() {
    const double maxTimeDiff = 0.01;
    while (1) {
        cv::Mat imLeft, imRight;
        sensor_msgs::ImageConstPtr leftImgMsg, rightImgMsg;
        double tImLeft = 0, tImRight = 0;
        if (!imgLeftBuf.empty() && !imgRightBuf.empty() && !mpImuGb->imuBuf.empty()) {
            leftImgMsg = imgLeftBuf.front();
            rightImgMsg = imgRightBuf.front();
            tImLeft = leftImgMsg->header.stamp.toSec();
            tImRight = rightImgMsg->header.stamp.toSec();

            this->mBufMutexRight.lock();
            while ((tImLeft - tImRight) > maxTimeDiff && imgRightBuf.size() > 1) {
                imgRightBuf.pop();
                tImRight = imgRightBuf.front()->header.stamp.toSec();
            }
            this->mBufMutexRight.unlock();

            this->mBufMutexLeft.lock();
            while ((tImRight - tImLeft) > maxTimeDiff && imgLeftBuf.size() > 1) {
                imgLeftBuf.pop();
                tImLeft = imgLeftBuf.front()->header.stamp.toSec();
            }
            this->mBufMutexLeft.unlock();

            if ((tImLeft - tImRight) > maxTimeDiff || (tImRight - tImLeft) > maxTimeDiff) {
                // std::cout << "big time difference" << std::endl;
                continue;
            }
            if (tImLeft > mpImuGb->imuBuf.back()->header.stamp.toSec()) {
                continue;
            }

            this->mBufMutexLeft.lock();
            imLeft = GetImage(imgLeftBuf.front());
            imgLeftBuf.pop();
            this->mBufMutexLeft.unlock();

            this->mBufMutexRight.lock();
            imRight = GetImage(imgRightBuf.front());
            imgRightBuf.pop();
            this->mBufMutexRight.unlock();

            vector<ORB_SLAM3::IMU::Point> vImuMeas;
            mpImuGb->mBufMutex.lock();
            if (!mpImuGb->imuBuf.empty()) {
                // Load imu measurements from buffer
                vImuMeas.clear();
                while (!mpImuGb->imuBuf.empty() && mpImuGb->imuBuf.front()->header.stamp.toSec() <= tImLeft) {
                    double t = mpImuGb->imuBuf.front()->header.stamp.toSec();
                    cv::Point3f acc(mpImuGb->imuBuf.front()->linear_acceleration.x,
                                    mpImuGb->imuBuf.front()->linear_acceleration.y,
                                    mpImuGb->imuBuf.front()->linear_acceleration.z);
                    cv::Point3f gyr(mpImuGb->imuBuf.front()->angular_velocity.x,
                                    mpImuGb->imuBuf.front()->angular_velocity.y,
                                    mpImuGb->imuBuf.front()->angular_velocity.z);
                    vImuMeas.push_back(ORB_SLAM3::IMU::Point(acc, gyr, t));
                    mpImuGb->imuBuf.pop();
                }
            }
            mpImuGb->mBufMutex.unlock();
            if (mbClahe) {
                mClahe->apply(imLeft, imLeft);
                mClahe->apply(imRight, imRight);
            }

            if (do_rectify) {
                cv::remap(imLeft, imLeft, M1l, M2l, cv::INTER_LINEAR);
                cv::remap(imRight, imRight, M1r, M2r, cv::INTER_LINEAR);
            }

            bool isKeyFrame = false;
            Sophus::SE3f tcw = mpSLAM->TrackStereo(imLeft, imRight, tImLeft, isKeyFrame, vImuMeas);
            // 发布相机位姿
            Sophus::SE3f::Transformation matrix = tcw.matrix();
            cv::Mat tcwCvMat;
            cv::eigen2cv(matrix, tcwCvMat);
            std::cout << tcwCvMat <<endl;
            if (!tcwCvMat.empty()) {
                cv::Mat Twc = tcwCvMat.inv();
                cv::Mat RWC = Twc.rowRange(0, 3).colRange(0, 3);
                cv::Mat tWC = Twc.rowRange(0, 3).col(3);
                //cv::Mat TWC=orbslam->mpTracker->mCurrentFrame.mTcw.inv();
                //cv::Mat RWC= Tcw.rowRange(0,3).colRange(0,3).t();//Tcw.rowRange(0,3).colRange(0,3);
                //cv::Mat tWC=  -RWC*Tcw.rowRange(0,3).col(3);//Tcw.rowRange(0,3).col(3);

                Eigen::Matrix<double, 3, 3> eigMat;
                eigMat << RWC.at<float>(0, 0), RWC.at<float>(0, 1), RWC.at<float>(0, 2),
                        RWC.at<float>(1, 0), RWC.at<float>(1, 1), RWC.at<float>(1, 2),
                        RWC.at<float>(2, 0), RWC.at<float>(2, 1), RWC.at<float>(2, 2);
                Eigen::Quaterniond q(eigMat);

                geometry_msgs::PoseStamped tcw_msg;
                tcw_msg.pose.position.x = tWC.at<float>(0);
                tcw_msg.pose.position.y = tWC.at<float>(1);
                tcw_msg.pose.position.z = tWC.at<float>(2);


//                tf::Vector3 v1(0,1,0);
//                tf::Quaternion q1(q.x(),q.y(),q.z(),q.w());
//                q1.setRotation(v1, -M_PI/2);
//                tcw_msg.pose.orientation.x=q1[0];
//                tcw_msg.pose.orientation.y=q1[1];
//                tcw_msg.pose.orientation.z=q1[2];
//                tcw_msg.pose.orientation.w=q1[3];
                tcw_msg.pose.orientation.x = q.x();
                tcw_msg.pose.orientation.y = q.y();
                tcw_msg.pose.orientation.z = q.z();
                tcw_msg.pose.orientation.w = q.w();

// 				  tf::Matrix3x3 M(RWC.at<float>(0,0),RWC.at<float>(0,1),RWC.at<float>(0,2),
// 							      RWC.at<float>(1,0),RWC.at<float>(1,1),RWC.at<float>(1,2),
// 							      RWC.at<float>(2,0),RWC.at<float>(2,1),RWC.at<float>(2,2));
// 				  tf::Vector3 V(tWC.at<float>(0), tWC.at<float>(1), tWC.at<float>(2));
//
// 				 tf::Quaternion q;
// 				  M.getRotation(q);
//
// 			      tf::Pose tf_pose(q,V);
//
// 				   double roll,pitch,yaw;
// 				   M.getRPY(roll,pitch,yaw);
// 				   cout<<"roll: "<<roll<<"  pitch: "<<pitch<<"  yaw: "<<yaw;
// 				   cout<<"    t: "<<tWC.at<float>(0)<<"   "<<tWC.at<float>(1)<<"    "<<tWC.at<float>(2)<<endl;
//
// 				   if(roll == 0 || pitch==0 || yaw==0)
// 					return ;
                // ------

                std_msgs::Header header;
                header.stamp = leftImgMsg->header.stamp;
                header.seq = leftImgMsg->header.seq;
                header.frame_id = "world";

                sensor_msgs::ImageConstPtr rgb_msg = leftImgMsg;
                sensor_msgs::ImageConstPtr depth_msg = rightImgMsg;

                //geometry_msgs::PoseStamped tcw_msg;
                tcw_msg.header = header;
                //tf::poseTFToMsg(tf_pose, tcw_msg.pose);

                // odometry information
                nav_msgs::Odometry odom_msg;
                odom_msg.pose.pose.position.x = tWC.at<float>(0);
                odom_msg.pose.pose.position.y = tWC.at<float>(1);
                odom_msg.pose.pose.position.z = tWC.at<float>(2);

                odom_msg.pose.pose.orientation.x = q.x();
                odom_msg.pose.pose.orientation.y = q.y();
                odom_msg.pose.pose.orientation.z = q.z();
                odom_msg.pose.pose.orientation.w = q.w();

                odom_msg.header = header;
                odom_msg.child_frame_id = "base_link";
// 				 // 发布TF 变换
// 				static tf::TransformBroadcaster odom_broadcaster;  //定义tf对象
// 				geometry_msgs::TransformStamped odom_trans;  //创建一个tf发布需要使用的TransformStamped类型消息
// 				geometry_msgs::Quaternion odom_quat;   //四元数变量
//
// 				//里程计的偏航角需要转换成四元数才能发布
// 				odom_quat = tf::createQuaternionMsgFromRollPitchYaw ( roll,  pitch,  yaw);
// 				//载入坐标（tf）变换时间戳
// 				odom_trans.header.stamp = msgLeft->header.stamp;
// 				odom_trans.header.seq = msgLeft->header.seq;
// 				//发布坐标变换的父子坐标系
// 				odom_trans.header.frame_id = "odom";
// 				odom_trans.child_frame_id = "camera";
// 				//tf位置数据：x,y,z,方向
// 				odom_trans.transform.translation.x = tWC.at<float>(0);
// 				odom_trans.transform.translation.y = tWC.at<float>(1);
// 				odom_trans.transform.translation.z = tWC.at<float>(2);
// 				odom_trans.transform.rotation = odom_quat;
// 				//发布tf坐标变化
// 				odom_broadcaster.sendTransform(odom_trans);

                cameraPath.header = header;
                cameraPath.poses.push_back(tcw_msg);
                pubOdom.publish(odom_msg);
                //相机轨迹
                pubCameraPath.publish(cameraPath);
                std::cout << "isKeyFrame: " << isKeyFrame <<endl;
                if (isKeyFrame) {
                    //Tcw位姿信息
                    pubTcw.publish(tcw_msg);
                    pubRgb.publish(rgb_msg);
                    pubDepth.publish(depth_msg);
                }
            }
            // 发布相机位姿

            std::chrono::milliseconds tSleep(1);
            std::this_thread::sleep_for(tSleep);
        }
    }
}

void ImuGrabber::GrabImu(const sensor_msgs::ImuConstPtr &imu_msg) {
    mBufMutex.lock();
    imuBuf.push(imu_msg);
    mBufMutex.unlock();
    return;
}