#include "gms_matcher.h"
#include <Eigen/Dense>
#include <iostream>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <string>

// #define USE_GPU
#ifdef USE_GPU
#include <opencv2/cudafeatures2d.hpp>
using cuda::GpuMat;
#endif

/****************************************************
 * 本程序演示了如何使用2D-2D的特征匹配估计相机运动
 * **************************************************/

void find_feature_matches(const cv::Mat &img_1, const cv::Mat &img_2,
                          std::vector<cv::KeyPoint> &keypoints_1,
                          std::vector<cv::KeyPoint> &keypoints_2,
                          std::vector<cv::DMatch> &matches);

void find_gms_matches(const cv::CommandLineParser &parser, const cv::Mat &img_1,
                      const cv::Mat &img_2,
                      std::vector<cv::KeyPoint> &keypoints_1,
                      std::vector<cv::KeyPoint> &keypoints_2,
                      std::vector<cv::DMatch> &matches);

void pose_estimation_2d2d(std::vector<cv::KeyPoint> keypoints_1,
                          std::vector<cv::KeyPoint> keypoints_2,
                          std::vector<cv::DMatch> matches, cv::Mat K,
                          cv::Mat &R, cv::Mat &t);
void get_random_tf(Eigen::Matrix4d &R);
void quat2matrix(const Eigen::Vector4d &qvec, Eigen::Matrix3d &mat);
void rvec2matrix(const Eigen::Vector3d &rvec, Eigen::Matrix3d &mat);

// 像素坐标转相机归一化坐标
cv::Point2d pixel2cam(const cv::Point2d &p, const cv::Mat &K);

int main(int argc, char **argv) {
  const char *keys =
      "{ h help        |                  | print help message  }"
      "{ l left        |                  | specify left (reference) image  }"
      "{ r right       |                  | specify right (query) image }"
      "{ camera        | 0                | specify the camera device number }"
      "{ nfeatures     | 10000            | specify the maximum number of ORB "
      "features }"
      "{ fastThreshold | 20               | specify the FAST threshold }"
      "{ drawSimple    | true             | do not draw not matched keypoints }"
      "{ withRotation  | false            | take rotation into account }"
      "{ withScale     | false            | take scale into account }";

  cv::CommandLineParser cmd(argc, argv, keys);
  if (cmd.has("help"))
	{
		std::cout << "Usage: gms_matcher [options]" << std::endl;
		std::cout << "Available options:" << std::endl;
		cmd.printMessage();
		return EXIT_SUCCESS;
	}

#ifdef USE_GPU
  int flag = cuda::getCudaEnabledDeviceCount();
  if (flag != 0) {
    cuda::setDevice(0);
    std::cout << "using GPU" << endl;
  }
#endif // USE_GPU

  cv::String imgL_path = cmd.get<cv::String>("left");
  cv::String imgR_path = cmd.get<cv::String>("right");

  if (!imgL_path.empty() && !imgR_path.empty()) {
    //-- 读取图像
    cv::Mat img_1 = cv::imread(imgR_path, cv::IMREAD_COLOR);
    cv::Mat img_2 = cv::imread(imgL_path, cv::IMREAD_COLOR);

    std::vector<cv::KeyPoint> keypoints_1, keypoints_2;
    std::vector<cv::DMatch> matches;
    find_gms_matches(cmd, img_1, img_2, keypoints_1, keypoints_2, matches);
    std::cout << "一共找到了" << matches.size() << "组匹配点" << std::endl;

    cv::Mat frameMatches;
    cv::drawMatches(img_1, keypoints_1, img_2, keypoints_2, matches,
                    frameMatches, cv::Scalar::all(-1), cv::Scalar::all(-1),
                    std::vector<char>(),
                    cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    cv::imshow("Matches GMS", frameMatches);
    cv::waitKey(0);
  }
  return 0;
}

void get_random_tf(Eigen::Matrix4d &R) {
  Eigen::Vector3d angle = Eigen::Vector3d::Random() * 30;
  std::cout << "random angles:" << angle.transpose() << std::endl;
  angle = angle * M_PI / 180.0;
  Eigen::Vector3d tvec = Eigen::Vector3d::Random() * 0.1;
  Eigen::Matrix3d cam_rot =
      (Eigen::AngleAxisd(angle[0], Eigen::Vector3d::UnitX()) *
       Eigen::AngleAxisd(angle[1], Eigen::Vector3d::UnitY()) *
       Eigen::AngleAxisd(angle[2], Eigen::Vector3d::UnitZ()))
          .toRotationMatrix();
  R = Eigen::Matrix4d::Identity();
  R.block<3, 3>(0, 0) = cam_rot;
  R.block<3, 1>(0, 3) = tvec;
}

void quat2matrix(const Eigen::Vector4d &qvec, Eigen::Matrix3d &mat) {
  Eigen::Quaterniond quat(qvec[0], qvec[1], qvec[2], qvec[3]); // w, x, y, z
  mat = quat.normalized().toRotationMatrix();
}

void rvec2matrix(const Eigen::Vector3d &rvec, Eigen::Matrix3d &mat) {
  double rot_angle = rvec.norm();
  Eigen::Vector3d axis = rvec / rot_angle;
  Eigen::AngleAxisd angle_axis = Eigen::AngleAxisd(rot_angle, axis);
  mat = angle_axis.toRotationMatrix();
}

void find_gms_matches(const cv::CommandLineParser &parser, const cv::Mat &img_1,
                      const cv::Mat &img_2,
                      std::vector<cv::KeyPoint> &keypoints_1,
                      std::vector<cv::KeyPoint> &keypoints_2,
                      std::vector<cv::DMatch> &matches) {
  bool withRotation = parser.get<bool>("withRotation");
  bool withScale = parser.get<bool>("withScale");

  cv::Mat d1, d2;
  std::vector<cv::DMatch> matches_all;

  cv::Ptr<cv::ORB> orb = cv::ORB::create(parser.get<int>("nfeatures"));
  orb->setFastThreshold(parser.get<int>("fastThreshold"));
  cv::Rect roi1(0, 0, img_1.cols, int(0.8 * img_1.rows));
  cv::Mat mask1 = cv::Mat::zeros(img_1.size(), CV_8UC1);
  cv::Rect roi2(0, 0, img_2.cols, int(0.8 * img_2.rows));
  cv::Mat mask2 = cv::Mat::zeros(img_2.size(), CV_8UC1);
  mask1(roi1).setTo(255);
  mask2(roi2).setTo(255);
  orb->detectAndCompute(img_1, mask1, keypoints_1, d1);
  orb->detectAndCompute(img_2, mask2, keypoints_2, d2);

#ifdef USE_GPU
  cv::GpuMat gd1(d1), gd2(d2);
  cv::Ptr<cuda::DescriptorMatcher> matcher =
      cv::cuda::DescriptorMatcher::createBFMatcher(cv::NORM_HAMMING);
  matcher->match(gd1, gd2, matches_all);
#else
  std::cout << "using CPU" << std::endl;
  cv::BFMatcher matcher(cv::NORM_HAMMING);
  matcher.match(d1, d2, matches_all);
#endif

  // GMS filter
  cv::matchGMS(img_1.size(), img_2.size(), keypoints_1, keypoints_2,
               matches_all, matches, withRotation, withScale);
  std::cout << "matchesGMS: " << matches.size() << std::endl;
}

void find_feature_matches(const cv::Mat &img_1, const cv::Mat &img_2,
                          std::vector<cv::KeyPoint> &keypoints_1,
                          std::vector<cv::KeyPoint> &keypoints_2,
                          std::vector<cv::DMatch> &matches) {
  //-- 初始化
  cv::Mat descriptors_1, descriptors_2;
  // used in OpenCV3
  cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create();
  cv::Ptr<cv::DescriptorExtractor> descriptor = cv::ORB::create();
  // use this if you are in OpenCV2
  // cv::Ptr<cv::FeatureDetector> detector = cv::FeatureDetector::create ( "ORB"
  // );
  // cv::Ptr<cv::DescriptorExtractor> descriptor =
  // cv::DescriptorExtractor::create ( "ORB" );
  cv::Ptr<cv::DescriptorMatcher> matcher =
      cv::DescriptorMatcher::create("BruteForce-Hamming");
  //-- 第一步:检测 Oriented FAST 角点位置
  detector->detect(img_1, keypoints_1);
  detector->detect(img_2, keypoints_2);

  //-- 第二步:根据角点位置计算 BRIEF 描述子
  descriptor->compute(img_1, keypoints_1, descriptors_1);
  descriptor->compute(img_2, keypoints_2, descriptors_2);

  //-- 第三步:对两幅图像中的BRIEF描述子进行匹配，使用 Hamming 距离
  std::vector<cv::DMatch> match;
  // cv::BFMatcher matcher ( NORM_HAMMING );
  matcher->match(descriptors_1, descriptors_2, match);

  //-- 第四步:匹配点对筛选
  double min_dist = 10000, max_dist = 0;

  //找出所有匹配之间的最小距离和最大距离,
  //即是最相似的和最不相似的两组点之间的距离
  for (int i = 0; i < descriptors_1.rows; i++) {
    double dist = match[i].distance;
    if (dist < min_dist)
      min_dist = dist;
    if (dist > max_dist)
      max_dist = dist;
  }

  printf("-- Max dist : %f \n", max_dist);
  printf("-- Min dist : %f \n", min_dist);

  //当描述子之间的距离大于两倍的最小距离时,即认为匹配有误.但有时候最小距离会非常小,设置一个经验值30作为下限.
  for (int i = 0; i < descriptors_1.rows; i++) {
    if (match[i].distance <= std::max(2 * min_dist, 30.0)) {
      matches.push_back(match[i]);
    }
  }
}

cv::Point2d pixel2cam(const cv::Point2d &p, const cv::Mat &K) {
  return cv::Point2d((p.x - K.at<double>(0, 2)) / K.at<double>(0, 0),
                     (p.y - K.at<double>(1, 2)) / K.at<double>(1, 1));
}

void pose_estimation_2d2d(std::vector<cv::KeyPoint> keypoints_1,
                          std::vector<cv::KeyPoint> keypoints_2,
                          std::vector<cv::DMatch> matches, cv::Mat K,
                          cv::Mat &R, cv::Mat &t) {
  //-- 把匹配点转换为vector<cv::Point2f>的形式
  std::vector<cv::Point2f> points1;
  std::vector<cv::Point2f> points2;

  for (int i = 0; i < (int)matches.size(); i++) {
    points1.push_back(keypoints_1[matches[i].queryIdx].pt);
    points2.push_back(keypoints_2[matches[i].trainIdx].pt);
  }

  //-- 计算基础矩阵
  // cv::Mat fundamental_matrix;
  // fundamental_matrix = cv::findFundamentalMat(points1, points2,
  // CV_FM_8POINT);
  // std::cout << "fundamental_matrix is " << std::endl
  //           << fundamental_matrix << std::endl;

  //-- 计算本质矩阵
  cv::Point2d principal_point(K.at<double>(0, 2), K.at<double>(1, 2));
  double focal_length = K.at<double>(0, 0); //相机焦距
  cv::Mat essential_matrix;
  essential_matrix = cv::findEssentialMat(points1, points2, K);
  // essential_matrix = cv::findEssentialMat(points1, points2, focal_length,
  // principal_point);
  std::cout << "essential_matrix is " << std::endl
            << essential_matrix << std::endl;

  //-- 计算单应矩阵
  // cv::Mat homography_matrix;
  // homography_matrix = cv::findHomography(points1, points2, cv::RANSAC, 3);
  // std::cout << "homography_matrix is " << std::endl
  //           << homography_matrix << std::endl;

  //-- 从本质矩阵中恢复旋转和平移信息.
  cv::recoverPose(essential_matrix, points1, points2, K, R, t);
  // cv::recoverPose(essential_matrix, points1, points2, R, t, focal_length,
  // principal_point);
  std::cout << "R is " << std::endl << R << std::endl;
  std::cout << "t is " << std::endl << t << std::endl;
}
