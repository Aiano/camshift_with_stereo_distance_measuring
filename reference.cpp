#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;

Mat rectifyImageL, rectifyImageR;


Rect validROIL;                                   //图像校正之后，会对图像进行裁剪，这里的validROI就是指裁剪之后的区域
Rect validROIR;
Mat Q;
Mat xyz;                                          //三维坐标
Ptr<StereoBM> bm = StereoBM::create(16, 9);
int blockSize = 8, uniquenessRatio = 20, numDisparities = 3;

void stereo_match_sgbm(int, void *)                                         //SGBM匹配算法
{
        bm->setBlockSize(2 * blockSize + 5);     //SAD窗口大小，5~21之间为宜
        bm->setROI1(validROIL);
        bm->setROI2(validROIR);
        //bm->setPreFilterType(StereoBM::PREFILTER_XSOBEL);
        bm->setPreFilterCap(31);
        bm->setMinDisparity(0);  //最小视差，默认值为0, 可以是负值，int型
        bm->setNumDisparities(numDisparities * 16 + 16);//视差窗口，即最大视差值与最小视差值之差,窗口大小必须是16的整数倍，int型
        bm->setTextureThreshold(10);
        bm->setUniquenessRatio(uniquenessRatio);//uniquenessRatio主要可以防止误匹配
        bm->setSpeckleWindowSize(100);
        bm->setSpeckleRange(32);
        bm->setDisp12MaxDiff(-1);
        Mat disp, disp8;
        bm->compute(rectifyImageL, rectifyImageR, disp);//输入图像必须为灰度图

        Mat element = getStructuringElement(MORPH_RECT, Size(10, 10));
        dilate(disp, disp, element);

        disp.convertTo(disp8, CV_8U, 255 / ((numDisparities * 16 + 16) * 16.));//计算出的视差是CV_16S格式
        reprojectImageTo3D(disp, xyz, Q,
                           true); //在实际求距离时，ReprojectTo3D出来的X / W, Y / W, Z / W都要乘以16(也就是W除以16)，才能得到正确的三维坐标信息。
        xyz = xyz * 16;
        imshow("disparity", disp8);

}


Point origin;                                     //鼠标按下的起始点
Rect selection;                                   //定义矩形选框
bool selectObject = false;                        //是否选择对象

//--描述：鼠标操作回调--------------------------------------------------
static void onMouse(int event, int x, int y, int, void *) {
        if (selectObject) {
                selection.x = MIN(x, origin.x);
                selection.y = MIN(y, origin.y);
                selection.width = std::abs(x - origin.x);
                selection.height = std::abs(y - origin.y);
        }

        switch (event) {
                case EVENT_LBUTTONDOWN:   //鼠标左按钮按下的事件
                        origin = Point(x, y);
                        selection = Rect(x, y, 0, 0);
                        selectObject = true;
                        cout << origin << "in world coordinate is: " << xyz.at<Vec3f>(origin) << endl;
                        break;
                case EVENT_LBUTTONUP:    //鼠标左按钮释放的事件
                        selectObject = false;
                        if (selection.width > 0 && selection.height > 0)
                                break;
        }

}

Mat intrMatFirst, intrMatSec, distCoeffsFirst, distCoffesSec;
Mat R, T, E, F, RFirst, RSec, PFirst, PSec;
VideoCapture cap1(4);
VideoCapture cap2(2);
Mat viewLeft, viewRight;
Rect validRoi[2];


int main() {


        Mat frame1, frame2;
        //viewLeft=imread(R"(/home/aiano/CLionProjects/get_picture_and_calibrate_camera/cmake-build-debug/pictures/left_23.png)");
        //viewRight=imread(R"(/home/aiano/CLionProjects/get_picture_and_calibrate_camera/cmake-build-debug/pictures/right_23.png)");
        cap2 >> frame2;
        resize(frame2, viewRight, Size(320, 240));
        Mat rot_mat = cv::getRotationMatrix2D(Point2f(viewRight.cols / 2, viewRight.rows / 2), 180, 1.0);

        FileStorage fs(R"(/home/aiano/CLionProjects/stereo_distance_measure/intrinsics.yml)", FileStorage::READ);
        if (fs.isOpened()) {
                cout << "read";
                fs["M1"] >> intrMatFirst;
                fs["D1"] >> distCoeffsFirst;
                fs["M2"] >> intrMatSec;
                fs["D2"] >> distCoffesSec;

                fs["R"] >> R;
                fs["T"] >> T;
                fs["Q"] >> Q;
                cout << "M1" << intrMatFirst << endl << distCoeffsFirst;

                fs.release();
        } else {
                cerr << "Can't open the file." << endl;
                return -1;
        }

        Size imageSize = Size(320, 240);
        cout << "stereo rectify..." << endl;
        stereoRectify(intrMatFirst, distCoeffsFirst, intrMatSec, distCoffesSec, imageSize, R, T, RFirst,
                      RSec, PFirst, PSec, Q, CALIB_ZERO_DISPARITY, -1, imageSize, &validROIL, &validROIR);
        Mat rmapFirst[2], rmapSec[2], rviewFirst, rviewSec;
        initUndistortRectifyMap(intrMatFirst, distCoeffsFirst, RFirst, PFirst,
                                imageSize, CV_16SC2, rmapFirst[0], rmapFirst[1]);//CV_16SC2
        initUndistortRectifyMap(intrMatSec, distCoffesSec, RSec, PSec,//CV_16SC2
                                imageSize, CV_16SC2, rmapSec[0], rmapSec[1]);
        //stereoRectify

        //--显示结果-------------------------------------------------------------------------------------
        namedWindow("disparity", WINDOW_NORMAL);

        //--创建SAD窗口 Trackbar-------------------------------------------------------------------------
        createTrackbar("BlockSize:\n", "disparity", &blockSize, 8, stereo_match_sgbm);

        //--创建视差唯一性百分比窗口 Trackbar------------------------------------------------------------
        createTrackbar("UniquenessRatio:\n", "disparity", &uniquenessRatio, 50, stereo_match_sgbm);

        //--创建视差窗口 Trackbar------------------------------------------------------------------------
        createTrackbar("NumDisparities:\n", "disparity", &numDisparities, 16, stereo_match_sgbm);

        //--鼠标响应函数setMouseCallback(窗口名称, 鼠标回调函数, 传给回调函数的参数，一般取0)------------
        setMouseCallback("disparity", onMouse, 0);

        char ch;
        while (1) {
                cap1 >> frame1;
                cap2 >> frame2;
                resize(frame1, viewLeft, Size(320, 240));
                resize(frame2, viewRight, Size(320, 240));
                warpAffine(viewRight, viewRight, rot_mat, viewRight.size());

                remap(viewLeft, rectifyImageL, rmapFirst[0], rmapFirst[1], INTER_LINEAR);
                remap(viewRight, rectifyImageR, rmapSec[0], rmapSec[1], INTER_LINEAR);
                cvtColor(rectifyImageL, rectifyImageL, CV_BGR2GRAY);
                cvtColor(rectifyImageR, rectifyImageR, CV_BGR2GRAY);
                //

                //

                imshow("remap_left", rectifyImageL);
                imshow("remap_right", rectifyImageR);


                ch = waitKey(23);
                if (ch == 27) {
                        break;
                } else if (ch == 10) {
                        cout << "done" << endl;
                        stereo_match_sgbm(0, 0);
                } else if (ch == 't') {
                        viewLeft = imread(
                                R"(/home/aiano/CLionProjects/stereo_distance_measure/cmake-build-debug/aloeL.jpg)",
                                IMREAD_GRAYSCALE);
                        viewRight = imread(
                                R"(/home/aiano/CLionProjects/stereo_distance_measure/cmake-build-debug/aloeR.jpg)",
                                IMREAD_GRAYSCALE);
                        stereo_match_sgbm(0, 0);
                }
        }

}





