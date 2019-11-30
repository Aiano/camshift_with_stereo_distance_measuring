#include <opencv2/core/utility.hpp>
#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/calib3d.hpp"
#include <iostream>


using namespace cv;
using namespace std;

Mat rectifyImageL, rectifyImageR, rectifyImageLL, rectifyImageRR;
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
        bm->compute(rectifyImageLL, rectifyImageRR, disp);//输入图像必须为灰度图

        Mat element = getStructuringElement(MORPH_RECT, Size(10, 10));
        dilate(disp, disp, element);

        disp.convertTo(disp8, CV_8U, 255 / ((numDisparities * 16 + 16) * 16.));//计算出的视差是CV_16S格式
        reprojectImageTo3D(disp, xyz, Q,
                           true); //在实际求距离时，ReprojectTo3D出来的X / W, Y / W, Z / W都要乘以16(也就是W除以16)，才能得到正确的三维坐标信息。
        xyz = xyz * 16;
        imshow("disparity", disp8);

}

int get_distance(Rect rec) {
        double max=0;
        for(int i=MAX(rec.y,0);i<rec.y+rec.height;i++)
                for(int j=rec.x;j<rec.x+rec.width;j++)
                        if(xyz.at<Vec3f>(Point(j,i))[2]<16000)
                                max=MAX(max,xyz.at<Vec3f>(Point(j,i))[2]);
        return (int)(max/10);
}

float hranges[] = {0, 180};
const float *phranges = hranges;
bool selectObject = false;
int trackObject = 0;
bool showHist = true;
Point origin;
Rect selection;
int vmin = 10, vmax = 256, smin = 30;

// User draws box around object to track. This triggers CAMShift to start tracking
static void onMouse(int event, int x, int y, int, void *) {
        if (selectObject) {
                selection.x = MIN(x, origin.x);
                selection.y = MIN(y, origin.y);
                selection.width = std::abs(x - origin.x);
                selection.height = std::abs(y - origin.y);

                selection &= Rect(0, 0, rectifyImageR.cols, rectifyImageR.rows);
        }

        switch (event) {
                case EVENT_LBUTTONDOWN:
                        origin = Point(x, y);
                        selection = Rect(x, y, 0, 0);
                        selectObject = true;
                        break;
                case EVENT_LBUTTONUP:
                        selectObject = false;
                        if (selection.width > 0 && selection.height > 0)
                                trackObject = -1;   // Set up CAMShift properties in main() loop
                        break;
        }
}

Mat intrMatFirst, intrMatSec, distCoeffsFirst, distCoffesSec;
Mat R, T, E, F, RFirst, RSec, PFirst, PSec;
VideoCapture cap1(2);
VideoCapture cap2(0);
Mat viewLeft, viewRight;
Rect validRoi[2];
Mat frame1, frame2;

int main() {
        Rect trackWindow;
        int hsize = 16;

        cap2 >> frame2;
        resize(frame2, viewRight, Size(320, 240));
        Mat rot_mat = cv::getRotationMatrix2D(Point2f(viewRight.cols / 2, viewRight.rows / 2), 180, 1.0);

        namedWindow("CamShift Demo", 0);
        setMouseCallback("CamShift Demo", onMouse, 0);
        namedWindow("disparity", WINDOW_NORMAL);
        setMouseCallback("disparity", onMouse, 0);
        namedWindow("remap_left",0);

        moveWindow("disparity",170,100);
        moveWindow("CamShift Demo",500,100);
        moveWindow("remap_left",830,100);

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

        Mat hsv, hue, mask, hist, histimg = Mat::zeros(200, 320, CV_8UC3), backproj;

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

        //--创建SAD窗口 Trackbar-------------------------------------------------------------------------
        //createTrackbar("BlockSize:\n", "disparity", &blockSize, 8, stereo_match_sgbm);

        //--创建视差唯一性百分比窗口 Trackbar------------------------------------------------------------
        //createTrackbar("UniquenessRatio:\n", "disparity", &uniquenessRatio, 50, stereo_match_sgbm);

        //--创建视差窗口 Trackbar------------------------------------------------------------------------
        //createTrackbar("NumDisparities:\n", "disparity", &numDisparities, 16, stereo_match_sgbm);

        //--鼠标响应函数setMouseCallback(窗口名称, 鼠标回调函数, 传给回调函数的参数，一般取0)------------


        while (true) {
                cap1 >> frame1;
                cap2 >> frame2;
                resize(frame1, viewLeft, Size(320, 240));
                resize(frame2, viewRight, Size(320, 240));
                warpAffine(viewRight, viewRight, rot_mat, viewRight.size());
                remap(viewLeft, rectifyImageL, rmapFirst[0], rmapFirst[1], INTER_LINEAR);
                remap(viewRight, rectifyImageR, rmapSec[0], rmapSec[1], INTER_LINEAR);
                cvtColor(rectifyImageL, rectifyImageLL, CV_BGR2GRAY);
                cvtColor(rectifyImageR, rectifyImageRR, CV_BGR2GRAY);

                cvtColor(rectifyImageR, hsv, COLOR_BGR2HSV);

                if (trackObject) {
                        int _vmin = vmin, _vmax = vmax;

                        inRange(hsv, Scalar(0, smin, MIN(_vmin, _vmax)),
                                Scalar(180, 256, MAX(_vmin, _vmax)), mask);
                        int ch[] = {0, 0};
                        hue.create(hsv.size(), hsv.depth());
                        mixChannels(&hsv, 1, &hue, 1, ch, 1);

                        if (trackObject < 0) {
                                // Object has been selected by user, set up CAMShift search properties once
                                Mat roi(hue, selection), maskroi(mask, selection);
                                calcHist(&roi, 1, 0, maskroi, hist, 1, &hsize, &phranges);
                                normalize(hist, hist, 0, 255, NORM_MINMAX);

                                trackWindow = selection;
                                trackObject = 1; // Don't set up again, unless user selects new ROI

                                histimg = Scalar::all(0);
                                int binW = histimg.cols / hsize;
                                Mat buf(1, hsize, CV_8UC3);
                                for (int i = 0; i < hsize; i++)
                                        buf.at<Vec3b>(i) = Vec3b(saturate_cast<uchar>(i * 180. / hsize), 255,
                                                                 255);
                                cvtColor(buf, buf, COLOR_HSV2BGR);

                                for (int i = 0; i < hsize; i++) {
                                        int val = saturate_cast<int>(hist.at<float>(i) * histimg.rows / 255);
                                        rectangle(histimg, Point(i * binW, histimg.rows),
                                                  Point((i + 1) * binW, histimg.rows - val),
                                                  Scalar(buf.at<Vec3b>(i)), -1, 8);
                                }
                        }

                        // Perform CAMShift
                        calcBackProject(&hue, 1, 0, hist, backproj, &phranges);
                        backproj &= mask;
                        RotatedRect trackBox = CamShift(backproj, trackWindow,
                                                        TermCriteria(TermCriteria::EPS | TermCriteria::COUNT,
                                                                     10, 1));
                        if (trackWindow.area() <= 1) {
                                int cols = backproj.cols, rows = backproj.rows, r = (MIN(cols, rows) + 5) / 6;
                                trackWindow = Rect(trackWindow.x - r, trackWindow.y - r,
                                                   trackWindow.x + r, trackWindow.y + r) &
                                              Rect(0, 0, cols, rows);
                        }
                        putText(rectifyImageR, "Distance: "+to_string(get_distance(trackBox.boundingRect()))+" cm", Point(20, 20), FONT_HERSHEY_PLAIN, 2.0,
                                Scalar(255, 255, 255));
                        rectangle(rectifyImageR, trackBox.boundingRect(), Scalar(0, 255, 0));
                }


                if (selectObject && selection.width > 0 && selection.height > 0) {
                        Mat roi(rectifyImageR, selection);
                        bitwise_not(roi, roi);
                }

                stereo_match_sgbm(0, 0);
                imshow("CamShift Demo", rectifyImageR);
                imshow("remap_left", rectifyImageL);
                //imshow("remap_right", rectifyImageR);

                char c = (char) waitKey(50);
                if (c == 27)
                        break;
                switch (c) {
                        case 'c':
                                trackObject = 0;
                                histimg = Scalar::all(0);
                                break;
                }
        }

        return 0;
}