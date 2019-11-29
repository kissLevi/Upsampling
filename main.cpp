#include <iostream>
#include <opencv2/opencv.hpp>

float gaussian(float x, float o);

float gaussian(cv::Point2d x, float o);

cv::Vec3f w(cv::Mat* image, cv::Point u,cv::Point v, float sigma1, float sigma2);

int main(int argc, char** argv) {

    if(argc != 4) {
        std::cout << "Parameters: RGB input image, Dept/Disparity image pair" << std::endl;
        return -1;
    }
    cv::Mat inputImage = cv::imread(argv[2]);

    cv::Mat inputImage2 = cv::imread(argv[2]);
    cv::Mat inputDisparity = cv::imread(argv[3]);

    if(!inputImage.data) {
        std::cout << "Error while reading image" << std::endl;
        return -1;
    }

    int filterSize = 9;
    filterSize /= 2;

    float sigma1 = 100;
    float sigma2 = 100;

    bool saveToFile = false;
    bool performJoint = true;

    cv::Mat bilateralOut = cv::Mat::zeros(inputImage.rows,inputImage.cols,inputImage.type());
    cv::Mat bilateralJointOut = cv::Mat::zeros(inputDisparity.rows,inputDisparity.cols,inputDisparity.type());


    for(int i = filterSize;i<(inputImage.rows-filterSize);++i){
        for(int j=filterSize;j<(inputImage.cols-filterSize);++j){
            cv::Vec3f newPoint;
            cv::Vec3f weight;

            cv::Vec3f newPoint2;
            cv::Vec3f weight2;
            for(int k = -filterSize;k<= filterSize;++k){
                for(int l = -filterSize;l<= filterSize;++l){
                    //Bilateral
                    cv::Vec3f tmp = w(&inputImage, cv::Point(j,i), cv::Point(j+l,i+k),sigma1, sigma2);
                    newPoint[0] += (float)inputImage.at<cv::Vec3b>(i+k,j+l)[0]*tmp[0];
                    weight[0] += tmp[0];
                    newPoint[1] += (float)inputImage.at<cv::Vec3b>(i+k,j+l)[1]*tmp[1];
                    weight[1] += tmp[1];
                    newPoint[2] += (float)inputImage.at<cv::Vec3b>(i+k,j+l)[2]*tmp[2];
                    weight[2] += tmp[2];
                    if(performJoint) {
                        //Joint Bilateral
                        cv::Vec3f tmp2 = w(&inputImage2, cv::Point(j, i), cv::Point(j+l, i+k),sigma1, sigma2);
                        newPoint2[0] += (float)inputDisparity.at<cv::Vec3b>(i+k,j+l)[0]*tmp2[0];
                        weight2[0] += tmp2[0];
                        newPoint2[1] += (float)inputDisparity.at<cv::Vec3b>(i+k,j+l)[1]*tmp2[1];
                        weight2[1] += tmp2[1];
                        newPoint2[2] += (float)inputDisparity.at<cv::Vec3b>(i+k,j+l)[2]*tmp2[2];
                        weight2[2] += tmp2[2];
                    }
                }
            }
            //Bilateral
            newPoint[0] /= weight[0];
            newPoint[1] /= weight[1];
            newPoint[2] /= weight[2];
            bilateralOut.at<cv::Vec3b>(i,j) = newPoint;

            if (performJoint) {
                //Joint Bilateral
                newPoint2[0] /= weight2[0];
                newPoint2[1] /= weight2[1];
                newPoint2[2] /= weight2[2];
                bilateralJointOut.at<cv::Vec3b>(i,j) = newPoint2;
            }
        }
    }
    if (saveToFile) {
        std::ostringstream ss;
        ss << "out/spectral" << sigma1 << "spatial" << sigma2 << ".jpg";
        cv::imwrite(ss.str(), bilateralOut);
    }


    cv::namedWindow("Input");
    cv::imshow("Input", inputImage);

    cv::namedWindow("OutputBilateral");
    cv::imshow("OutputBilateral", bilateralOut);

    if(performJoint) {
        cv::namedWindow("OutputJointBilateral");
        cv::imshow("OutputJointBilateral", bilateralJointOut);
    }
    cv::waitKey(0);
    return 0;
}

float gaussian(float x, float o) {
    return (1.0 / (o *sqrtf(CV_2PI)))*expf(-1.0*(powf(x,2)/(2.0*powf(o,2))));
}

float gaussian(cv::Point2d x, float o){
    return (1.0 / (powf(o,2) *CV_2PI))*expf(-1*(((powf(x.x,2)+powf(x.y,2))/2.0*powf(o,2))));
}

cv::Vec3f w(cv::Mat* image, cv::Point u,cv::Point v, float sigma1, float sigma2){
    cv::Vec3f difference;
    difference[0] = (float)image->at<cv::Vec3b>(u)[0]-(float)image->at<cv::Vec3b>(v)[0];

    difference[1] = (float)image->at<cv::Vec3b>(u)[1]-(float)image->at<cv::Vec3b>(v)[1];
    difference[2] = (float)image->at<cv::Vec3b>(u)[2]-(float)image->at<cv::Vec3b>(v)[2];
    //std::cout << difference << std::endl;
    //return gaussian(sqrtf(powf(difference[0],2) + powf(difference[1],2) + powf(difference[2],2)),sigma1)*gaussian(v,sigma2);


    float gaussian2 = gaussian(sqrtf(powf(u.x-v.x,2) + powf(u.y-v.y,2)),sigma2);
    cv::Vec3f out;
    out[0] = gaussian(sqrtf(powf(difference[0],2)),sigma1)*gaussian2;
    out[1] = gaussian(sqrtf(powf(difference[1],2)),sigma1)*gaussian2;
    out[2] = gaussian(sqrtf(powf(difference[2],2)),sigma1)*gaussian2;
    return out;
}