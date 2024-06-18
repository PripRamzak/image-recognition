#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/highgui.hpp>
#include <opencv4/opencv2/imgcodecs.hpp>
#include <opencv4/opencv2/imgproc.hpp>

#include <iostream>

cv::Mat difference_of_gaussians(
    cv::Mat& src, int k1, double s1, int k2, double s2);
cv::Mat substract_no_saturation(cv::Mat& mat1, cv::Mat& mat2);

void thresh_callback(int, void*);

/**
 * @function main
 */
int main(int argc, char** argv)
{
    //! [setup]
    /// Load source image
    cv::CommandLineParser parser(
        argc, argv, "{@input | ../img/vegetables.jpg | input image}");
    cv::Mat src =
        cv::imread(cv::samples::findFile(parser.get<cv::String>("@input")),
                   cv::IMREAD_UNCHANGED);
    if (src.empty())
    {
        std::cout << "Could not open or find the image!\n" << std::endl;
        std::cout << "usage: " << argv[0] << " <Input image>" << std::endl;
        return -1;
    }

    cv::Mat src_gray;
    cv::cvtColor(src, src_gray, cv::COLOR_BGR2GRAY);
    cv::Mat copy = src.clone();

    cv::Mat dog_output = difference_of_gaussians(src_gray, 5, 5., 21, 19.);

    cv::imshow("DoG", dog_output);

    cv::Mat threshold_output;
    cv::threshold(dog_output,
                  threshold_output,
                  127,
                  255,
                  cv::THRESH_BINARY + cv::THRESH_OTSU);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(
        threshold_output, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

    for (auto& contour : contours)
    {
        double area = cv::contourArea(contour);
        if (area > 1500)
        {
            cv::Rect r      = cv::boundingRect(contour);
            double   extent = area / (r.width * r.height);
            if (extent > 0.6)
            {
                cv::RotatedRect min_r = cv::minAreaRect(contour);

                cv::Point2f points[4];
                min_r.points(points);
                for (int i = 0; i < 4; i++)
                    cv::line(copy,
                             points[i],
                             points[(i + 1) % 4],
                             cv::Scalar(0, 255, 0),
                             4);
            }
        }
    }

    cv::imshow("result", copy);

    cv::waitKey();
    return 0;
}

cv::Mat difference_of_gaussians(
    cv::Mat& src, int k1, double s1, int k2, double s2)
{
    cv::Mat gaus_blur_output1;
    cv::Mat gaus_blur_output2;
    cv::GaussianBlur(src, gaus_blur_output1, cv::Size{ k1, k1 }, s1);
    cv::GaussianBlur(src, gaus_blur_output2, cv::Size{ k2, k2 }, s2);

    return substract_no_saturation(gaus_blur_output1, gaus_blur_output2);
}

cv::Mat substract_no_saturation(cv::Mat& mat1, cv::Mat& mat2)
{
    assert(mat1.rows == mat2.rows || mat1.cols == mat2.cols);

    cv::Mat result(mat1.rows, mat1.cols, mat1.type());
    for (int row = 0; row < mat1.rows; row++)
        for (int col = 0; col < mat1.cols; col++)
            result.at<uint8_t>(row, col) =
                mat1.at<uint8_t>(row, col) - mat2.at<uint8_t>(row, col);

    return result;
}