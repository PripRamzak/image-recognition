#include "image_recognition.hxx"

#include <opencv4/opencv2/highgui.hpp>
#include <opencv4/opencv2/imgproc.hpp>

#include <iostream>

int main(int argc, char** argv)
{
    cv::CommandLineParser parser(
        argc, argv, "{@input | ../img/fruits5.jpg | input image}");
    cv::Mat src =
        cv::imread(cv::samples::findFile(parser.get<cv::String>("@input")),
                   cv::IMREAD_UNCHANGED);
    if (src.empty())
    {
        std::cout << "Could not open or find the image!\n" << std::endl;
        std::cout << "usage: " << argv[0] << " <Input image>" << std::endl;
        return EXIT_FAILURE;
    }

    // binarize image
    cv::Mat gray;
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    imshow("gray", gray);

    // smothing image
    cv::Mat filtred;
    cv::bilateralFilter(gray, filtred, 11, 17., 17.);

    std::vector<image_recognition::object> objects;
    image_recognition::find_objects(filtred, objects, 25., 50.);

    cv::Mat with_contours;
    src.copyTo(with_contours);
    image_recognition::draw_contours(with_contours, objects);
    cv::imshow("with_contours", with_contours);

    cv::Mat result;
    src.copyTo(result);
    image_recognition::find_similar(gray, result, objects, 0.9, 0.675);

    imshow("result", result);
    cv::waitKey(0);
}