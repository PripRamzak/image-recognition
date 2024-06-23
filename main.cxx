#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/highgui.hpp>
#include <opencv4/opencv2/imgcodecs.hpp>
#include <opencv4/opencv2/imgproc.hpp>

#include <iostream>

cv::RNG rng(12345);

using contour = std::vector<cv::Point>;

struct objects
{
    cv::Mat img;
    contour contour_;
    bool    have_similar;
};

int main(int argc, char** argv)
{
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

    cv::Mat gray;
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);

    cv::Mat filtred;
    cv::bilateralFilter(gray, filtred, 11, 17, 17);

    cv::Canny(filtred, filtred, 30, 90);

    cv::imshow("canny", filtred);
    cv::waitKey(0);

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7));

    cv::Mat closed;
    cv::morphologyEx(filtred, closed, cv::MORPH_CLOSE, kernel);

    std::vector<contour> contours;
    cv::findContours(
        closed, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

    std::vector<objects> templates;

    cv::Mat with_contours;
    src.copyTo(with_contours);

    for (int i = 0; i < contours.size(); i++)
        if (cv::contourArea(contours[i]) > 100.)
        {
            cv::Rect rect = cv::boundingRect(contours[i]);
            templates.push_back({ gray(cv::Range(rect.y, rect.y + rect.height),
                                       cv::Range(rect.x, rect.x + rect.width)),
                                  contours[i],
                                  false });
            cv::drawContours(
                with_contours, contours, i, cv::Scalar(0, 255, 0), 2);
        }

    imshow("contours", with_contours);
    cv::waitKey(0);

    for (int i = 0; i < templates.size(); i++)
    {
        imshow("template", templates[i].img);

        cv::Mat res(gray.rows - templates[i].img.rows + 1,
                    gray.cols - templates[i].img.cols + 1,
                    CV_32FC1);
        cv::matchTemplate(gray, templates[i].img, res, cv::TM_CCOEFF_NORMED);

        double thresh =
            templates[i].img.rows * templates[i].img.cols < 10000 ? 0.9 : 0.75;
        cv::threshold(res, res, thresh, 1., cv::THRESH_BINARY);

        res.convertTo(res, CV_8U, 255.);
        imshow("result_thresh", res);

        std::vector<cv::Rect> found_similar;
        while (true)
        {
            double    maxval, threshold = 255. * 0.8;
            cv::Point maxloc;
            minMaxLoc(res, NULL, &maxval, NULL, &maxloc);

            if (maxval >= threshold)
            {
                found_similar.emplace_back(
                    maxloc,
                    cv::Point(maxloc.x + templates[i].img.cols,
                              maxloc.y + templates[i].img.rows));
                floodFill(res, maxloc, 0);
            }
            else
                break;
        }

        if (found_similar.size() > 1)
        {
            cv::Scalar color(rng.uniform(0., 255.),
                             rng.uniform(0., 255.),
                             rng.uniform(0., 255.));

            for (auto& object : found_similar)
                for (int j = i; j < templates.size(); j++)
                    if (!templates[j].have_similar)
                    {
                        cv::Rect intersect_rect =
                            cv::boundingRect(templates[j].contour_) & object;
                        double coef =
                            static_cast<double>(intersect_rect.area()) /
                            object.area();

                        if (coef > 0.7)
                        {
                            cv::RotatedRect min_rect =
                                cv::minAreaRect(templates[j].contour_);

                            cv::Point2f points[4];
                            min_rect.points(points);

                            for (int k = 0; k < 4; k++)
                                line(src,
                                     points[k],
                                     points[(k + 1) % 4],
                                     color,
                                     2);

                            templates[j].have_similar = true;
                        }
                    }
        }

        cv::waitKey(0);
    }

    imshow("final", src);
    cv::waitKey(0);
}