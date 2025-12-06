// MMX example
#ifdef __x86_64__
#include <x86intrin.h>
#else
#include "sse2neon.h"
#endif
#include <unistd.h>

#include <iostream>
#include <chrono>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs.hpp>

using namespace cv;

// Computes the x component of the gradient vector
// at a given point in a image.
// returns gradient in the x direction
int xGradientAsm(Mat image, int x, int y) {
    _mm_empty();
    __m128i part1 = _mm_cvtsi32_si128((int) image.at<uchar>(y - 1, x - 1));
    __m128i part2 = _mm_cvtsi32_si128((int) image.at<uchar>(y, x - 1));
    __m128i part3 = _mm_cvtsi32_si128((int) image.at<uchar>(y + 1, x - 1));
    __m128i part4 = _mm_cvtsi32_si128((int) image.at<uchar>(y - 1, x + 1));
    __m128i part5 = _mm_cvtsi32_si128((int) image.at<uchar>(y, x + 1));
    __m128i part6 = _mm_cvtsi32_si128((int) image.at<uchar>(y + 1, x + 1));
    __m128i multiplier = _mm_cvtsi32_si128(2);
    __m128i part2s = _mm_mullo_epi16(part2, multiplier);
    __m128i part5s = _mm_mullo_epi16(part2, multiplier);
    __m128i gx = _mm_add_epi32(part1, part2s);
    __m128i gx1 = _mm_add_epi32(gx, part3);
    __m128i gx2 = _mm_sub_epi32(gx1, part4);
    __m128i gx3 = _mm_sub_epi32(gx2, part5s);
    __m128i gx4 = _mm_sub_epi32(gx3, part6);


    return _mm_cvtsi128_si32(gx4);
}


// Computes the y component of the gradient vector
// at a given point in a image
// returns gradient in the y direction
int yGradientAsm(Mat image, int x, int y) {
    _mm_empty();
    __m128i part1 = _mm_cvtsi32_si128((int) image.at<uchar>(y - 1, x - 1));
    __m128i part2 = _mm_cvtsi32_si128((int) image.at<uchar>(y - 1, x));
    __m128i part3 = _mm_cvtsi32_si128((int) image.at<uchar>(y - 1, x + 1));
    __m128i part4 = _mm_cvtsi32_si128((int) image.at<uchar>(y + 1, x - 1));
    __m128i part5 = _mm_cvtsi32_si128((int) image.at<uchar>(y + 1, x));
    __m128i part6 = _mm_cvtsi32_si128((int) image.at<uchar>(y + 1, x + 1));
    __m128i multiplier = _mm_cvtsi32_si128(2);
    __m128i part2s = _mm_mullo_epi16(part2, multiplier);
    __m128i part5s = _mm_mullo_epi16(part2, multiplier);
    __m128i gx = _mm_add_epi32(part1, part2s);
    __m128i gx1 = _mm_add_epi32(gx, part3);
    __m128i gx2 = _mm_sub_epi32(gx1, part4);
    __m128i gx3 = _mm_sub_epi32(gx2, part5s);
    __m128i gx4 = _mm_sub_epi32(gx3, part6);

    return  _mm_cvtsi128_si32(gx4);
}


int main(int argc, char* argv[]) {
    char const *src_file = "lena.bmp";
	char const *proc_file = "lena_proc.bmp";
	char const *proc2_file = "lena_proc2.bmp";

    std::chrono::time_point<std::chrono::system_clock> before;
    std::chrono::time_point<std::chrono::system_clock> after;

    Mat src, dst, asmDst;
    Mat grad_x, grad_y;
    Mat abs_grad_x, abs_grad_y;

    int gx, gy, sum;

      // Load an image
    src = imread(src_file,  IMREAD_GRAYSCALE);

    if( !src.data )
    { return -1;  }

    before = std::chrono::system_clock::now();
    GaussianBlur( src, src, Size(3,3), 0, 0, BORDER_DEFAULT );

    // Gradient X
    Sobel(src, grad_x, CV_16S, 1, 0);
    // Gradient Y
    Sobel(src, grad_y, CV_16S, 0, 1);

    convertScaleAbs( grad_x, abs_grad_x );
    convertScaleAbs( grad_y, abs_grad_y );
    addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, dst);
    after = std::chrono::system_clock::now();
    std::chrono::duration<double> nativeTimer = after - before;


    asmDst = src.clone();
    before = std::chrono::system_clock::now();
    for (int y = 1; y < src.rows - 1; y++) {
        for (int x = 1; x < src.cols - 1; x++) {
            gx = xGradientAsm(src, x, y);
            gy = yGradientAsm(src, x, y);
            sum = abs(gx) + abs(gy);
            sum = sum > 255 ? 255 : sum;
            sum = sum < 0 ? 0 : sum;
            asmDst.at<uchar>(y,x) = sum;
        }
    }
    after = std::chrono::system_clock::now();
    std::chrono::duration<double> asmTimer = after - before;

    namedWindow("C++ Sobel");
    imshow("C++ Sobel", dst);
    imwrite(proc_file, dst);

    namedWindow("Assembly Sobel");
    imshow("Assembly Sobel", asmDst);
    imwrite(proc2_file, asmDst);


    namedWindow("initial");
    imshow("initial", src);

    std::cout << "C++ timing: " << nativeTimer.count() << " Assembly timing: " << asmTimer.count() << std::endl;

    waitKey(0);

    return 0;
}

