#include "Utilities.h"
//#include "Histograms.cpp"
#include <iostream>
#include <fstream>
#include <list>
#include <experimental/filesystem> // C++-standard header file name
#include <filesystem> // Microsoft-specific implementation header file name
using namespace std::experimental::filesystem::v1;
using namespace std;

#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/calib3d.hpp"

using namespace cv;
using namespace std;

// Data provided:  Filename, White pieces, Black pieces
// Note that this information can ONLY be used to evaluate performance.  It must not be used during processing of the images.
const string GROUND_TRUTH_FOR_BOARD_IMAGES[][3] = {
	{"DraughtsGame1Move0.JPG", "1,2,3,4,5,6,7,8,9,10,11,12", "21,22,23,24,25,26,27,28,29,30,31,32"},
	{"DraughtsGame1Move1.JPG", "1,2,3,4,5,6,7,8,10,11,12,13", "21,22,23,24,25,26,27,28,29,30,31,32"},
	{"DraughtsGame1Move2.JPG", "1,2,3,4,5,6,7,8,10,11,12,13", "20,21,22,23,25,26,27,28,29,30,31,32"},
	{"DraughtsGame1Move3.JPG", "1,2,3,4,5,7,8,9,10,11,12,13", "20,21,22,23,25,26,27,28,29,30,31,32"},
	{"DraughtsGame1Move4.JPG", "1,2,3,4,5,7,8,9,10,11,12,13", "17,20,21,23,25,26,27,28,29,30,31,32"},
	{"DraughtsGame1Move5.JPG", "1,2,3,4,5,7,8,9,10,11,12,22", "20,21,23,25,26,27,28,29,30,31,32"},
	{"DraughtsGame1Move6.JPG", "1,2,3,4,5,7,8,9,10,11,12", "17,20,21,23,25,27,28,29,30,31,32"},
	{"DraughtsGame1Move7.JPG", "1,2,3,4,5,7,8,10,11,12,13", "17,20,21,23,25,27,28,29,30,31,32"},
	{"DraughtsGame1Move8.JPG", "1,2,3,4,5,7,8,10,11,12,13", "17,20,21,23,25,26,27,28,29,31,32"},
	{"DraughtsGame1Move9.JPG", "1,2,3,4,5,7,8,10,11,12,22", "20,21,23,25,26,27,28,29,31,32"},
	{"DraughtsGame1Move10.JPG", "1,2,3,4,5,7,8,10,11,12", "18,20,21,23,26,27,28,29,31,32"},
	{"DraughtsGame1Move11.JPG", "1,2,3,4,5,7,8,10,11,16", "18,20,21,23,26,27,28,29,31,32"},
	{"DraughtsGame1Move12.JPG", "1,2,3,4,5,7,8,10,11,16", "14,20,21,23,26,27,28,29,31,32"},
	{"DraughtsGame1Move13.JPG", "1,2,3,4,5,7,8,11,16,17", "20,21,23,26,27,28,29,31,32"},
	{"DraughtsGame1Move14.JPG", "1,2,3,4,5,7,8,11,16", "14,20,23,26,27,28,29,31,32"},
	{"DraughtsGame1Move15.JPG", "1,3,4,5,6,7,8,11,16", "14,20,23,26,27,28,29,31,32"},
	{"DraughtsGame1Move16.JPG", "1,3,4,5,6,7,8,11,16", "14,20,22,23,27,28,29,31,32"},
	{"DraughtsGame1Move17.JPG", "1,3,4,5,7,8,9,11,16", "14,20,22,23,27,28,29,31,32"},
	{"DraughtsGame1Move18.JPG", "1,3,4,5,7,8,9,11,16", "14,18,20,23,27,28,29,31,32"},
	{"DraughtsGame1Move19.JPG", "1,3,4,5,7,8,9,15,16", "14,18,20,23,27,28,29,31,32"},
	{"DraughtsGame1Move20.JPG", "1,3,4,5,8,9,16", "K2,14,20,23,27,28,29,31,32"},
	{"DraughtsGame1Move21.JPG", "1,3,4,5,8,16,18", "K2,20,23,27,28,29,31,32"},
	{"DraughtsGame1Move22.JPG", "1,3,4,5,8,16", "K2,14,20,27,28,29,31,32"},
	{"DraughtsGame1Move23.JPG", "1,4,5,7,8,16", "K2,14,20,27,28,29,31,32"},
	{"DraughtsGame1Move24.JPG", "1,4,5,7,8", "K2,11,14,27,28,29,31,32"},
	{"DraughtsGame1Move25.JPG", "1,4,5,8,16", "K2,14,27,28,29,31,32"},
	{"DraughtsGame1Move26.JPG", "1,4,5,8,16", "K7,14,27,28,29,31,32"},
	{"DraughtsGame1Move27.JPG", "1,4,5,11,16", "K7,14,27,28,29,31,32"},
	{"DraughtsGame1Move28.JPG", "1,4,5,11,16", "K7,14,24,28,29,31,32"},
	{"DraughtsGame1Move29.JPG", "4,5,6,11,16", "K7,14,24,28,29,31,32"},
	{"DraughtsGame1Move30.JPG", "4,5,6,11,16", "K2,14,24,28,29,31,32"},
	{"DraughtsGame1Move31.JPG", "4,5,9,11,16", "K2,14,24,28,29,31,32"},
	{"DraughtsGame1Move32.JPG", "4,5,9,11,16", "K2,10,24,28,29,31,32"},
	{"DraughtsGame1Move33.JPG", "4,5,11,14,16", "K2,10,24,28,29,31,32"},
	{"DraughtsGame1Move34.JPG", "4,5,11,14,16", "K2,7,24,28,29,31,32"},
	{"DraughtsGame1Move35.JPG", "4,5,11,16,17", "K2,7,24,28,29,31,32"},
	{"DraughtsGame1Move36.JPG", "4,5,11,16,17", "K2,K3,24,28,29,31,32"},
	{"DraughtsGame1Move37.JPG", "4,5,15,16,17", "K2,K3,24,28,29,31,32"},
	{"DraughtsGame1Move38.JPG", "4,5,15,16,17", "K2,K3,20,28,29,31,32"},
	{"DraughtsGame1Move39.JPG", "4,5,15,17,19", "K2,K3,20,28,29,31,32"},
	{"DraughtsGame1Move40.JPG", "4,5,15,17,19", "K2,K7,20,28,29,31,32"},
	{"DraughtsGame1Move41.JPG", "4,5,17,18,19", "K2,K7,20,28,29,31,32"},
	{"DraughtsGame1Move42.JPG", "4,5,17,18,19", "K2,K10,20,28,29,31,32"},
	{"DraughtsGame1Move43.JPG", "4,5,17,19,22", "K2,K10,20,28,29,31,32"},
	{"DraughtsGame1Move44.JPG", "4,5,17,19,22", "K2,K14,20,28,29,31,32"},
	{"DraughtsGame1Move45.JPG", "4,5,19,21,22", "K2,K14,20,28,29,31,32"},
	{"DraughtsGame1Move46.JPG", "4,5,19,21,22", "K2,K17,20,28,29,31,32"},
	{"DraughtsGame1Move47.JPG", "4,5,19,22,25", "K2,K17,20,28,29,31,32"},
	{"DraughtsGame1Move48.JPG", "4,5,19,25", "K2,20,K26,28,29,31,32"},
	{"DraughtsGame1Move49.JPG", "4,5,19,K30", "K2,20,K26,28,29,31,32"},
	{"DraughtsGame1Move50.JPG", "4,5,19,K30", "K2,20,K26,27,28,29,32"},
	{"DraughtsGame1Move51.JPG", "4,5,19,K23", "K2,20,27,28,29,32"},
	{"DraughtsGame1Move52.JPG", "4,5,19", "K2,18,20,28,29,32"},
	{"DraughtsGame1Move53.JPG", "4,5,23", "K2,18,20,28,29,32"},
	{"DraughtsGame1Move54.JPG", "4,5,23", "K2,15,20,28,29,32"},
	{"DraughtsGame1Move55.JPG", "4,5,26", "K2,15,20,28,29,32"},
	{"DraughtsGame1Move56.JPG", "4,5,26", "K2,11,20,28,29,32"},
	{"DraughtsGame1Move57.JPG", "4,5,K31", "K2,11,20,28,29,32"},
	{"DraughtsGame1Move58.JPG", "4,5,K31", "K2,11,20,27,28,29"},
	{"DraughtsGame1Move59.JPG", "4,5,K24", "K2,11,20,28,29"},
	{"DraughtsGame1Move60.JPG", "4,5", "K2,11,19,20,29"},
	{"DraughtsGame1Move61.JPG", "4,9", "K2,11,19,20,29"},
	{"DraughtsGame1Move62.JPG", "4,9", "K2,11,19,20,25"},
	{"DraughtsGame1Move63.JPG", "4,14", "K2,11,19,20,25"},
	{"DraughtsGame1Move64.JPG", "4", "K2,11,15,19,20"},
	{"DraughtsGame1Move65.JPG", "8", "K2,11,15,19,20,29"},
	{"DraughtsGame1Move66.JPG", "", "K2,K4,15,19,20,29"}
};

// Data provided:  Approx. frame number, From square number, To square number
// Note that the first move is a White move (and then the moves alternate Black, White, Black, White...)
// This data corresponds to the video:  DraughtsGame1.avi
// Note that this information can ONLY be used to evaluate performance.  It must not be used during processing of the video.
const int GROUND_TRUTH_FOR_DRAUGHTSGAME1_VIDEO_MOVES[][3] = {
{ 17, 9, 13 },
{ 37, 24, 20 },
{ 50, 6, 9 },
{ 65, 22, 17 },
{ 85, 13, 22 },
{ 108, 26, 17 },
{ 123, 9, 13 },
{ 161, 30, 26 },
{ 180, 13, 22 },
{ 201, 25, 18 },
{ 226, 12, 16 },
{ 244, 18, 14 },
{ 266, 10, 17 },
{ 285, 21, 14 },
{ 308, 2, 6 },
{ 326, 26, 22 },
{ 343, 6, 9 },
{ 362, 22, 18 },
{ 393, 11, 15 },
{ 433, 18, 2 },
{ 453, 9, 18 },
{ 472, 23, 14 },
{ 506, 3, 7 },
{ 530, 20, 11 },
{ 546, 7, 16 },
{ 582, 2, 7 },
{ 617, 8, 11 },
{ 641, 27, 24 },
{ 673, 1, 6 },
{ 697, 7, 2 },
{ 714, 6, 9 },
{ 728, 14, 10 },
{ 748, 9, 14 },
{ 767, 10, 7 },
{ 781, 14, 17 },
{ 801, 7, 3 },
{ 814, 11, 15 },
{ 859, 24, 20 },
{ 870, 16, 19 },
{ 891, 3, 7 },
{ 923, 15, 18 },
{ 936, 7, 10 },
{ 955, 18, 22 },
{ 995, 10, 14 },
{ 1014, 17, 21 },
{ 1034, 14, 17 },
{ 1058, 21, 25 },
{ 1075, 17, 26 },
{ 1104, 25, 30 },
{ 1129, 31, 27 },
{ 1147, 30, 23 },
{ 1166, 27, 18 },
{ 1182, 19, 23 },
{ 1201, 18, 15 },
{ 1213, 23, 26 },
{ 1243, 15, 11 },
{ 1266, 26, 31 },
{ 1280, 32, 27 },
{ 1298, 31, 24 },
{ 1324, 28, 19 },
{ 1337, 5, 9 },
{ 1358, 29, 25 },
{ 1387, 9, 14 },
{ 1450, 25, 15 },
{ 1465, 4, 8 },
{ 1490, 11, 4 }
};


#define EMPTY_SQUARE 0
#define WHITE_MAN_ON_SQUARE 1
#define BLACK_MAN_ON_SQUARE 3
#define WHITE_KING_ON_SQUARE 2
#define BLACK_KING_ON_SQUARE 4
#define NUMBER_OF_SQUARES_ON_EACH_SIDE 8
#define NUMBER_OF_SQUARES (NUMBER_OF_SQUARES_ON_EACH_SIDE*NUMBER_OF_SQUARES_ON_EACH_SIDE/2)

// histogram classes taken from "Histograms.cpp" for histogram back-projection

class Histogram
{
protected:
	Mat mImage;
	int mNumberChannels;
	int* mChannelNumbers;
	int* mNumberBins;
	float mChannelRange[2];
public:
	Histogram(Mat image, int number_of_bins)
	{
		mImage = image;
		mNumberChannels = mImage.channels();
		mChannelNumbers = new int[mNumberChannels];
		mNumberBins = new int[mNumberChannels];
		mChannelRange[0] = 0.0;
		mChannelRange[1] = 255.0;
		for (int count = 0; count < mNumberChannels; count++)
		{
			mChannelNumbers[count] = count;
			mNumberBins[count] = number_of_bins;
		}
		//ComputeHistogram();
	}
	virtual void ComputeHistogram() = 0;
	virtual void NormaliseHistogram() = 0;
	static void Draw1DHistogram(MatND histograms[], int number_of_histograms, Mat& display_image)
	{
		int number_of_bins = histograms[0].size[0];
		double max_value = 0, min_value = 0;
		double channel_max_value = 0, channel_min_value = 0;
		for (int channel = 0; (channel < number_of_histograms); channel++)
		{
			minMaxLoc(histograms[channel], &channel_min_value, &channel_max_value, 0, 0);
			max_value = ((max_value > channel_max_value) && (channel > 0)) ? max_value : channel_max_value;
			min_value = ((min_value < channel_min_value) && (channel > 0)) ? min_value : channel_min_value;
		}
		float scaling_factor = ((float)256.0) / ((float)number_of_bins);

		Mat histogram_image((int)(((float)number_of_bins) * scaling_factor) + 1, (int)(((float)number_of_bins) * scaling_factor) + 1, CV_8UC3, Scalar(255, 255, 255));
		display_image = histogram_image;
		line(histogram_image, Point(0, 0), Point(0, histogram_image.rows - 1), Scalar(0, 0, 0));
		line(histogram_image, Point(histogram_image.cols - 1, histogram_image.rows - 1), Point(0, histogram_image.rows - 1), Scalar(0, 0, 0));
		int highest_point = static_cast<int>(0.9 * ((float)number_of_bins) * scaling_factor);
		for (int channel = 0; (channel < number_of_histograms); channel++)
		{
			int last_height;
			for (int h = 0; h < number_of_bins; h++)
			{
				float value = histograms[channel].at<float>(h);
				int height = static_cast<int>(value * highest_point / max_value);
				int where = (int)(((float)h) * scaling_factor);
				if (h > 0)
					line(histogram_image, Point((int)(((float)(h - 1)) * scaling_factor) + 1, (int)(((float)number_of_bins) * scaling_factor) - last_height),
						Point((int)(((float)h) * scaling_factor) + 1, (int)(((float)number_of_bins) * scaling_factor) - height),
						Scalar(channel == 0 ? 255 : 0, channel == 1 ? 255 : 0, channel == 2 ? 255 : 0));
				last_height = height;
			}
		}
	}
};
class OneDHistogram : public Histogram
{
private:
	MatND mHistogram[3];
public:
	OneDHistogram(Mat image, int number_of_bins) :
		Histogram(image, number_of_bins)
	{
		ComputeHistogram();
	}
	void ComputeHistogram()
	{
		vector<Mat> image_planes(mNumberChannels);
		split(mImage, image_planes);
		for (int channel = 0; (channel < mNumberChannels); channel++)
		{
			const float* channel_ranges = mChannelRange;
			int* mch = { 0 };
			calcHist(&(image_planes[channel]), 1, mChannelNumbers, Mat(), mHistogram[channel], 1, mNumberBins, &channel_ranges);
		}
	}
	void SmoothHistogram()
	{
		for (int channel = 0; (channel < mNumberChannels); channel++)
		{
			MatND temp_histogram = mHistogram[channel].clone();
			for (int i = 1; i < mHistogram[channel].rows - 1; ++i)
			{
				mHistogram[channel].at<float>(i) = (temp_histogram.at<float>(i - 1) + temp_histogram.at<float>(i) + temp_histogram.at<float>(i + 1)) / 3;
			}
		}
	}
	MatND getHistogram(int index)
	{
		return mHistogram[index];
	}
	void NormaliseHistogram()
	{
		for (int channel = 0; (channel < mNumberChannels); channel++)
		{
			normalize(mHistogram[channel], mHistogram[channel], 1.0);
		}
	}
	Mat BackProject(Mat& image)
	{
		Mat& result = image.clone();
		if (mNumberChannels == 1)
		{
			const float* channel_ranges[] = { mChannelRange, mChannelRange, mChannelRange };
			for (int channel = 0; (channel < mNumberChannels); channel++)
			{
				calcBackProject(&image, 1, mChannelNumbers, *mHistogram, result, channel_ranges, 255.0);
			}
		}
		else
		{
		}
		return result;
	}
	void Draw(Mat& display_image)
	{
		Draw1DHistogram(mHistogram, mNumberChannels, display_image);
	}
};

class ColourHistogram : public Histogram
{
private:
	MatND mHistogram;
public:
	ColourHistogram(Mat all_images[], int number_of_images, int number_of_bins) :
		Histogram(all_images[0], number_of_bins)
	{
		const float* channel_ranges[] = { mChannelRange, mChannelRange, mChannelRange };
		for (int index = 0; index < number_of_images; index++)
			calcHist(&mImage, 1, mChannelNumbers, Mat(), mHistogram, mNumberChannels, mNumberBins, channel_ranges, true, true);
	}
	ColourHistogram(Mat image, int number_of_bins) :
		Histogram(image, number_of_bins)
	{
		ComputeHistogram();
	}
	void ComputeHistogram()
	{
		const float* channel_ranges[] = { mChannelRange, mChannelRange, mChannelRange };
		calcHist(&mImage, 1, mChannelNumbers, Mat(), mHistogram, mNumberChannels, mNumberBins, channel_ranges);
	}
	void NormaliseHistogram()
	{
		normalize(mHistogram, mHistogram, 1.0);
	}
	Mat BackProject(Mat& image)
	{
		Mat& result = image.clone();
		const float* channel_ranges[] = { mChannelRange, mChannelRange, mChannelRange };
		calcBackProject(&image, 1, mChannelNumbers, mHistogram, result, channel_ranges, 255.0);
		return result;
	}
	MatND getHistogram()
	{
		return mHistogram;
	}
};


class DraughtsBoard
{
private:
	int mBoardGroundTruth[NUMBER_OF_SQUARES];
	Mat mOriginalImage;
	Mat white_piece_sample_image, black_piece_sample_image, white_square_sample_image, black_square_sample_image;
	Mat white, black, empty;

	void loadGroundTruth(string pieces, int man_type, int king_type);
	Mat backProj(Mat& original_image, Mat& sample_image);
	int return_class(int lum1, int lum2, int lum3, int lum4);
	int max(int i, int j);
public:
	DraughtsBoard(string filename, string white_pieces_ground_truth, string black_pieces_ground_truth);
};

DraughtsBoard::DraughtsBoard(string filename, string white_pieces_ground_truth, string black_pieces_ground_truth)
{
	for (int square_count = 1; square_count <= NUMBER_OF_SQUARES; square_count++)
	{
		mBoardGroundTruth[square_count - 1] = EMPTY_SQUARE;
	}
	loadGroundTruth(white_pieces_ground_truth, WHITE_MAN_ON_SQUARE, WHITE_KING_ON_SQUARE);
	loadGroundTruth(black_pieces_ground_truth, BLACK_MAN_ON_SQUARE, BLACK_KING_ON_SQUARE);

	string full_filename = "Media/" + filename;
	string white_piece_image_name = "Media/DraughtsGame1WhitePieces.png";
	string black_piece_image_name = "Media/DraughtsGame1BlackPieces.png";
	string white_square_image_name = "Media/DraughtsGame1WhiteSquares.png";
	string black_square_image_name = "Media/DraughtsGame1BlackSquares.png";

	string white_name = "Media/white_piece_sample.jpg";
	string black_name = "Media/black_piece_sample.jpg";
	string empty_name = "Media/empty_square_sample.jpg";

	mOriginalImage = imread(full_filename, -1);

	white_piece_sample_image = imread(white_piece_image_name);
	black_piece_sample_image = imread(black_piece_image_name);
	white_square_sample_image = imread(white_square_image_name);
	black_square_sample_image = imread(black_square_image_name);

	white = imread(white_name);
	black = imread(black_name);
	empty = imread(empty_name);

	if (mOriginalImage.empty() || white_piece_sample_image.empty() || black_piece_sample_image.empty() || white_square_sample_image.empty() || black_square_sample_image.empty() || white.empty() || black.empty() || empty.empty())
		//cout << "Cannot open image file: " << full_filename << endl;
		cout << "Cannot open some of the image files" << endl;
	else {
		// display the original image
		imshow(full_filename, mOriginalImage);

		// *********** PART 1 ***********

// back-project the four sample images onto the original image
// this generates four separate probability models for the four object types
		Mat white_piece_back_proj_probs = backProj(mOriginalImage, white_piece_sample_image);
		Mat black_piece_back_proj_probs = backProj(mOriginalImage, black_piece_sample_image);
		Mat white_square_back_proj_probs = backProj(mOriginalImage, white_square_sample_image);
		Mat black_square_back_proj_probs = backProj(mOriginalImage, black_square_sample_image);

		Mat intermediate1, intermediate2, intermediate3, intermediate4;
		Mat white_piece, black_piece, white_square, black_square;
		cvtColor(white_piece_back_proj_probs, intermediate1, COLOR_GRAY2BGR);
		cvtColor(intermediate1, white_piece, COLOR_BGR2HLS);
		cvtColor(black_piece_back_proj_probs, intermediate2, COLOR_GRAY2BGR);
		cvtColor(intermediate2, black_piece, COLOR_BGR2HLS);
		cvtColor(white_square_back_proj_probs, intermediate3, COLOR_GRAY2BGR);
		cvtColor(intermediate3, white_square, COLOR_BGR2HLS);
		cvtColor(black_square_back_proj_probs, intermediate4, COLOR_GRAY2BGR);
		cvtColor(intermediate4, black_square, COLOR_BGR2HLS);

		Mat part1 = mOriginalImage.clone();

		for (int row = 0; row < part1.rows; row++) {
			for (int col = 0; col < part1.cols; col++) {
				Vec3b pixel1 = white_piece.at<Vec3b>(row, col);
				Vec3b pixel2 = black_piece.at<Vec3b>(row, col);
				Vec3b pixel3 = white_square.at<Vec3b>(row, col);
				Vec3b pixel4 = black_square.at<Vec3b>(row, col);

				int pixel_class = return_class(pixel1[1], pixel2[1], pixel3[1], pixel4[1]); // returns class to which the pixel belongs

				switch (pixel_class) {
				case 1:
					// white pieces = white
					part1.at<Vec3b>(row, col)[0] = 255;
					part1.at<Vec3b>(row, col)[1] = 255;
					part1.at<Vec3b>(row, col)[2] = 255;
					break;

				case 2:
					// black pieces = black
					part1.at<Vec3b>(row, col)[0] = 0;
					part1.at<Vec3b>(row, col)[1] = 0;
					part1.at<Vec3b>(row, col)[2] = 0;
					break;

				case 3:
					// white squares = red
					part1.at<Vec3b>(row, col)[0] = 0;
					part1.at<Vec3b>(row, col)[1] = 0;
					part1.at<Vec3b>(row, col)[2] = 255;
					break;
				case 4:
					// black squares = green
					part1.at<Vec3b>(row, col)[0] = 0;
					part1.at<Vec3b>(row, col)[1] = 255;
					part1.at<Vec3b>(row, col)[2] = 0;
					break;
				default:
					// background/non-object pixels = blue
					part1.at<Vec3b>(row, col)[0] = 255;
					part1.at<Vec3b>(row, col)[1] = 0;
					part1.at<Vec3b>(row, col)[2] = 0;
					break;
				}
			}
		}


		imshow("Part 1", part1);

		// ************ PART 2 ************

				// Perform a perspective transformation on the board
		Mat perspective_matrix(3, 3, CV_32FC1), perspective_warped_image, perspective_warped_part1;
		perspective_warped_image = Mat::zeros(400, 400, mOriginalImage.type());
		perspective_warped_part1 = Mat::zeros(400, 400, part1.type());

		Point2f source_points[4], destination_points[4];
		source_points[0] = Point2f(114.0, 17.0);
		source_points[1] = Point2f(53.0, 245.0);
		source_points[2] = Point2f(355.0, 20.0);
		source_points[3] = Point2f(433.0, 241.0);

		destination_points[0] = Point2f(1.0, 1.0);
		destination_points[1] = Point2f(1.0, 399.0);
		destination_points[2] = Point2f(399.0, 1.0);
		destination_points[3] = Point2f(399.0, 399.0);

		perspective_matrix = getPerspectiveTransform(source_points, destination_points);
		warpPerspective(mOriginalImage, perspective_warped_image, perspective_matrix, perspective_warped_image.size());
		warpPerspective(part1, perspective_warped_part1, perspective_matrix, perspective_warped_image.size());


		Mat perspective_transform_output = JoinImagesHorizontally(perspective_warped_image, "Original Image", perspective_warped_part1, "Classified Image");
		//imshow("Perspective Transformation", perspective_transform_output);

		// Divide the image into 64 equally sized blocks: in other words, 64 50x50 blocks (assuming the image is 400x400)
		// Each block corresponds to one of the squares on the board
		// Only need to process 32 of these: the 32 squares that are black

		int block_width = perspective_warped_image.cols / 8;
		int block_height = perspective_warped_image.rows / 8;

		vector<Mat> result_blocks;

		for (int i = 0; i < 8; i++) {
			for (int j = 0; j < 8; j++) {
				// for only odd-even and even-odd values (i.e. for only black squares)
				if ((i % 2 != 0) && (j % 2 == 0) || (j % 2 != 0) && (i % 2 == 0)) {
					Rect current_square(block_width * i, block_height * j, block_width, block_height);
					result_blocks.push_back(perspective_warped_image(current_square));
				}
			}
		}

		// The 32 squares are now stored in a vector in an order consistent with Portable Draughts Notation
		// Now go through each square one-by-one and identify whether it contains a white piece, black piece or is empty

		// I AM GOING TO RESORT TO USE OF HISTOGRAM COMPARISON FOR THE TIME BEING
		// I WILL SET A THRESHOLD FOR A PIECE TO BE COUNTED TO MITIGATE THE BOARD-GLARE ISSUE
		// THIS WILL ENABLE ME TO MOVE ON TO LOADING ALL 67 FRAMES AND COMPLETING A PROVISIONAL DRAFT FOR PART 2
		// WHICH MEANS I CAN MOVE ON TO PART 3
		// IF STUDY OF EDGES GIVES AN IDEA FOR A BETTER SOLUTION, I WILL RETURN AND IMPROVE THE CURRENT ITERATION

		// generate three histograms to compare each square against 
		// the samples used here were created using frame 21: not an ideal approach but gives the best results at the moment

		Mat hls_empty_square, hls_white_piece, hls_black_piece, hls_unclassified_square;

		cvtColor(empty, hls_empty_square, COLOR_BGR2HLS);
		ColourHistogram empty_square_histogram(hls_empty_square, 4);
		//imshow("Empty Sample", empty);
		cvtColor(white, hls_white_piece, COLOR_BGR2HLS);
		ColourHistogram white_piece_histogram(hls_white_piece, 4);
		//imshow("White Sample", white);
		cvtColor(black, hls_black_piece, COLOR_BGR2HLS);
		ColourHistogram black_piece_histogram(hls_black_piece, 4);
		//imshow("Black Sample", black);

		double empty_scores[32];
		double white_scores[32];
		double black_scores[32];
		int status[32];
		string white_pieces = "";
		string black_pieces = "";

		for (int i = 0; i < result_blocks.size(); i++) {
			cvtColor(result_blocks[i], hls_unclassified_square, COLOR_BGR2HLS);
			ColourHistogram reference_histogram(hls_unclassified_square, 4);
			reference_histogram.NormaliseHistogram();
			empty_scores[i] = compareHist(reference_histogram.getHistogram(), empty_square_histogram.getHistogram(), cv::HISTCMP_CORREL);
			white_scores[i] = compareHist(reference_histogram.getHistogram(), white_piece_histogram.getHistogram(), cv::HISTCMP_CORREL);
			black_scores[i] = compareHist(reference_histogram.getHistogram(), black_piece_histogram.getHistogram(), cv::HISTCMP_CORREL);

			if (abs(empty_scores[i] - white_scores[i]) > 0.2 || abs(empty_scores[i] - black_scores[i]) > 0.2 || abs(black_scores[i] - white_scores[i]) > 0.2) {
				if (empty_scores[i] > white_scores[i]) {
					if (empty_scores[i] > black_scores[i]) {
						status[i] = 0; // represent empty square with 0
					}
					else {
						status[i] = 2; // represent black piece with 2
					}
				}
				else if (white_scores[i] > black_scores[i]) {
					status[i] = 1; // represent white piece with 1
				}
				else {
					status[i] = 2;
				}
			}
			else
				status[i] = 0; // default to an empty square if not confident enough of a piece being present 

			// NOTE: THE ABOVE METHOD IS STUPID AND SHOULD BE IMPROVED EVENTUALLY

			if (status[i] == 1) {
				if (white_pieces == "") {
					white_pieces = white_pieces + to_string(i + 1);
				}
				else white_pieces = white_pieces + "," + to_string(i + 1);
			}
			else if (status[i] == 2) {
				if (black_pieces == "") {
					black_pieces = black_pieces + to_string(i + 1);
				}
				else black_pieces = black_pieces + "," + to_string(i + 1);
			}

			//std::cout << i + 1 << endl;
			//std::cout << "Empty Score: " << empty_scores[i] << endl;
			//std::cout << "White Score: " << white_scores[i] << endl;
			//std::cout << "Black Score: " << black_scores[i] << endl;
		}

		std::cout << "\nWhite Pieces:	           " << white_pieces << endl;
		std::cout << "White Pieces Ground Truth: " << GROUND_TRUTH_FOR_BOARD_IMAGES[21][1] << endl;
		std::cout << "\nBlack Pieces:		   " << black_pieces << endl;
		std::cout << "Black Pieces Ground Truth: " << GROUND_TRUTH_FOR_BOARD_IMAGES[21][2] << endl;
	}
}



void DraughtsBoard::loadGroundTruth(string pieces, int man_type, int king_type)
{
	int index = 0;
	while (index < pieces.length())
	{
		bool is_king = false;
		if (pieces.at(index) == 'K')
		{
			is_king = true;
			index++;
		}
		int location = 0;
		while ((index < pieces.length()) && (pieces.at(index) >= '0') && (pieces.at(index) <= '9'))
		{
			location = location * 10 + (pieces.at(index) - '0');
			index++;
		}
		index++;
		if ((location > 0) && (location <= NUMBER_OF_SQUARES))
			mBoardGroundTruth[location - 1] = (is_king) ? king_type : man_type;
	}
}

// function that returns a greyscale probability image from back projection of input sample image on input original image
Mat DraughtsBoard::backProj(Mat& original_image, Mat& sample_image)
{
	Mat hls_image;
	cvtColor(sample_image, hls_image, COLOR_BGR2HLS);
	ColourHistogram histogram3D(hls_image, 8);
	histogram3D.NormaliseHistogram();
	cvtColor(original_image, hls_image, COLOR_BGR2HLS);
	Mat back_proj_probs = histogram3D.BackProject(hls_image);
	back_proj_probs = StretchImage(back_proj_probs);

	return back_proj_probs;
}

int DraughtsBoard::max(int i, int j) {
	if (i > j) {
		return i;
	}
	else return j;
}

// compares the luminance values of four probability pixels and determines which class (if any) the pixel belongs to
int DraughtsBoard::return_class(int lum1, int lum2, int lum3, int lum4) {
	
	int comp1 = max(lum1, lum2);
	int comp2 = max(lum3, lum4);
	int overall = max(comp1, comp2);

	int prob_thresh = 0; // arbitrary probability threshold selection

	if (overall > prob_thresh) {
		if (overall == lum1) {
			return 1;
		}
		else if (overall == lum2) {
			return 2;
		}
		else if (overall == lum3) {
			return 3;
		}
		else if (overall == lum4) {
			return 4;
		}
		else return 5; // error handling - should never reach this point
	}
	else return 5;

}


void MyApplication()
{	
	string video_filename("Media/DraughtsGame1.avi");
	VideoCapture video;
	video.open(video_filename);

	int pieces[32];
	string black_pieces_filename("Media/DraughtsGame1BlackPieces.jpg");
	Mat black_pieces_image = imread(black_pieces_filename, -1);
	string white_pieces_filename("Media/DraughtsGame1WhitePieces.jpg");
	Mat white_pieces_image = imread(white_pieces_filename, -1);
	string black_squares_filename("Media/DraughtsGame1BlackSquares.jpg");
	Mat black_squares_image = imread(black_squares_filename, -1);
	string white_squares_filename("Media/DraughtsGame1WhiteSquares.jpg");
	Mat white_squares_image = imread(white_squares_filename, -1);
	string background_filename("Media/DraughtsGame1EmptyBoard.JPG");
	Mat static_background_image = imread(background_filename, -1);
	if ((!video.isOpened()) || (black_pieces_image.empty()) || (white_pieces_image.empty()) || (black_squares_image.empty()) || (white_squares_image.empty())  || (static_background_image.empty()))
	{
		// Error attempting to load something.
		if (!video.isOpened())
			cout << "Cannot open video file: " << video_filename << endl;
		if (black_pieces_image.empty())
			cout << "Cannot open image file: " << black_pieces_filename << endl;
		if (white_pieces_image.empty())
			cout << "Cannot open image file: " << white_pieces_filename << endl;
		if (black_squares_image.empty())
			cout << "Cannot open image file: " << black_squares_filename << endl;
		if (white_squares_image.empty())
			cout << "Cannot open image file: " << white_squares_filename << endl;
		if (static_background_image.empty())
			cout << "Cannot open image file: " << background_filename << endl;
	}
	else
	{
		// Sample loading of image and ground truth
		int image_index = 21;
		DraughtsBoard current_board(GROUND_TRUTH_FOR_BOARD_IMAGES[image_index][0], GROUND_TRUTH_FOR_BOARD_IMAGES[image_index][1], GROUND_TRUTH_FOR_BOARD_IMAGES[image_index][2]);

		// Process video frame by frame
		Mat current_frame;
		video.set(cv::CAP_PROP_POS_FRAMES, 1);
		video >> current_frame;
		double last_time = static_cast<double>(getTickCount());
		double frame_rate = video.get(cv::CAP_PROP_FPS);
		double time_between_frames = 1000.0 / frame_rate;
		while (!current_frame.empty())
		{
			double current_time = static_cast<double>(getTickCount());
			double duration = (current_time - last_time) / getTickFrequency() / 1000.0;
			int delay = (time_between_frames > duration) ? ((int)(time_between_frames - duration)) : 1;
			last_time = current_time;
			imshow("Draughts video", current_frame);
			video >> current_frame;
			char c = cv::waitKey(delay);  // If you replace delay with 1 it will play the video as quickly as possible.
		}
		cv::destroyAllWindows();
	}
}
