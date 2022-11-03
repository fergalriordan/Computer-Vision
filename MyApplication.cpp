#include "Utilities.h"
#include <iostream>
#include <iomanip>
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
	{"DraughtsGame1Move64.JPG", "4,14", "K2,11,19,20,22"},
	{"DraughtsGame1Move65.JPG", "4,18", "K2,11,19,20,22"},
	{"DraughtsGame1Move66.JPG", "4", "K2,11,15,19,20"},
	{"DraughtsGame1Move67.JPG", "8", "K2,11,15,19,20"},
	{"DraughtsGame1Move68.JPG", "", "K2,K4,15,19,20"} 
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
{ 1450, 25, 22 },
{ 1465, 14, 18 },
{ 1490, 22, 15 },
{ 1515, 4, 8 },
{ 1540, 11, 4 }
};


#define EMPTY_SQUARE 0
#define WHITE_MAN_ON_SQUARE 1
#define BLACK_MAN_ON_SQUARE 3
#define WHITE_KING_ON_SQUARE 2
#define BLACK_KING_ON_SQUARE 4
#define NUMBER_OF_SQUARES_ON_EACH_SIDE 8
#define NUMBER_OF_SQUARES (NUMBER_OF_SQUARES_ON_EACH_SIDE*NUMBER_OF_SQUARES_ON_EACH_SIDE/2)

class DraughtsBoard
{
private:
	int mBoardGroundTruth[NUMBER_OF_SQUARES];
	Mat mOriginalImage;

	void loadGroundTruth(string pieces, int man_type, int king_type);

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

	mOriginalImage = imread(full_filename, -1);

	if (mOriginalImage.empty())
		cout << "Cannot open image file: " << full_filename << endl;
	else {
		// display the original image
		//imshow(full_filename, mOriginalImage);
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


// **************************** PART 1 FUNCTIONS *******************************************

// function that returns a greyscale probability image from back projection of input sample image on input original image
Mat backProj(Mat& original_image, Mat& sample_image)
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

// compares the luminance values of four probability pixels and determines which class (if any) the pixel belongs to
int return_class(int lum1, int lum2, int lum3, int lum4) {
	int comp1 = 0;
	int comp2 = 0;
	int overall = 0;

	// store the highest probability in the "overall" variable
	if (lum1 > lum2)
		comp1 = lum1;
	else
		comp1 = lum2;
	if (lum3 > lum4)
		comp2 = lum3;
	else
		comp2 = lum4;
	if (comp1 > comp2)
		overall = comp1;
	else
		overall = comp2;

	int prob_thresh = 5; // arbitrary probability threshold selection: turns out it needs to be very low, otherwise many board pixels are incorrectly classified as background

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

void Part1(string filename, Mat white_pieces_image, Mat black_pieces_image, Mat white_squares_image, Mat black_squares_image, Mat& output) {

	string full_filename = "Media/" + filename;
	Mat original_image = imread(full_filename, -1);

	if (original_image.empty())
		cout << "Cannot open image file: " << full_filename << endl;

	// back-project the four sample images onto the original image to generate four separate probability models for the four object types
	Mat white_piece_back_proj_probs = backProj(original_image, white_pieces_image);
	Mat black_piece_back_proj_probs = backProj(original_image, black_pieces_image);
	Mat white_square_back_proj_probs = backProj(original_image, white_squares_image);
	Mat black_square_back_proj_probs = backProj(original_image, black_squares_image);

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

	output = original_image.clone();

	for (int row = 0; row < output.rows; row++) {
		for (int col = 0; col < output.cols; col++) {
			Vec3b pixel1 = white_piece.at<Vec3b>(row, col);
			Vec3b pixel2 = black_piece.at<Vec3b>(row, col);
			Vec3b pixel3 = white_square.at<Vec3b>(row, col);
			Vec3b pixel4 = black_square.at<Vec3b>(row, col);

			int pixel_class = return_class(pixel1[1], pixel2[1], pixel3[1], pixel4[1]); // returns class to which the pixel belongs

			switch (pixel_class) {
			case 1:
				// white pieces = white
				output.at<Vec3b>(row, col)[0] = 255;
				output.at<Vec3b>(row, col)[1] = 255;
				output.at<Vec3b>(row, col)[2] = 255;
				break;

			case 2:
				// black pieces = black
				output.at<Vec3b>(row, col)[0] = 0;
				output.at<Vec3b>(row, col)[1] = 0;
				output.at<Vec3b>(row, col)[2] = 0;
				break;

			case 3:
				// white squares = red
				output.at<Vec3b>(row, col)[0] = 0;
				output.at<Vec3b>(row, col)[1] = 0;
				output.at<Vec3b>(row, col)[2] = 255;
				break;
			case 4:
				// black squares = green
				output.at<Vec3b>(row, col)[0] = 0;
				output.at<Vec3b>(row, col)[1] = 255;
				output.at<Vec3b>(row, col)[2] = 0;
				break;
			default:
				// background/non-object pixels = blue
				output.at<Vec3b>(row, col)[0] = 255;
				output.at<Vec3b>(row, col)[1] = 0;
				output.at<Vec3b>(row, col)[2] = 0;
				break;
			}
		}
	}
}

// ************************** PART 2 FUNCTIONS **********************************************

// takes a filename for a board image and three samples for histogram comparison as inputs
// outputs strings representing the positions of white and black pieces
void Part2(string filename, Mat white_pieces_image, Mat black_pieces_image, Mat black_squares_image, string& whites, string& blacks) {

	string full_filename = "Media/" + filename;
	Mat original_image = imread(full_filename, -1);

	if (original_image.empty())
		cout << "Cannot open image file: " << full_filename << endl;

	Mat perspective_matrix(3, 3, CV_32FC1), perspective_warped_image;
	perspective_warped_image = Mat::zeros(400, 400, original_image.type());

	Point2f source_points[4], destination_points[4];
	source_points[0] = Point2f(114.0, 17.0);
	source_points[1] = Point2f(53.0, 245.0);
	source_points[2] = Point2f(355.0, 20.0);
	source_points[3] = Point2f(433.0, 241.0);

	destination_points[0] = Point2f(0, 0);
	destination_points[1] = Point2f(0, 400.0);
	destination_points[2] = Point2f(400.0, 0);
	destination_points[3] = Point2f(400.0, 400.0);

	// perform a perspective transformation using the four board corner coordinates
	perspective_matrix = getPerspectiveTransform(source_points, destination_points);
	warpPerspective(original_image, perspective_warped_image, perspective_matrix, perspective_warped_image.size());
	
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
				// don't even take the full square: use a smaller inner section where a piece is most likely to be found
				Rect current_square((block_width * i + 12.5), (block_height * j + 12.5), (block_width - 25), (block_height - 25));
				result_blocks.push_back(perspective_warped_image(current_square));
			}
		}
	}

	// The 32 squares are now stored in a vector in an order consistent with Portable Draughts Notation
	// Now go through each square one-by-one and identify whether it contains a white piece, black piece or is empty

	// generate three histograms to compare each square against 
	Mat hls_empty_square, hls_white_piece, hls_black_piece, hls_unclassified_square;
	cvtColor(black_squares_image, hls_empty_square, COLOR_BGR2HLS);
	ColourHistogram empty_square_histogram(hls_empty_square, 4);
	cvtColor(white_pieces_image, hls_white_piece, COLOR_BGR2HLS);
	ColourHistogram white_piece_histogram(hls_white_piece, 4);
	cvtColor(black_pieces_image, hls_black_piece, COLOR_BGR2HLS);
	ColourHistogram black_piece_histogram(hls_black_piece, 4);

	double empty_score = 0;
	double white_score = 0;
	double black_score = 0;

	whites = "";
	blacks = ""; // ensure that the position strings are empty

	for (int i = 0; i < result_blocks.size(); i++) {
		cvtColor(result_blocks[i], hls_unclassified_square, COLOR_BGR2HLS);
		ColourHistogram reference_histogram(hls_unclassified_square, 4);
		reference_histogram.NormaliseHistogram();
		empty_score = compareHist(reference_histogram.getHistogram(), empty_square_histogram.getHistogram(), cv::HISTCMP_CORREL);
		white_score = compareHist(reference_histogram.getHistogram(), white_piece_histogram.getHistogram(), cv::HISTCMP_CORREL);
		black_score = compareHist(reference_histogram.getHistogram(), black_piece_histogram.getHistogram(), cv::HISTCMP_CORREL);

		if (abs(empty_score - white_score) > 0.2 || abs(empty_score - black_score) > 0.2 || abs(black_score - white_score) > 0.2) {			
			if (white_score > empty_score) {
				if (white_score > black_score) {
					if (whites == "") {
						whites = whites + to_string(i + 1);
					}
					else whites = whites + "," + to_string(i + 1);
				}
				else if (black_score > white_score) {
					if (blacks == "") {
						blacks = blacks + to_string(i + 1);
					}
					else blacks = blacks + "," + to_string(i + 1);
				}
			}
			else if (black_score > empty_score) {
				if (blacks == "") {
					blacks = blacks + to_string(i + 1);
				}
				else blacks = blacks + "," + to_string(i + 1);
			}
		}
	}
}

// function to convert a board representation to a length-32 integer array where empty = 0, white = 1, black = 2 
// this makes the confusion matrix calculations easier (though the conversion to an array is so convoluted that it was probably not a good idea)
void board_representation_from_strings_to_array(string whites, string blacks, string (&positions)[32]) {
	vector<string> white_positions;
	vector<string> white_king_positions;
	vector<string> black_positions;
	vector<string> black_king_positions;

	// need to store the king prefixes and piece positions separately as I am using the positions as integers to index into a length-32 board array

	int whites_buffer[12]; // buffers that store the string-to-integer-converted piece positions
	int blacks_buffer[12]; // once the pieces are positioned in the 32-string array, I will add the king prefixes back in the correct places

	for (int i = 0; i < 12; i++) {
		whites_buffer[i] = -1;
		blacks_buffer[i] = -1; // initialise the two buffers to a value that doesn't represent any of the three possible states
	}

	stringstream sw(whites);
	stringstream sb(blacks);

	while (sw.good()) {
		string next_white;
		getline(sw, next_white, ',');
		string prefix = next_white.substr(0, 1);
		if (prefix != "K" && !next_white.empty()) { 
			white_positions.push_back(next_white); // if it is just a white man, simply push to the white position vector
		}
		else if (!next_white.empty()){ // if a king is detected, subtract the king prefix and then push to the white position vector
			string white_king_position = next_white.substr(1); // record the remainder of the string that follows the K
			white_positions.push_back(white_king_position);
			white_king_positions.push_back(white_king_position);
		}

	}

	while (sb.good()) {
		string next_black;
		getline(sb, next_black, ',');
		string prefix = next_black.substr(0, 1);
		if (prefix != "K" && !next_black.empty()) {
			black_positions.push_back(next_black);
		}
		else if (!next_black.empty()) {
			string black_king_position = next_black.substr(1); // record the position of the king
			black_positions.push_back(black_king_position); // and store it
			black_king_positions.push_back(black_king_position);
		}
	}

	for (int i = 0; i < white_positions.size(); i++) {
		whites_buffer[i] = stoi(white_positions[i]) - 1;
	}

	for (int i = 0; i < black_positions.size(); i++) {
		blacks_buffer[i] = stoi(black_positions[i]) - 1;
	}

	for (int i = 0; i < 32; i++) {
		for (int j = 0; j < 12; j++) {
			if (i == whites_buffer[j]) {
				positions[i] = "1";
			}
			else if (i == blacks_buffer[j]) {
				positions[i] = "2";
			}
			
		}
	}
}

struct ConfusionMatrix
{
	int pred_empt_truth_empt, pred_empt_truth_white, pred_empt_truth_black;
	int pred_white_truth_empt, pred_white_truth_white, pred_white_truth_black;
	int pred_black_truth_empt, pred_black_truth_white, pred_black_truth_black;
};

void display_matrix(ConfusionMatrix conf) {
	const char space = ' ';
	const int nameWidth = 15;
	const int numWidth = 15;

	cout << endl;
	cout << left << setw(nameWidth) << setfill(space) << "" << left << setw(nameWidth) << setfill(space) << "" << left << setw(nameWidth) << setfill(space) << "" << left << setw(nameWidth) << setfill(space) << "Ground Truth" << left << setfill(space) << "" << endl;
	cout << left << setw(nameWidth) << setfill(space) << "" << left << setw(nameWidth) << setfill(space) << "" << left << setw(nameWidth) << setfill(space) << "No Piece" << left << setw(nameWidth) << setfill(space) << "White Piece" << left << setfill(space) << "Black Piece" << endl;
	cout << left << setw(nameWidth) << setfill(space) << "" << left << setw(nameWidth) << setfill(space) << "No Piece" << left << setw(nameWidth) << setfill(space) << conf.pred_empt_truth_empt << left << setw(nameWidth) << setfill(space) << conf.pred_empt_truth_white << left << setw(nameWidth) << setfill(space) << conf.pred_empt_truth_black << endl;
	cout << left << setw(nameWidth) << setfill(space) << "Detected" << left << setw(nameWidth) << setfill(space) << "White Piece" << left << setw(nameWidth) << setfill(space) << conf.pred_white_truth_empt << left << setw(nameWidth) << setfill(space) << conf.pred_white_truth_white << left << setw(nameWidth) << setfill(space) << conf.pred_white_truth_black << endl;
	cout << left << setw(nameWidth) << setfill(space) << "" << left << setw(nameWidth) << setfill(space) << "Black Piece" << left << setw(nameWidth) << setfill(space) << conf.pred_black_truth_empt << left << setw(nameWidth) << setfill(space) << conf.pred_black_truth_white << left << setw(nameWidth) << setfill(space) << conf.pred_black_truth_black << endl;
	cout << endl;
}

// **************************** PART 3 FUNCTIONS **********************************************

void process_move(string before, string after, int& from, int& to) {
	// check the two strings
	// if the two strings contain the same number of values but differ by one position, a valid move has been recorded
	// identify the two mismatched values
	// store them in the moves string as "pos1, pos2"

	vector<string> bef;
	vector<string> aft;
	stringstream sb(before);
	stringstream sa(after);

	while (sb.good()) {
		string next_bef;
		getline(sb, next_bef, ',');
		bef.push_back(next_bef);
	}

	while (sa.good()) {
		string next_aft;
		getline(sa, next_aft, ',');
		aft.push_back(next_aft);
	}

	int size_bef = bef.size();
	int size_aft = aft.size();

	if (size_bef != size_aft) {
		from = -1;
		to = -1; // if the same number of pieces aren't detected before and after then record that an invalid move was detected
	}
	else if (bef == aft) {
		from = 0;
		to = 0; // if the two vectors are exactly identical then record that no move (i.e. an invalid move) was detected
	}
	else {
		// if a valid move is detected
		for (int i = 0; i < size_bef; i++) {
			// search for each "before" position in the "after" position vector, one-by-one
			string search_term1 = bef[i];
			bool found1 = false;
			for (int j = 0; j < size_aft; j++) {
				if (search_term1 == aft[j])
					found1 = true;
			}
			if (!found1) {
				string prefix1 = search_term1.substr(0, 1); // check for a King prefix
				if (prefix1 == "K") {
					string position1 = search_term1.substr(1); // if a King prefix is found, discard it
					from = stoi(position1);
				}
				else
					from = stoi(search_term1); // if no King prefix is found, process as normal
			}
		}

		for (int i = 0; i < size_aft; i++) {
			// search for each "after" position in the "before" position vector, one-by-one
			string search_term2 = aft[i];
			bool found2 = false;
			for (int j = 0; j < size_bef; j++) {
				if (search_term2 == bef[j])
					found2 = true;
			}
			if (!found2) {
				string prefix2 = search_term2.substr(0, 1);
				if (prefix2 == "K") {
					string position2 = search_term2.substr(1); // if a King prefix is found, discard it
					to = stoi(position2);
				}
				else
					to = stoi(search_term2); // if no King prefix is found, process as normal
			}
		}
	}
}

// **************************** PART 4 FUNCTIONS **********************************************

// **************************** PART 5 FUNCTIONS **********************************************



int Circles(Mat result_image, vector<Vec3f> circles, Scalar passed_colour = -1.0)
{
	int count = 0;
	for (vector<cv::Vec3f>::const_iterator current_circle = circles.begin();
		(current_circle != circles.end()); current_circle++)
	{
		Scalar colour(rand() & 0xFF, rand() & 0xFF, rand() & 0xFF);
		circle(result_image, Point((int)((*current_circle)[0]), int((*current_circle)[1])), (int)((*current_circle)[2]),
			(passed_colour.val[0] == -1.0) ? colour : passed_colour,
			2);
		count++;
	}
	return count; // return the number of circles detected
}

// much of this function is identical to Part2() - it is simply an extension of the Part 2 function
// they are only kept separate for demo purposes: Part5() does all that Part 2 does and more.

Mat perspective(string filename) {
	string full_filename = "Media/" + filename;
	Mat original_image = imread(full_filename, -1);

	if (original_image.empty())
		cout << "Cannot open image file: " << full_filename << endl;

	Mat perspective_matrix(3, 3, CV_32FC1), perspective_warped_image;
	perspective_warped_image = Mat::zeros(400, 400, original_image.type());

	Point2f source_points[4], destination_points[4];
	source_points[0] = Point2f(114.0, 17.0);
	source_points[1] = Point2f(53.0, 245.0);
	source_points[2] = Point2f(355.0, 20.0);
	source_points[3] = Point2f(433.0, 241.0);

	destination_points[0] = Point2f(0, 0);
	destination_points[1] = Point2f(0, 400.0);
	destination_points[2] = Point2f(400.0, 0);
	destination_points[3] = Point2f(400.0, 400.0);

	// perform a perspective transformation using the four board corner coordinates
	perspective_matrix = getPerspectiveTransform(source_points, destination_points);
	warpPerspective(original_image, perspective_warped_image, perspective_matrix, perspective_warped_image.size());

	return perspective_warped_image;
}


void Part5(Mat perspective_warped_image, vector<Vec3f> circles, Mat white_pieces_image, Mat black_pieces_image, Mat black_squares_image, string& whites, string& blacks) {

	Mat gray_warped;
	perspective_warped_image.convertTo(perspective_warped_image, CV_8UC1);
	circles.clear();

	cvtColor(perspective_warped_image, gray_warped, COLOR_BGR2GRAY);
	medianBlur(gray_warped, gray_warped, 5);

	HoughCircles(gray_warped, circles, cv::HOUGH_GRADIENT, 1, 8, 100, 10, 15, 20);//2,20,100,20,5,30);
	Mat hough_circles_image = perspective_warped_image.clone();
	int circle_count = Circles(hough_circles_image, circles);
	imshow("Hough transformation", hough_circles_image);

	int square_circles[32];
	for (int i = 0; i < 32; i++) {
		square_circles[i] = 0;
	}
	
	// for some reason when i perform the hough circle operation on individual squares, it doesnt work properly
	// therefore my only choice is to perform it on the full image
	// and then retrospectively work out which square each of the circles resides in retrospectively
	// that is what i am doing below
	for (int i = 0; i < circles.size(); i++) {
		for (int j = 0; j < 4; j++) {
			if (circles[i][0] >= (j*100) && circles[i][0] <= (j*100 + 50)) {
				if (circles[i][1] >= 50 && circles[i][1] <= 100) {
					square_circles[8*j]++;
				}
				else if (circles[i][1] >= 150 && circles[i][1] <= 200) {
					square_circles[8*j+1]++;
				}
				else if (circles[i][1] >= 250 && circles[i][1] <= 300) {
					square_circles[8*j+2]++;
				}
				else if (circles[i][1] >= 350 && circles[i][1] <= 400) {
					square_circles[8*j+3]++;
				}
			}
		}

		for (int j = 0; j < 4; j++) {
			if (circles[i][0] >= (j * 100 + 50) && circles[i][0] <= (j * 100 + 100)) {
				if (circles[i][1] >= 0 && circles[i][1] <= 50) {
					square_circles[8 * j + 4]++;
				}
				else if (circles[i][1] >= 100 && circles[i][1] <= 150) {
					square_circles[8 * j + 5]++;
				}
				else if (circles[i][1] >= 200 && circles[i][1] <= 250) {
					square_circles[8 * j + 6]++;
				}
				else if (circles[i][1] >= 300 && circles[i][1] <= 350) {
					square_circles[8 * j + 7]++;
				}
			}
		}

	}
	
	//***************************************************************************************************************
	
	// Divide the image into 64 equally sized blocks: in other words, 64 50x50 blocks (assuming the image is 400x400)
	// Each block corresponds to one of the squares on the board
	// Only need to process 32 of these: the 32 squares that are black
	int block_width = perspective_warped_image.cols / 8;
	int block_height = perspective_warped_image.rows / 8;
	
	vector<Mat> colour_blocks, circle_blocks;
	for (int i = 0; i < 8; i++) {
		for (int j = 0; j < 8; j++) {
			// for only odd-even and even-odd values (i.e. for only black squares)
			if ((i % 2 != 0) && (j % 2 == 0) || (j % 2 != 0) && (i % 2 == 0)) {
				// don't even take the full square: use a smaller inner section where a piece is most likely to be found
				Rect inner_square((block_width * i + 12.5), (block_height * j + 12.5), (block_width - 25), (block_height - 25));
				colour_blocks.push_back(perspective_warped_image(inner_square));

				// use full square for Hough circle detection
				Rect full_square(block_width*i, block_height*j, block_width, block_height);
				circle_blocks.push_back(hough_circles_image(full_square));
			}
		}
	}
	// The 32 squares are now stored in a vector in an order consistent with Portable Draughts Notation
	// Now go through each square one-by-one and identify whether it contains a white piece, black piece or is empty
	
	// generate three histograms to compare each square against 
	Mat hls_empty_square, hls_white_piece, hls_black_piece, hls_unclassified_square, grayscale_man_vs_king;
	cvtColor(black_squares_image, hls_empty_square, COLOR_BGR2HLS);
	ColourHistogram empty_square_histogram(hls_empty_square, 4);
	cvtColor(white_pieces_image, hls_white_piece, COLOR_BGR2HLS);
	ColourHistogram white_piece_histogram(hls_white_piece, 4);
	cvtColor(black_pieces_image, hls_black_piece, COLOR_BGR2HLS);
	ColourHistogram black_piece_histogram(hls_black_piece, 4);

	double empty_score = 0;
	double white_score = 0;
	double black_score = 0;

	whites = "";
	blacks = ""; // ensure that the position strings are empty
	
	for (int i = 0; i < colour_blocks.size(); i++) {
		
		// colour classification
		cvtColor(colour_blocks[i], hls_unclassified_square, COLOR_BGR2HLS);
		ColourHistogram reference_histogram(hls_unclassified_square, 4);
		reference_histogram.NormaliseHistogram();
		empty_score = compareHist(reference_histogram.getHistogram(), empty_square_histogram.getHistogram(), cv::HISTCMP_CORREL);
		white_score = compareHist(reference_histogram.getHistogram(), white_piece_histogram.getHistogram(), cv::HISTCMP_CORREL);
		black_score = compareHist(reference_histogram.getHistogram(), black_piece_histogram.getHistogram(), cv::HISTCMP_CORREL);		

		// check if the square contains a king
		bool king = false;
		if (square_circles[i] >= 2) {
			king = true;
		}
		
		// square classification based on empty/colour and king status
		if (abs(empty_score - white_score) > 0.2 || abs(empty_score - black_score) > 0.2 || abs(black_score - white_score) > 0.2) {
			if (white_score > empty_score) {
				if (white_score > black_score) {
					if (whites == "" && king) 
						whites = whites + "K" + to_string(i + 1);
					else if (whites == "" && !king) 
						whites = whites + to_string(i + 1);
					else if (whites != "" && king) 
						whites = whites + "," + "K" + to_string(i + 1);
					else 
						whites = whites + "," + to_string(i + 1);
				}
				else if (black_score > white_score) {
					if (blacks == "" && king) 
						blacks = blacks + "K" + to_string(i + 1);
					else if (blacks == "" && !king) 
						blacks = blacks + to_string(i + 1);
					else if (blacks != "" && king) 
						blacks = blacks + "," + "K" + to_string(i + 1);
					else 
						blacks = blacks + "," + to_string(i + 1);
				}
			}
			else if (black_score > empty_score) {
				if (blacks == "" && king) 
					blacks = blacks + "K" + to_string(i + 1);
				else if (blacks == "" && !king)
					blacks = blacks + to_string(i + 1);
				else if (blacks != "" && king)
					blacks = blacks + "," + "K" + to_string(i + 1);
				else 
					blacks = blacks + "," + to_string(i + 1);
			}
		}
	}
}

// extension of the function used in part 2 so that Kings are represented
void extended_board_representation_from_strings_to_array(string whites, string blacks, string(&positions)[32]) {
	vector<string> white_positions;
	vector<string> black_positions;

	vector<bool> white_kings;
	vector<bool> black_kings;

	// need to store the king prefixes and piece positions separately as I am using the positions as integers to index into a length-32 board array

	int whites_buffer[12]; // buffers that store the string-to-integer-converted piece positions
	int blacks_buffer[12]; // once the pieces are positioned in the 32-string array, I will add the king prefixes back in the correct places

	bool white_king_status[12];
	bool black_king_status[12];

	for (int i = 0; i < 12; i++) {
		whites_buffer[i] = -1;
		blacks_buffer[i] = -1; // initialise the two buffers to a value that doesn't represent any of the three possible states
	}

	stringstream sw(whites);
	stringstream sb(blacks);

	while (sw.good()) {
		string next_white;
		getline(sw, next_white, ',');
		string prefix = next_white.substr(0, 1);
		if (prefix != "K" && !next_white.empty()) {
			white_positions.push_back(next_white); // if it is just a white man, simply push to the white position vector
			white_kings.push_back(false);
		}
		else if (!next_white.empty()) { // if a king is detected, subtract the king prefix and then push to the white position vector
			string white_king_position = next_white.substr(1); // record the remainder of the string that follows the K
			white_positions.push_back(white_king_position);
			white_kings.push_back(true);
		}

	}

	while (sb.good()) {
		string next_black;
		getline(sb, next_black, ',');
		string prefix = next_black.substr(0, 1);
		if (prefix != "K" && !next_black.empty()) {
			black_positions.push_back(next_black);
			black_kings.push_back(false);
		}
		else if (!next_black.empty()) {
			string black_king_position = next_black.substr(1); // record the position of the king
			black_positions.push_back(black_king_position); // and store it
			black_kings.push_back(true);
		}
	}

	for (int i = 0; i < white_positions.size(); i++) {
		whites_buffer[i] = stoi(white_positions[i]) - 1;
	}

	for (int i = 0; i < black_positions.size(); i++) {
		blacks_buffer[i] = stoi(black_positions[i]) - 1;
	}

	for (int i = 0; i < 32; i++) {
		for (int j = 0; j < 12; j++) {
			if (i == whites_buffer[j] && white_kings[j] == false) {
				positions[i] = "1";
			}
			else if (i == whites_buffer[j] && white_kings[j] == true) {
				positions[i] = "2";
			}
			else if (i == blacks_buffer[j] && black_kings[j] == false) {
				positions[i] = "3";
			}
			else if (i == blacks_buffer[j] && black_kings[j] == true) {
				positions[i] = "4";
			}
		}
	}
}

struct ExtendedConfusionMatrix
{
	int pred_empt_truth_empt, pred_empt_truth_white_man, pred_empt_truth_white_king, pred_empt_truth_black_man, pred_empt_truth_black_king;
	int pred_white_man_truth_empt, pred_white_man_truth_white_man, pred_white_man_truth_white_king, pred_white_man_truth_black_man, pred_white_man_truth_black_king;
	int pred_white_king_truth_empt, pred_white_king_truth_white_man, pred_white_king_truth_white_king, pred_white_king_truth_black_man, pred_white_king_truth_black_king;
	int pred_black_man_truth_empt, pred_black_man_truth_white_man, pred_black_man_truth_white_king, pred_black_man_truth_black_man, pred_black_man_truth_black_king;
	int pred_black_king_truth_empt, pred_black_king_truth_white_man, pred_black_king_truth_white_king, pred_black_king_truth_black_man, pred_black_king_truth_black_king;
};

void initialise_extended_matrix(ExtendedConfusionMatrix& ext_conf) {
	ext_conf.pred_empt_truth_empt = 0;
	ext_conf.pred_empt_truth_white_man = 0;
	ext_conf.pred_empt_truth_white_king = 0;
	ext_conf.pred_empt_truth_black_man = 0;
	ext_conf.pred_empt_truth_black_king = 0;

	ext_conf.pred_white_man_truth_empt = 0;
	ext_conf.pred_white_man_truth_white_man = 0;
	ext_conf.pred_white_man_truth_white_king = 0;
	ext_conf.pred_white_man_truth_black_man = 0;
	ext_conf.pred_white_man_truth_black_king = 0;

	ext_conf.pred_white_king_truth_empt = 0;
	ext_conf.pred_white_king_truth_white_man = 0;
	ext_conf.pred_white_king_truth_white_king = 0;
	ext_conf.pred_white_king_truth_black_man = 0;
	ext_conf.pred_white_king_truth_black_king = 0;

	ext_conf.pred_black_man_truth_empt = 0;
	ext_conf.pred_black_man_truth_white_man = 0;
	ext_conf.pred_black_man_truth_white_king = 0;
	ext_conf.pred_black_man_truth_black_man = 0;
	ext_conf.pred_black_man_truth_black_king = 0;

	ext_conf.pred_black_king_truth_empt = 0;
	ext_conf.pred_black_king_truth_white_man = 0;
	ext_conf.pred_black_king_truth_white_king = 0;
	ext_conf.pred_black_king_truth_black_man = 0;
	ext_conf.pred_black_king_truth_black_king = 0;
}

void update_extended_confusion_matrix(string prediction, string BOARD_TRUTH, ExtendedConfusionMatrix& ext_conf) {

	if (prediction == "0" && BOARD_TRUTH == "0") {
		ext_conf.pred_empt_truth_empt++;
	}
	else if (prediction == "0" && BOARD_TRUTH == "1") {
		ext_conf.pred_empt_truth_white_man++;
	}
	else if (prediction == "0" && BOARD_TRUTH == "2") {
		ext_conf.pred_empt_truth_white_king++;
	}
	else if (prediction == "0" && BOARD_TRUTH == "3") {
		ext_conf.pred_empt_truth_black_man++;
	}
	else if (prediction == "0" && BOARD_TRUTH == "4") {
		ext_conf.pred_empt_truth_black_king++;
	}

	else if (prediction == "1" && BOARD_TRUTH == "0") {
		ext_conf.pred_white_man_truth_empt++;
	}
	else if (prediction == "1" && BOARD_TRUTH == "1") {
		ext_conf.pred_white_man_truth_white_man++;
	}
	else if (prediction == "1" && BOARD_TRUTH == "2") {
		ext_conf.pred_white_man_truth_white_king++;
	}
	else if (prediction == "1" && BOARD_TRUTH == "3") {
		ext_conf.pred_white_man_truth_black_man++;
	}
	else if (prediction == "1" && BOARD_TRUTH == "4") {
		ext_conf.pred_white_man_truth_black_king++;
	}

	else if (prediction == "2" && BOARD_TRUTH == "0") {
		ext_conf.pred_white_king_truth_empt++;
	}
	else if (prediction == "2" && BOARD_TRUTH == "1") {
		ext_conf.pred_white_king_truth_white_man++;
	}
	else if (prediction == "2" && BOARD_TRUTH == "2") {
		ext_conf.pred_white_king_truth_white_king++;
	}
	else if (prediction == "2" && BOARD_TRUTH == "3") {
		ext_conf.pred_white_king_truth_black_man++;
	}
	else if (prediction == "2" && BOARD_TRUTH == "4") {
		ext_conf.pred_white_king_truth_black_king++;
	}

	else if (prediction == "3" && BOARD_TRUTH == "0") {
		ext_conf.pred_black_man_truth_empt++;
	}
	else if (prediction == "3" && BOARD_TRUTH == "1") {
		ext_conf.pred_black_man_truth_white_man++;
	}
	else if (prediction == "3" && BOARD_TRUTH == "2") {
		ext_conf.pred_black_man_truth_white_king++;
	}
	else if (prediction == "3" && BOARD_TRUTH == "3") {
		ext_conf.pred_black_man_truth_black_man++;
	}
	else if (prediction == "3" && BOARD_TRUTH == "4") {
		ext_conf.pred_black_man_truth_black_king++;
	}

	else if (prediction == "4" && BOARD_TRUTH == "0") {
		ext_conf.pred_black_king_truth_empt++;
	}
	else if (prediction == "4" && BOARD_TRUTH == "1") {
		ext_conf.pred_black_king_truth_white_man++;
	}
	else if (prediction == "4" && BOARD_TRUTH == "2") {
		ext_conf.pred_black_king_truth_white_king++;
	}
	else if (prediction == "4" && BOARD_TRUTH == "3") {
		ext_conf.pred_black_king_truth_black_man++;
	}
	else if (prediction == "4" && BOARD_TRUTH == "4") {
		ext_conf.pred_black_king_truth_black_king++;
	}
}

void display_extended_matrix(ExtendedConfusionMatrix conf) {
	const char space = ' ';
	const int nameWidth = 15;
	const int numWidth = 15;

	cout << endl;
	cout << left << setw(nameWidth) << setfill(space) << "" << left << setw(nameWidth) << setfill(space) << "" << left << setw(nameWidth) << setfill(space) << "" << left << setw(nameWidth) << setfill(space) << "Ground Truth" << left << setfill(space) << "" << endl;
	cout << left << setw(nameWidth) << setfill(space) << "" << left << setw(nameWidth) << setfill(space) << "" << left << setw(nameWidth) << setfill(space) << "No Piece" << left << setw(nameWidth) << setfill(space) << "White Man" << left << setw(nameWidth) << setfill(space) << "White King" << left << setw(nameWidth) << setfill(space) << "Black Man" << left << setfill(space) << "Black King" << endl;
	cout << left << setw(nameWidth) << setfill(space) << "" << left << setw(nameWidth) << setfill(space) << "No Piece" << left << setw(nameWidth) << setfill(space) << conf.pred_empt_truth_empt << left << setw(nameWidth) << setfill(space) << conf.pred_empt_truth_white_man << left << setw(nameWidth) << setfill(space) << conf.pred_empt_truth_white_king << left << setw(nameWidth) << setfill(space) << conf.pred_empt_truth_black_man << left << setw(nameWidth) << setfill(space) << conf.pred_empt_truth_black_king << endl;
	cout << left << setw(nameWidth) << setfill(space) << "" << left << setw(nameWidth) << setfill(space) << "White Man" << left << setw(nameWidth) << setfill(space) << conf.pred_white_man_truth_empt << left << setw(nameWidth) << setfill(space) << conf.pred_white_man_truth_white_man << left << setw(nameWidth) << setfill(space) << conf.pred_white_man_truth_white_king << left << setw(nameWidth) << setfill(space) << conf.pred_white_man_truth_black_man << left << setw(nameWidth) << setfill(space) << conf.pred_white_man_truth_black_king << endl;
	cout << left << setw(nameWidth) << setfill(space) << "Detected" << left << setw(nameWidth) << setfill(space) << "White King" << left << setw(nameWidth) << setfill(space) << conf.pred_white_king_truth_empt << left << setw(nameWidth) << setfill(space) << conf.pred_white_king_truth_white_man << left << setw(nameWidth) << setfill(space) << conf.pred_white_king_truth_white_king << left << setw(nameWidth) << setfill(space) << conf.pred_white_king_truth_black_man << left << setw(nameWidth) << setfill(space) << conf.pred_white_king_truth_black_king << endl;
	cout << left << setw(nameWidth) << setfill(space) << "" << left << setw(nameWidth) << setfill(space) << "Black Man" << left << setw(nameWidth) << setfill(space) << conf.pred_black_man_truth_empt << left << setw(nameWidth) << setfill(space) << conf.pred_black_man_truth_white_man << left << setw(nameWidth) << setfill(space) << conf.pred_black_man_truth_white_king << left << setw(nameWidth) << setfill(space) << conf.pred_black_man_truth_black_man << left << setw(nameWidth) << setfill(space) << conf.pred_black_man_truth_black_king << endl;
	cout << left << setw(nameWidth) << setfill(space) << "" << left << setw(nameWidth) << setfill(space) << "Black King" << left << setw(nameWidth) << setfill(space) << conf.pred_black_king_truth_empt << left << setw(nameWidth) << setfill(space) << conf.pred_black_king_truth_white_man << left << setw(nameWidth) << setfill(space) << conf.pred_black_king_truth_white_king << left << setw(nameWidth) << setfill(space) << conf.pred_black_king_truth_black_man << left << setw(nameWidth) << setfill(space) << conf.pred_black_king_truth_black_king << endl;
	cout << endl;
}

// ********************************************************************************************

void MyApplication()
{
	string video_filename("Media/DraughtsGame1.avi");
	VideoCapture video;
	video.open(video_filename);

	//int pieces[32];
	string black_pieces_filename("Media/DraughtsGame1BlackPieces.jpg");
	Mat black_pieces_image = imread(black_pieces_filename, -1);
	string white_pieces_filename("Media/DraughtsGame1WhitePieces.jpg");
	Mat white_pieces_image = imread(white_pieces_filename, -1);
	string black_squares_filename("Media/DraughtsGame1BlackSquares.jpg");
	Mat black_squares_image = imread(black_squares_filename, -1);
	string white_squares_filename("Media/DraughtsGame1WhiteSquares.jpg");
	Mat white_squares_image = imread(white_squares_filename, -1);

	string move0_ground_truth_filename = "Media/DraughtsGame1Move0GroundTruth.png";
	Mat move0_ground_truth_image = imread(move0_ground_truth_filename, -1);

	string background_filename("Media/DraughtsGame1EmptyBoard.JPG");
	Mat static_background_image = imread(background_filename, -1);
	if ((!video.isOpened()) || (black_pieces_image.empty()) || (white_pieces_image.empty()) || (black_squares_image.empty()) || (white_squares_image.empty()) || (static_background_image.empty()) || (move0_ground_truth_image.empty()))
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
		if (move0_ground_truth_image.empty())
			cout << "Cannot open image file: " << move0_ground_truth_filename << endl;
	}
	else
	{
		// ********************************************** PART 1 *******************************************************
		/*
		int image_index = 0;

		cout << "PART 1. " << endl << endl;

		cout << "Image number " << image_index << " is classified into white pieces, black pieces, white squares, black squares and background." << endl << endl;

		Mat output;

		Part1(GROUND_TRUTH_FOR_BOARD_IMAGES[image_index][0], white_pieces_image, black_pieces_image, white_squares_image, black_squares_image, output);

		double correct_pixels = 0;
		double total_pixels = 0;

		// if the user has selected the first image, the ground truth can be used for a quantitative assessment of the solution
		if (image_index == 0) {
			for (int row = 0; row < output.rows; row++) {
				for (int col = 0; col < output.cols; col++) {
					total_pixels++;

					Vec3b check_pixel = output.at<Vec3b>(row, col);
					Vec3b ground_truth_pixel = move0_ground_truth_image.at<Vec3b>(row, col);

					if (check_pixel[0] == ground_truth_pixel[0] && check_pixel[1] == ground_truth_pixel[1] && check_pixel[2] == ground_truth_pixel[2]) {
						correct_pixels++;
					}

				}
			}

			double accuracy = (correct_pixels / total_pixels) * 100;

			imshow("Pixel Classification", output);
			imshow("Ground Truth", move0_ground_truth_image);
			cout << "Correct Pixels: " << correct_pixels << endl;
			cout << "Total Pixels: " << total_pixels << endl;
			cout.precision(3);
			cout << "Accuracy = " << accuracy << "%" << endl << endl;
		}

		else {
			imshow("Pixel Classification", output);
			cout << "Set the image_index variable to 0 next time to get a quantitative assessment of the solution." << endl << endl;
		}
		*/
		// ********************************************** PART 2 *******************************************************
		/*
		cout << setw(10) << "PART 2. " << endl << endl;

		cout << "Processing the ground truth still images..." << endl << endl;

		// initialise a struct to record the confusion matrix data associated with the move predictions
		// this should be in a function but for some reason an error was being thrown so the struct parameters are initialised here instead
		ConfusionMatrix conf;
		conf.pred_empt_truth_empt = 0;
		conf.pred_empt_truth_white = 0;
		conf.pred_empt_truth_black = 0;
		conf.pred_white_truth_empt = 0;
		conf.pred_white_truth_white = 0;
		conf.pred_white_truth_black = 0;
		conf.pred_black_truth_empt = 0;
		conf.pred_black_truth_white = 0;
		conf.pred_black_truth_black = 0;

		// convert the provided ground truth data to a two-dimensional array: this makes the computation of a confusion matrix simpler
		string BOARD_TRUTH[69][32];
		string buffer[32];
		for (int i = 0; i < 32; i++) {
			buffer[i] = "0"; // start with an empty buffer
		}
		for (int i = 0; i < 69; i++) {
			board_representation_from_strings_to_array(GROUND_TRUTH_FOR_BOARD_IMAGES[i][1], GROUND_TRUTH_FOR_BOARD_IMAGES[i][2], buffer);
			for (int j = 0; j < 32; j++) {
				BOARD_TRUTH[i][j] = buffer[j];
				buffer[j] = "0"; // reset the buffer to all 0s for the next iteration
			}
		}

		// process each of the 69 images one by one and compare the resulting predictions array against the correct row of the 2D ground truth array
		string predictions[32];
		string whites = "";
		string blacks = "";
		for (int i = 0; i < 69; i++) {
			for (int j = 0; j < 32; j++) {
				predictions[j] = "0"; // start with empty predictions
			}

			Part2(GROUND_TRUTH_FOR_BOARD_IMAGES[i][0], white_pieces_image, black_pieces_image, black_squares_image, whites, blacks); // IMAGE PROCESSING OCCURS HERE

			board_representation_from_strings_to_array(whites, blacks, predictions); // convert the prediction strings to an array for easy generation of confusion matrix

			// compare the predicted value of each square against the ground truth and update the confusion matrix
			for (int j = 0; j < 32; j++) {
				if (predictions[j] == "0" && BOARD_TRUTH[i][j] == "0") {
					conf.pred_empt_truth_empt++;
				}
				else if (predictions[j] == "0" && BOARD_TRUTH[i][j] == "1") {
					conf.pred_empt_truth_white++;
				}
				else if (predictions[j] == "0" && BOARD_TRUTH[i][j] == "2") {
					conf.pred_empt_truth_black++;
				}
				else if (predictions[j] == "1" && BOARD_TRUTH[i][j] == "0") {
					conf.pred_white_truth_empt++;
				}
				else if (predictions[j] == "1" && BOARD_TRUTH[i][j] == "1") {
					conf.pred_white_truth_white++;
				}
				else if (predictions[j] == "1" && BOARD_TRUTH[i][j] == "2") {
					conf.pred_white_truth_black++;
				}
				else if (predictions[j] == "2" && BOARD_TRUTH[i][j] == "0") {
					conf.pred_black_truth_empt++;
				}
				else if (predictions[j] == "2" && BOARD_TRUTH[i][j] == "1") {
					conf.pred_black_truth_white++;
				}
				else if (predictions[j] == "2" && BOARD_TRUTH[i][j] == "2") {
					conf.pred_black_truth_black++;
				}
			}
		}

		cout << "Confusion Matrix: " << endl;
		display_matrix(conf); // display the confusion matrix
		*/
		// ********************************************** PART 3 *******************************************************
		/*
		cout << "Part 3: Screenshot generation in process... " << endl << endl;

		// Process video frame by frame
		Mat current_frame;
		video.set(cv::CAP_PROP_POS_FRAMES, 1);
		video >> current_frame;
		double last_time = static_cast<double>(getTickCount());
		double frame_rate = video.get(cv::CAP_PROP_FPS);
		double time_between_frames = 1000.0 / frame_rate;

		Ptr<BackgroundSubtractorMOG2> gmm = createBackgroundSubtractorMOG2();

		int frame_count = 0;
		int frame_number = 0;
		int screenshot_number = 0;
		string board_screenshot_name = "Move";
		string foreground_screenshot_name = "Fore";
		string back_proj_img_name = "BP";

		imwrite("Media/Screenshots/Move0.png", current_frame); // store the first frame

		//while (!current_frame.empty()) // this was throwing an exception at the end of the video - unsure how to resolve this so using one of the last frames as a stopping point instead
		while (frame_number < 1527) {
			double current_time = static_cast<double>(getTickCount());
			double duration = (current_time - last_time) / getTickFrequency() / 1000.0;
			int delay = (time_between_frames > duration) ? ((int)(time_between_frames - duration)) : 1;
			last_time = current_time;
			video >> current_frame;

			Mat foreground_mask, thresholded_image, closed_image;
			Mat structuring_element(3, 3, CV_8U, Scalar(1));
			Mat foreground_image = Mat::zeros(current_frame.size(), CV_8UC3);

			// apply gaussian mixture model to the video
			gmm->apply(current_frame, foreground_mask);
			// Clean the resultant binary (moving pixel) mask using an opening.
			threshold(foreground_mask, thresholded_image, 150, 255, THRESH_BINARY);
			Mat moving_incl_shadows, shadow_points;
			threshold(foreground_mask, moving_incl_shadows, 50, 255, THRESH_BINARY);
			absdiff(thresholded_image, moving_incl_shadows, shadow_points);
			Mat cleaned_foreground_mask;
			morphologyEx(thresholded_image, closed_image, MORPH_CLOSE, structuring_element);
			morphologyEx(closed_image, cleaned_foreground_mask, MORPH_OPEN, structuring_element);
			foreground_image.setTo(Scalar(0, 0, 0));
			current_frame.copyTo(foreground_image, cleaned_foreground_mask);
			// Create an average background image
			Mat mean_background_image;
			gmm->getBackgroundImage(mean_background_image);

			// use the foreground image to determine when to take a screenshot
			// use a frame number delay (15 frames) to reduce the number of duplicate screenshots taken
			// (hopefully after 15 frames another piece has been or is currently being moved)
			if (frame_count > 15) {

				int non_zero_pixel_count = 0;
				for (int row = 0; row < foreground_image.rows; row++) {
					for (int col = 0; col < foreground_image.cols; col++) {
						Vec3b pixel = foreground_image.at<Vec3b>(row, col);
						if (pixel[0] != 0 || pixel[1] != 0 && pixel[2] != 0)
							non_zero_pixel_count++;
					}
				}

				// if the number of non-zero pixels in the foreground image is less than the threshold, it means things are pretty stationary in the current frame
				// probably a good time to take a screenshot
				if (non_zero_pixel_count < 2000) {
					screenshot_number++;
					frame_count = 0; // reset the frame count to 0, i.e. take no more screenshots for another 15 frames after this one

					board_screenshot_name = board_screenshot_name + to_string(screenshot_number);
					foreground_screenshot_name = foreground_screenshot_name + to_string(screenshot_number);
					back_proj_img_name = back_proj_img_name + to_string(screenshot_number);
					//cout << "Writing file " << board_screenshot_name << " to Media/Screenshots" << endl << endl;
					imwrite("Media/Screenshots/" + board_screenshot_name + ".png", current_frame);
					//imwrite("Media/Foregrounds/" + foreground_screenshot_name + ".png", foreground_image);
					board_screenshot_name = "Move";
					foreground_screenshot_name = "Fore";
				}
			}

			frame_number++;
			frame_count++;

			Mat temp_gaussian_output = JoinImagesHorizontally(current_frame, "", mean_background_image, "GMM Background", 4);
			Mat gaussian_output = JoinImagesHorizontally(temp_gaussian_output, "", foreground_image, "Foreground", 4);
			imshow("Gaussian Mixture Model", gaussian_output);

			char c = cv::waitKey(1);  // If you replace delay with 1 it will play the video as quickly as possible.

		}

		// now armed with a collection of still images taken of the draughts board
		// process these in the same manner as the ground truth stills
		// eliminate any duplicates
		// then process the moves

		cout << endl << "Processing screenshots, eliminating duplicates and processing moves - this may take a while... " << endl << endl;

		string prefix = "Screenshots/Move";
		string previous, current;

		vector<string> white_pieces, black_pieces;
		Mat image;

		string whites1 = "";
		string whites2 = "";
		string blacks1 = "";
		string blacks2 = "";

		int count = 1;

		string saved_still = "Media/Stills/Board";

		for (int i = 0; i < 83; i++) {
			previous = prefix + to_string(i) + ".png";
			current = prefix + to_string(i + 1) + ".png";

			Part2(previous, white_pieces_image, black_pieces_image, black_squares_image, whites1, blacks1); // get the current and previous piece positions on the board
			Part2(current, white_pieces_image, black_pieces_image, black_squares_image, whites2, blacks2); // these can then be fed to the move processing subroutine

			if (whites1 != whites2 || blacks1 != blacks2) { // if one of the sides has made a move (i.e. no duplicate detected)
				white_pieces.push_back(whites1);
				black_pieces.push_back(blacks1);
				Mat still_to_save = imread("Media/" + previous); // read in the image i want to save
				imwrite(saved_still + to_string(count) + ".png", still_to_save); // write it to the new folder location
				count++;
			}

			prefix = "Screenshots/Move";
		}

		int moves[68][2];
		//int white_moves[72][2], black_moves[72][2];

		cout << setw(20) <<"Part 3, Move Detection: " << endl << endl;

		cout << "Move" << setw(10) << " " << "G. T." << setw(12) << " " << "Detected" << endl;

		int turn = 1;

		for (int i = 1; i < 68; i++) {
			// use the "turn" variable as a boolean indicator of which side's turn it is
			if (turn > 0) {
				turn = turn * -1;
				process_move(white_pieces[i - 1], white_pieces[i], moves[i][0], moves[i][1]);
				if (moves[i][0] == 0 || moves[i][1] == 0) { // if no move has occurred, this means we are checking the wrong side
					process_move(black_pieces[i - 1], black_pieces[i], moves[i][0], moves[i][1]);
					turn = turn * -1;
				}

			}
			else {
				turn = turn * -1;
				process_move(black_pieces[i - 1], black_pieces[i], moves[i][0], moves[i][1]);
				if (moves[i][0] == 0 || moves[i][1] == 0) {
					process_move(white_pieces[i - 1], white_pieces[i], moves[i][0], moves[i][1]);
					turn = turn * -1;
				}
			}
		}

		for (int i = 1; i < 69; i++) {
			string out = "";
			if (moves[i][0] > -2 && moves[i][1] > -2) {
				out = to_string(moves[i][0]) + "-" + to_string(moves[i][1]);
			}
			else
				out = "N/A";
			string ground_t = to_string(GROUND_TRUTH_FOR_DRAUGHTSGAME1_VIDEO_MOVES[i - 1][1]) + "-" + to_string(GROUND_TRUTH_FOR_DRAUGHTSGAME1_VIDEO_MOVES[i - 1][2]);

			if (i == 35) {
				cout << endl << endl;
			}

			if (moves[i][0] == -1 || moves[i][1] == -1)
				cout << setw(2) << i << setw(12) << " " << GROUND_TRUTH_FOR_DRAUGHTSGAME1_VIDEO_MOVES[i - 1][1] << "-" << GROUND_TRUTH_FOR_DRAUGHTSGAME1_VIDEO_MOVES[i - 1][2] << setw(12) << " " << "x-x" << endl;
			else if (moves[i][0] == 0 || moves[i][1] == 0)
				cout << setw(2) << i << setw(12) << " " << GROUND_TRUTH_FOR_DRAUGHTSGAME1_VIDEO_MOVES[i - 1][1] << " - " << GROUND_TRUTH_FOR_DRAUGHTSGAME1_VIDEO_MOVES[i - 1][2] << setw(12) << " " << "o-o" << endl;
			else
				cout << setw(2) << i << setw(12) << " " << setw(5) << ground_t << setw(12) << " " << out << endl;
		}

		cout << "\nKey: " << endl;
		cout << "x-x = Mismatched number of pieces before and after, i.e. an invalid move was detected." << endl;
		cout << "o-o = Pieces are identical before and after, i.e. no move was detected." << endl << endl;

		cout << "Note: Move 38 is missed by the video processing algorithm and thus one move less than the ground truth is detected." << endl;
		cout << "As a result, each output Detected[i] after Move 38 corresponds to GroundTruth[i+1]" << endl;
		*/
		// ********************************************** PART 4 *******************************************************

		// ********************************************** PART 5 *******************************************************

		ExtendedConfusionMatrix ext_conf;
		initialise_extended_matrix(ext_conf);

		string BOARD_TRUTH[69][32];
		string buffer[32];
		string predictions[32];
		string white_team = "";
		string black_team = "";

		for (int i = 0; i < 32; i++) {
			buffer[i] = "0"; // start with an empty buffer
		}
		for (int i = 0; i < 69; i++) {
			extended_board_representation_from_strings_to_array(GROUND_TRUTH_FOR_BOARD_IMAGES[i][1], GROUND_TRUTH_FOR_BOARD_IMAGES[i][2], buffer);
			for (int j = 0; j < 32; j++) {
				BOARD_TRUTH[i][j] = buffer[j];
				buffer[j] = "0"; // reset the buffer to all 0s for the next iteration
			}
		}


		// &&&&&&&&&&&&&&&&&&&&&&&&&&& WORK OUT HOW TO LOOP THIS FOR 69 FRAMES WITHOUT AN ASSERTION ERROR AND PART 5 IS COMPLETE
		
		for (int i = 0; i < 1; i++) {

			for (int j = 0; j < 32; j++) {
				predictions[j] = "0"; // start with empty predictions
			}

			Mat perspective_transformed_img = perspective(GROUND_TRUTH_FOR_BOARD_IMAGES[i][0]);
			vector<Vec3f> circles;

			Part5(perspective_transformed_img, circles, white_pieces_image, black_pieces_image, black_squares_image, white_team, black_team); // IMAGE PROCESSING OCCURS HERE

			extended_board_representation_from_strings_to_array(white_team, black_team, predictions); // convert the prediction strings to an array for easy generation of confusion matrix

			// compare the predicted value of each square against the ground truth and update the confusion matrix
			for (int j = 0; j < 32; j++) {
				update_extended_confusion_matrix(predictions[j], BOARD_TRUTH[i][j], ext_conf);
			}

		}
		// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

		display_extended_matrix(ext_conf);

		// *********************************************** ~~~ *********************************************************

		//cv::destroyAllWindows();
		
	}
}