#include <tbb/task_scheduler_init.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include <tbb/blocked_range.h>
#include <tbb/blocked_range2d.h>
#include <cstdlib>
#include <math.h>
#include <random>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <functional>
#include <vector>
#include <string>
#include <FreeImagePlus.h>
#include <mutex>

using namespace std;
using namespace std::chrono;
using namespace tbb;

// Global constants
const float PI = 3.142f;


// Return the square of a
template <typename T>
T sqr(const T& a) {

	return a * a;
}


const int K_SIZE = 3;


void generateKernel(float kernel[K_SIZE][K_SIZE], double sigma) {

	auto lambda = [=](float x, float y) {
		return 1.0f / (2.0f * PI * sqr(sigma)) * exp(-(sqr(x - K_SIZE / 2) + sqr(y - K_SIZE / 2)) / (2.0f * sqr(sigma)));
	};

	float sum = 0;
	for (int x = 0; x < K_SIZE; x++) {
		for (int y = 0; y < K_SIZE; y++) {
			kernel[x][y] = lambda(x, y);
			sum += kernel[x][y];
		}
	}

	for (int x = 0; x < K_SIZE; x++) {
		for (int y = 0; y < K_SIZE; y++) {
			kernel[x][y] /= sum;
		}
	}
}


void grayscale_gaussian_blur() {

	// Setup and load input image dataset
	fipImage inputImage;
	inputImage.load("../Images/low_rez.jpg");
	inputImage.convertToFloat();

	auto width = inputImage.getWidth();
	auto height = inputImage.getHeight();
	float* inputBuffer = (float*)inputImage.accessPixels();

	if (!inputImage.isValid())
		throw;

	// Setup output image array
	fipImage outputImage;
	outputImage = fipImage(FIT_FLOAT, width, height, 32);
	float *outputBuffer = (float*)outputImage.accessPixels();

	// Get total array size
	uint64_t numElements = width * height;

	float kernel[K_SIZE][K_SIZE];
	generateKernel(kernel, 2);

	int mid = K_SIZE / 2;

	// Test sequential vs. parallel_for versions - run test multiple times and track speed-up
	double meanSpeedup = 0.0f;
	uint64_t numTests = 50;

	for (uint64_t testIndex = 0; testIndex < numTests; ++testIndex) {

		auto st1 = high_resolution_clock::now();

		for (int y = 0; y < height; y++) {
			for (int x = 0; x < width; x++) {
				for (int i = -mid; i <= mid; i++) {
					for (int j = -mid; j <= mid; j++) {
						uint64_t currInput = (y + i) * width + x + j;
						uint64_t currOutput = y * width + x;
						if (currInput < numElements) {
							outputBuffer[currOutput] +=
								inputBuffer[currInput] * kernel[i + mid][j + mid];
						}
					}
				}
			}
		}
		auto st2 = high_resolution_clock::now();
		auto st_dur = duration_cast<microseconds>(st2 - st1);
		std::cout << "sequential negative operation took = " << st_dur.count() << "\n";

		// parallel_for version

		auto pt1 = high_resolution_clock::now();
		
		tbb::parallel_for(blocked_range2d<uint64_t, uint64_t>(0, height, 8, 0, width, width >> 2),
			[=](const blocked_range2d<uint64_t, uint64_t> &r) {

			auto y1 = r.rows().begin();
			auto y2 = r.rows().end();
			auto x1 = r.cols().begin();
			auto x2 = r.cols().end();

			for (uint64_t y = y1; y < y2; ++y) {
				for (uint64_t x = x1; x < x2; ++x) {
					for (int i = -mid; i <= mid; i++) {
						for (int j = -mid; j <= mid; j++) {
							uint64_t currInput = (y + i) * width + x + j;
							uint64_t currOutput = y * width + x;
							if (currInput < numElements) {
								outputBuffer[currOutput] +=
									inputBuffer[currInput] * kernel[i + mid][j + mid];
							}
						}
					}
				}
			}
		});
		
		auto pt2 = high_resolution_clock::now();
		auto pt_dur = duration_cast<microseconds>(pt2 - pt1);
		std::cout << "parallel negative operation took = " << pt_dur.count() << "\n";

		double speedup = double(st_dur.count()) / double(pt_dur.count());
		std::cout << "Test " << testIndex << " speedup = " << speedup << endl;

		meanSpeedup += speedup;
	}

	meanSpeedup /= double(numTests);
	std::cout << "Mean speedup = " << meanSpeedup << endl;




	outputImage.convertToType(FREE_IMAGE_TYPE::FIT_BITMAP);
	outputImage.convertTo24Bits();
	outputImage.save("Images/grayscale_gaussian_k3.bmp");

}

int main()
{
	int nt = task_scheduler_init::default_num_threads();
	task_scheduler_init T(nt);

	//Part 1 (Greyscale Gaussian blur): -----------DO NOT REMOVE THIS COMMENT----------------------------//

	//grayscale_gaussian_blur();


	//Part 2 (Colour image processing): -----------DO NOT REMOVE THIS COMMENT----------------------------//


	// Setup Input image array
	fipImage inputImage1;
	inputImage1.load("../Images/render_1.png");
	fipImage inputImage2;
	inputImage2.load("../Images/render_2.png");

	unsigned int width = inputImage1.getWidth() > inputImage2.getWidth() ? inputImage1.getWidth() : inputImage2.getWidth();
	unsigned int height = inputImage1.getHeight() > inputImage2.getHeight() ? inputImage1.getHeight() : inputImage2.getHeight();

	int total_pixels = width * height;

	// Setup Output image array
	fipImage outputImage;
	outputImage = fipImage(FIT_BITMAP, width, height, 24);

	//2D Vector to hold the RGB colour data of an image
	vector<vector<RGBQUAD>> rgbValues;
	rgbValues.resize(height, vector<RGBQUAD>(width));

	RGBQUAD rgb;  //FreeImage structure to hold RGB values of a single pixel

	// Test sequential vs. parallel_for versions - run test multiple times and track speed-up
	double meanSpeedup = 0.0f;
	uint64_t numTests = 50;


	auto st1 = high_resolution_clock::now();
	/*
			//Extract colour data from image and store it as individual RGBQUAD elements for every pixel
			for (int y = 0; y < height; y++)
			{
				for (int x = 0; x < width; x++)
				{
					inputImage.getPixelColor(x, y, &rgb); //Extract pixel(x,y) colour data and place it in rgb

					rgbValues[y][x].rgbRed = rgb.rgbRed;
					rgbValues[y][x].rgbGreen = rgb.rgbGreen;
					rgbValues[y][x].rgbBlue = rgb.rgbBlue;
				}
			}
			for (int y = 0; y < height; y++)
			{
				for (int x = 0; x < width; x++)
				{
					outputImage.setPixelColor(x, y, &rgbValues[y][x]);
				}
			} */

	auto st2 = high_resolution_clock::now();
	auto st_dur = duration_cast<microseconds>(st2 - st1);
	std::cout << "sequential negative operation took = " << st_dur.count() << "\n";

	// parallel_for version

	auto pt1 = high_resolution_clock::now();

	tbb::parallel_for(blocked_range2d<uint64_t, uint64_t>(0, height, 0, width),
		[&](const blocked_range2d<uint64_t, uint64_t> &r) {
		RGBQUAD _rgb1;
		RGBQUAD _rgb2;
		auto y1 = r.rows().begin();
		auto y2 = r.rows().end();
		auto x1 = r.cols().begin();
		auto x2 = r.cols().end();

		for (uint64_t y = y1; y != y2; y++) {
			for (uint64_t x = x1; x != x2; x++) {

				inputImage1.getPixelColor(x, y, &_rgb1); //Extract pixel(x,y) colour data and place it in rgb
				inputImage2.getPixelColor(x, y, &_rgb2);

				rgbValues[y][x].rgbRed = abs(_rgb1.rgbRed - _rgb2.rgbRed);
				rgbValues[y][x].rgbGreen = abs(_rgb1.rgbGreen - _rgb2.rgbGreen);
				rgbValues[y][x].rgbBlue = abs(_rgb1.rgbBlue - _rgb2.rgbBlue);

				if (rgbValues[y][x].rgbRed != 0 || rgbValues[y][x].rgbGreen != 0 || rgbValues[y][x].rgbBlue != 0)
				{
					rgbValues[y][x].rgbRed = 255;
					rgbValues[y][x].rgbGreen = 255;
					rgbValues[y][x].rgbBlue = 255;
				}
			}
		}
	});


	int count = parallel_reduce(
		blocked_range2d<uint64_t, uint64_t>(0, height, 0, width),
		0.0f,
		[&](const blocked_range2d<uint64_t, uint64_t> &r, int value)->int {
		auto y1 = r.rows().begin();
		auto y2 = r.rows().end();
		auto x1 = r.cols().begin();
		auto x2 = r.cols().end();

		for (uint64_t y = y1; y != y2; y++) {
			for (uint64_t x = x1; x != x2; x++) {

				if (rgbValues[y][x].rgbRed == 255 && rgbValues[y][x].rgbGreen == 255 && rgbValues[y][x].rgbBlue == 255)
				{
					value++;
				}
			}
		}
		return value;
	},
		[&](float x, float y)->int {
		return x + y;
	}
	);


	int x = rand() % width;
	int y = rand() % height;
	cout << "Random Poisition:  x = " << x << "  y = " << y << endl;
	rgbValues[y][x].rgbRed = 255;
	rgbValues[y][x].rgbGreen = 0;
	rgbValues[y][x].rgbBlue = 0;


	int posX = -1;
	int posY = -1;


	tbb::parallel_for(
		blocked_range2d<uint64_t, uint64_t>(0, height, 0, width),
		[&](const blocked_range2d<uint64_t, uint64_t> &r) {
		auto y1 = r.rows().begin();
		auto y2 = r.rows().end();
		auto x1 = r.cols().begin();
		auto x2 = r.cols().end();

		for (uint64_t y = y1; y != y2; y++) {
			for (uint64_t x = x1; x != x2; x++) {
				if (rgbValues[y][x].rgbRed == 255 && rgbValues[y][x].rgbGreen == 0 && rgbValues[y][x].rgbBlue == 0) {
					if (task::self().cancel_group_execution()) {
						posX = x;
						posY = y;
					}
				}
			}
		}
	});

	cout << "Found red pixel at:  x = " << posX << "  y = " << posY << endl;


	tbb::parallel_for(blocked_range2d<uint64_t, uint64_t>(0, height, 0, width),
		[&](const blocked_range2d<uint64_t, uint64_t> &r) {
		auto y1 = r.rows().begin();
		auto y2 = r.rows().end();
		auto x1 = r.cols().begin();
		auto x2 = r.cols().end();

		for (uint64_t y = y1; y < y2; ++y) {
			for (uint64_t x = x1; x < x2; ++x) {
				outputImage.setPixelColor(x, y, &rgbValues[y][x]);
			}
		}
	});

	auto pt2 = high_resolution_clock::now();
	auto pt_dur = duration_cast<microseconds>(pt2 - pt1);
	std::cout << "parallel negative operation took = " << pt_dur.count() << "\n";

	double speedup = double(st_dur.count()) / double(pt_dur.count());
	std::cout << "speedup = " << speedup << endl;

	meanSpeedup += speedup;




	meanSpeedup /= double(numTests);
	float percentage = (float)count * 100 / total_pixels;
	std::cout << "Mean speedup = " << meanSpeedup << endl;
	std::cout << "Total Pixels: " << total_pixels << std::endl;
	std::cout << "White pixels: " << count << "   " << percentage << "%" << std::endl;

	cout << "saving image..." << endl;

	//Save the processed image
	outputImage.save("Images/RGB_processed.png");


	return 0;
}