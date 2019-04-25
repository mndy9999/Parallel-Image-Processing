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
#include <fstream>

using namespace std;
using namespace std::chrono;
using namespace tbb;

// Global constants
const float PI = 3.142f;

RGBQUAD white = { 255, 255, 255 };
RGBQUAD black = { 0, 0, 0 };
RGBQUAD red = { 0, 0, 255 };
RGBQUAD green = { 0, 255, 0 };
RGBQUAD blue = { 255, 0, 0 };


// Return the square of a
template <typename T>
T sqr(const T& a) {

	return a * a;
}


const int K_SIZE = 3;


void generateKernel(float kernel[K_SIZE][K_SIZE], double sigma, bool parallel = true) {

	auto lambda = [=](float x, float y) {
		return 1.0f / (2.0f * PI * sqr(sigma)) * exp(-(sqr(x - K_SIZE / 2) + sqr(y - K_SIZE / 2)) / (2.0f * sqr(sigma)));
	};

	float sum = 0;

	if (parallel) {

		tbb::parallel_for(blocked_range2d<uint64_t, uint64_t>(0, K_SIZE, 0, K_SIZE),
			[=](const blocked_range2d<uint64_t, uint64_t> &r) {

			auto y1 = r.rows().begin();
			auto y2 = r.rows().end();
			auto x1 = r.cols().begin();
			auto x2 = r.cols().end();

			for (uint64_t y = y1; y < y2; ++y) {
				for (uint64_t x = x1; x < x2; ++x) {
					kernel[x][y] = lambda(x, y);
				}
			}
		});

		sum = parallel_reduce(
			blocked_range2d<uint64_t, uint64_t>(0, K_SIZE, 0, K_SIZE),
			0.0f,
			[&](const blocked_range2d<uint64_t, uint64_t> &r, float value)->float {

			auto y1 = r.rows().begin();
			auto y2 = r.rows().end();
			auto x1 = r.cols().begin();
			auto x2 = r.cols().end();

			for (uint64_t y = y1; y != y2; y++) {
				for (uint64_t x = x1; x != x2; x++) {
					value += kernel[y][x];
				}
			}
			return value;
		},
			[&](float x, float y)->float {
			return x + y;
		}
		);
		tbb::parallel_for(blocked_range2d<uint64_t, uint64_t>(0, K_SIZE, 0, K_SIZE),
			[&](const blocked_range2d<uint64_t, uint64_t> &r) {

			auto y1 = r.rows().begin();
			auto y2 = r.rows().end();
			auto x1 = r.cols().begin();
			auto x2 = r.cols().end();

			for (uint64_t y = y1; y < y2; ++y) {
				for (uint64_t x = x1; x < x2; ++x) {
					kernel[x][y] /= sum;
				}
			}
		});
	}


	else {
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
}



void grayscale_gaussian_blur() {

	ofstream file;
	file.open("Results/results.csv");
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

	double meanSpeedup = 0.0f;
	uint64_t numTests = 50;

	for (uint64_t testIndex = 0; testIndex < numTests; ++testIndex) {

		auto st1 = high_resolution_clock::now();

		generateKernel(kernel, 2, false);

		auto st2 = high_resolution_clock::now();
		auto st_dur = duration_cast<microseconds>(st2 - st1);
		std::cout << "sequential negative operation took = " << st_dur.count() << "\n";

		auto pt1 = high_resolution_clock::now();

		generateKernel(kernel, 2, true);

		auto pt2 = high_resolution_clock::now();
		auto pt_dur = duration_cast<microseconds>(pt2 - pt1);
		std::cout << "parallel negative operation took = " << pt_dur.count() << "\n";



		double speedup = double(st_dur.count()) / double(pt_dur.count());
		std::cout << "Test " << testIndex << " speedup = " << speedup << endl;

		meanSpeedup += speedup;

		file << st_dur.count() << "," << pt_dur.count() << "," << speedup << endl;

	}

	meanSpeedup /= double(numTests);
	std::cout << "Mean speedup = " << meanSpeedup << endl;

	int mid = K_SIZE / 2;
	file << "\n\n";

	for (uint64_t testIndex = 0; testIndex < numTests; ++testIndex) {

		auto st1 = high_resolution_clock::now();

		for (int y = 0; y < height; y++) {
			for (int x = 0; x < width; x++) {
				int currOutput = y * width + x;
				for (int i = -mid; i <= mid; i++) {
					for (int j = -mid; j <= mid; j++) {
						if (y + i >= 0 && x + j >= 0) {
							int currInput = (y + i) * width + x + j;
							if (currInput < numElements) {
								outputBuffer[currOutput] += inputBuffer[currInput] * kernel[i + mid][j + mid];
							}
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
		
		tbb::parallel_for(blocked_range2d<uint64_t, uint64_t>(0, height, 0, width),
			[=](const blocked_range2d<uint64_t, uint64_t> &r) {

			auto y1 = r.rows().begin();
			auto y2 = r.rows().end();
			auto x1 = r.cols().begin();
			auto x2 = r.cols().end();

			for (int y = y1; y < y2; ++y) {
				for (int x = x1; x < x2; ++x) {
					int currOutput = y * width + x;
					for (int i = -mid; i <= mid; i++) {
						for (int j = -mid; j <= mid; j++) {
							if (y + i >= 0 && x + j >= 0) {
								int currInput = (y + i) * width + x + j;							
								if (currInput < numElements ) {
									outputBuffer[currOutput] += inputBuffer[currInput] * kernel[i + mid][j + mid];
								}
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

		file << st_dur.count() << "," << pt_dur.count() << "," << speedup << endl;

	}

	meanSpeedup /= double(numTests);
	std::cout << "Mean speedup = " << meanSpeedup << endl;

	file.close();


	outputImage.convertToType(FREE_IMAGE_TYPE::FIT_BITMAP);
	outputImage.convertTo24Bits();
	outputImage.save("Images/grayscale_gaussian_k3.bmp");

}

fipImage computeAbsoluteDifference(fipImage inputImage1, fipImage inputImage2, bool parallel = true) {

	int height = inputImage1.getHeight();
	int width = inputImage2.getWidth();

	fipImage outputImage;
	outputImage = fipImage(FIT_BITMAP, width, height, 24);


	if (parallel) {
		tbb::parallel_for(blocked_range2d<uint64_t, uint64_t>(0, height, 0, width),
			[&](const blocked_range2d<uint64_t, uint64_t> &r) {

			RGBQUAD _rgb1;
			RGBQUAD _rgb2;
			RGBQUAD rgb;

			auto y1 = r.rows().begin();
			auto y2 = r.rows().end();
			auto x1 = r.cols().begin();
			auto x2 = r.cols().end();

			for (uint64_t y = y1; y != y2; y++) {
				for (uint64_t x = x1; x != x2; x++) {

					//Extract pixel(x,y) colour data
					inputImage1.getPixelColor(x, y, &_rgb1); 
					inputImage2.getPixelColor(x, y, &_rgb2);

					//calculate absolute difference for each colour value
					rgb.rgbRed = abs(_rgb1.rgbRed - _rgb2.rgbRed);
					rgb.rgbGreen = abs(_rgb1.rgbGreen - _rgb2.rgbGreen);
					rgb.rgbBlue = abs(_rgb1.rgbBlue - _rgb2.rgbBlue);

					//set pixel colour in the new image
					outputImage.setPixelColor(x, y, &rgb);
				}
			}
		});
	}
	else {

		RGBQUAD _rgb1;
		RGBQUAD _rgb2;
		RGBQUAD rgb;
		for (uint64_t y = 0; y < height; y++) {
			for (uint64_t x = 0; x < width; x++) {

				inputImage1.getPixelColor(x, y, &_rgb1); //Extract pixel(x,y) colour data and place it in rgb
				inputImage2.getPixelColor(x, y, &_rgb2);

				rgb.rgbRed = abs(_rgb1.rgbRed - _rgb2.rgbRed);
				rgb.rgbGreen = abs(_rgb1.rgbGreen - _rgb2.rgbGreen);
				rgb.rgbBlue = abs(_rgb1.rgbBlue - _rgb2.rgbBlue);

				outputImage.setPixelColor(x, y, &rgb);
			}
		}
	}
	return outputImage;
}

void applyBinaryThreshold(fipImage &image, bool parallel = true) {

	int height = image.getHeight();
	int width = image.getWidth();
	if (parallel) {
		tbb::parallel_for(blocked_range2d<uint64_t, uint64_t>(0, height, 0, width),
			[&](const blocked_range2d<uint64_t, uint64_t> &r) {

			RGBQUAD rgb;

			auto y1 = r.rows().begin();
			auto y2 = r.rows().end();
			auto x1 = r.cols().begin();
			auto x2 = r.cols().end();

			for (uint64_t y = y1; y != y2; y++) {
				for (uint64_t x = x1; x != x2; x++) {
					image.getPixelColor(x, y, &rgb);
					if (rgb.rgbRed > 0 || rgb.rgbGreen > 0 || rgb.rgbBlue > 0)
						image.setPixelColor(x, y, &white);
				}
			}
		});
	}
	else {
		RGBQUAD rgb;
		for (uint64_t y = 0; y < height; y++) {
			for (uint64_t x = 0; x < width; x++) {
				image.getPixelColor(x, y, &rgb);
				if (rgb.rgbRed != 0 || rgb.rgbGreen != 0 || rgb.rgbBlue != 0)
					image.setPixelColor(x, y, &white);
			}
		}
	}
}

int countNumberOfColouredPixels(RGBQUAD colour, fipImage image, bool parallel = true) {

	int height = image.getHeight();
	int width = image.getWidth();

	int count = 0;

	if (parallel) {
		count = parallel_reduce(
			blocked_range2d<uint64_t, uint64_t>(0, height, 0, width),
			0.0f,
			[&](const blocked_range2d<uint64_t, uint64_t> &r, int value)->int {

			RGBQUAD rgb;

			auto y1 = r.rows().begin();
			auto y2 = r.rows().end();
			auto x1 = r.cols().begin();
			auto x2 = r.cols().end();

			for (uint64_t y = y1; y != y2; y++) {
				for (uint64_t x = x1; x != x2; x++) {
					image.getPixelColor(x, y, &rgb);
					if (rgb.rgbRed == colour.rgbRed && rgb.rgbGreen == colour.rgbGreen && rgb.rgbBlue == colour.rgbBlue) 
						value++;
				}
			}
			return value;
		},
			[&](float x, float y)->int {
			return x + y;
		}
		);
		
	}
	else {
		for (uint64_t y = 0; y < height; y++) {
			for (uint64_t x = 0; x < width; x++) {
				RGBQUAD rgb;
				image.getPixelColor(x, y, &rgb);
				if ((rgb.rgbRed == colour.rgbRed && rgb.rgbGreen == colour.rgbGreen && rgb.rgbBlue == colour.rgbBlue))
					count++;
			}
		}
	}
	return count;
}

void changePixelColourAt(vector<uint64_t> pos, RGBQUAD colour, fipImage &image) {
	image.setPixelColor(pos[0], pos[1], &colour);
}

vector<uint64_t> findColouredPixel(RGBQUAD colour, fipImage image, bool parallel = true) {

	int height = image.getHeight();
	int width = image.getWidth();

	vector<uint64_t> pos;
	if (parallel) {
		tbb::parallel_for(
			blocked_range2d<uint64_t, uint64_t>(0, height, 0, width),
			[&](const blocked_range2d<uint64_t, uint64_t> &r) {
			auto y1 = r.rows().begin();
			auto y2 = r.rows().end();
			auto x1 = r.cols().begin();
			auto x2 = r.cols().end();

			for (uint64_t y = y1; y != y2; y++) {
				for (uint64_t x = x1; x != x2; x++) {

					RGBQUAD rgb;
					image.getPixelColor(x, y, &rgb);

					if (rgb.rgbRed == colour.rgbRed && rgb.rgbGreen == colour.rgbGreen && rgb.rgbBlue == colour.rgbBlue) {
						if (task::self().cancel_group_execution()) 
							pos = { x, y };
					}
				}
			}
		});
	}
	else {
		RGBQUAD rgb;
		for (uint64_t y = 0; y < height; y++) {
			for (uint64_t x = 0; x < width; x++) {
				image.getPixelColor(x, y, &rgb);
				if (rgb.rgbRed == colour.rgbRed && rgb.rgbGreen == colour.rgbGreen && rgb.rgbBlue == colour.rgbBlue)
					pos = { x, y };
			}
		}
	}

	return pos;
}


int main()
{
	int nt = task_scheduler_init::default_num_threads();
	task_scheduler_init T(nt);

	//Part 1 (Greyscale Gaussian blur): -----------DO NOT REMOVE THIS COMMENT----------------------------//

	grayscale_gaussian_blur();


	//Part 2 (Colour image processing): -----------DO NOT REMOVE THIS COMMENT----------------------------//

	
	// Setup Input image array
	fipImage inputImage1;
	inputImage1.load("../Images/render_1.png");
	fipImage inputImage2;
	inputImage2.load("../Images/render_2.png");

	unsigned int width = inputImage1.getWidth() > inputImage2.getWidth() ? inputImage1.getWidth() : inputImage2.getWidth();
	unsigned int height = inputImage1.getHeight() > inputImage2.getHeight() ? inputImage1.getHeight() : inputImage2.getHeight();

	// Setup Output image array
	fipImage outputImage;
	outputImage = fipImage(FIT_BITMAP, width, height, 24);


	int total_pixels = width * height;


	uint64_t count = 0;
	vector<uint64_t>randomPos;
	vector<uint64_t> pos;

	RGBQUAD rgb;  //FreeImage structure to hold RGB values of a single pixel

	// Test sequential vs. parallel_for versions - run test multiple times and track speed-up
	double meanSpeedup = 0.0f;
	uint64_t numTests = 50;




	/****************** SEQUENTIAL APPROACH ****************************/

	//save current time
	auto st1 = high_resolution_clock::now();

	//create output image using the absolute difference of 2 input images
	outputImage = computeAbsoluteDifference(inputImage1, inputImage2, false);

	//apply the binary threshold to the image created previously
	applyBinaryThreshold(outputImage, false);

	//count all the *white* pixels in the image
	count = countNumberOfColouredPixels(white, outputImage, false);

	//generate a random position using the width and height as limits
	randomPos = { rand() % width, rand() % height };
	std::cout << "Random Poisition:  x = " << randomPos[0] << "  y = " << randomPos[1] << std::endl;

	//change the pixel at the generated location with the desired colour
	changePixelColourAt(randomPos, red, outputImage);

	//find and return the first pixel with the desired colour
	pos = findColouredPixel(red, outputImage, false);
	std::cout << "Found red pixel at:  x = " << pos[0] << "  y = " << pos[1] << std::endl;



	//save current time and calulate the time it took to create the image using this approach
	auto st2 = high_resolution_clock::now();
	auto st_dur = duration_cast<microseconds>(st2 - st1);
	std::cout << "sequential negative operation took = " << st_dur.count() << "\n\n";

	//calculate the percentage of *white* pixels and output results
	float percentage = (float)count * 100 / total_pixels;
	std::cout << "Total Pixels: " << total_pixels << std::endl;
	std::cout << "White pixels: " << percentage << "%" << "  (" << count << ")" << std::endl;

	//Save the processed image
	std::cout << "saving image..." << std::endl;
	outputImage.save("Images/RGB_processed_sequence.png");
	std::cout << "-------------------------------------------------\n" << std::endl;




	/****************** PARALLEL APPROACH ****************************/

	//save current time
	auto pt1 = high_resolution_clock::now();

	//create output image using the absolute difference of 2 input images
	outputImage = computeAbsoluteDifference(inputImage1, inputImage2);

	//apply the binary threshold to the image created previously
	applyBinaryThreshold(outputImage);

	//count all the *white* pixels in the image
	count = countNumberOfColouredPixels(white, outputImage);

	//generate a random position using the width and height as limits
	randomPos = { rand() % width, rand() % height };
	std::cout << "Random Poisition:  x = " << randomPos[0] << "  y = " << randomPos[1] << std::endl;

	//change the pixel at the generated location with the desired colour
	changePixelColourAt(randomPos, red, outputImage);

	//find and return the first pixel with the desired colour
	pos = findColouredPixel(red, outputImage);
	std::cout << "Found red pixel at:  x = " << pos[0] << "  y = " << pos[1] << std::endl;



	//save current time and calulate the time it took to create the image using this approach
	auto pt2 = high_resolution_clock::now();
	auto pt_dur = duration_cast<microseconds>(pt2 - pt1);
	std::cout << "parallel negative operation took = " << pt_dur.count() << "\n\n";

	//calculate the percentage of *white* pixels and output results
	percentage = (float)count * 100 / total_pixels;
	std::cout << "Total Pixels: " << total_pixels << std::endl;
	std::cout << "White pixels: " << percentage << "%" << "  (" << count << ")" << std::endl;

	//Save the processed image
	std::cout << "saving image..." << std::endl;
	outputImage.save("Images/RGB_processed_parallel.png");
	std::cout << "-------------------------------------------------\n" << std::endl;





	double speedup = double(st_dur.count()) / double(pt_dur.count());
	std::cout << "speedup = " << speedup << endl;

	meanSpeedup += speedup;

	meanSpeedup /= double(numTests);
	std::cout << "Mean speedup = " << meanSpeedup << endl << endl;

	

	return 0;
	
}