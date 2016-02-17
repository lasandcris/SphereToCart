#include <iostream>
#include <fstream>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/stitching.hpp"
#include "opencv2/stitching/detail/autocalib.hpp"
#include "opencv2/stitching/detail/blenders.hpp"
#include "opencv2/stitching/detail/timelapsers.hpp"
#include "opencv2/stitching/detail/camera.hpp"
#include "opencv2/stitching/detail/exposure_compensate.hpp"
#include "opencv2/stitching/detail/matchers.hpp"
#include "opencv2/stitching/detail/motion_estimators.hpp"
#include "opencv2/stitching/detail/seam_finders.hpp"
#include "opencv2/stitching/detail/util.hpp"
#include "opencv2/stitching/detail/warpers.hpp"
#include "opencv2/stitching/warpers.hpp"

using namespace std;
using namespace cv;
using namespace cv::detail;

bool try_use_gpu = false;
vector<Mat> imgs;
bool SphericalDistOnly = true;


void printUsage();
int parseCmdArgs(int argc, char** argv);

Mat map_x, map_y;
Mat imagesHere;
float PI = 3.14159265359;
Mat SphereUndistort;
Mat SphereUndistort2;
Mat croppedImage;
bool doCrop = false;

int ROIcrop = 3050;
int offsetX = -50;
int offsetY = -30;

//this can't by higher than ROIcrop
int FinalCrop = ROIcrop;

float _fov = 160.0;

/*In order to easily rotate an image in 3D space, this is a simple method that will do just that. It accepts rotations (in degrees) along each of the three axis (x, y and z), with 90 degrees being the "normal" position.
It also supports translations along each of the axis, and a variable focal distance (you would usually want the focal distance to be the same as your dz).

The parameters are:
input: the image that you want rotated.
output: the Mat object to put the resulting file in.
alpha: the rotation around the x axis
beta: the rotation around the y axis
gamma: the rotation around the z axis (basically a 2D rotation)
dx: translation along the x axis
dy: translation along the y axis
dz: translation along the z axis (distance to the image)
f: focal distance (distance between camera and image, a smaller number exaggerates the effect)

Example usage to rotate an image 45° around the y-axis:
rotateImage(orignalImage, outputImage, 90, 135, 90, 200, 200);
*/

void rotateImage(const Mat &input, Mat &output, double alpha, double beta, double gamma, double dx, double dy, double dz, double f)
{

	alpha = (alpha - 90.)*CV_PI / 180.;
	beta = (beta - 90.)*CV_PI / 180.;
	gamma = (gamma - 90.)*CV_PI / 180.;

	// get width and height for ease of use in matrices
	double w = (double)input.cols;
	double h = (double)input.rows;

	// Projection 2D -> 3D matrix
	Mat A1 = (Mat_<double>(4, 3) <<
		1, 0, -w / 2,
		0, 1, -h / 2,
		0, 0, 0,
		0, 0, 1);

	// Rotation matrices around the X, Y, and Z axis
	Mat RX = (Mat_<double>(4, 4) <<
		1, 0, 0, 0,
		0, cos(alpha), -sin(alpha), 0,
		0, sin(alpha), cos(alpha), 0,
		0, 0, 0, 1);

	Mat RY = (Mat_<double>(4, 4) <<
		cos(beta), 0, -sin(beta), 0,
		0, 1, 0, 0,
		sin(beta), 0, cos(beta), 0,
		0, 0, 0, 1);

	Mat RZ = (Mat_<double>(4, 4) <<
		cos(gamma), -sin(gamma), 0, 0,
		sin(gamma), cos(gamma), 0, 0,
		0, 0, 1, 0,
		0, 0, 0, 1);

	// Composed rotation matrix with (RX, RY, RZ)
	Mat R = RX * RY * RZ;

	// Translation matrix
	Mat T = (Mat_<double>(4, 4) <<
		1, 0, 0, dx,
		0, 1, 0, dy,
		0, 0, 1, dz,
		0, 0, 0, 1);

	// 3D -> 2D matrix
	Mat A2 = (Mat_<double>(3, 4) <<
		f, 0, w / 2, 0,
		0, f, h / 2, 0,
		0, 0, 1, 0);

	// Final transformation matrix
	Mat trans = A2 * (T * (R * A1));

	// Apply matrix transformation
	warpPerspective(input, output, trans, input.size(), INTER_LANCZOS4);
}

void buildMap(int Ws, int Hs, int Wd, int Hd, float hfovd = 160.0, float vfovd = 160.0)
{
	
	float vfov = (vfovd / 180.0)*PI;
	float hfov = (hfovd / 180.0)*PI;
	float vstart = ((180.0 - vfovd) / 180.00)*PI / 2.0;
	float hstart = ((180.0 - hfovd) / 180.00)*PI / 2.0;

	//need to scale to changed range from our
	// smaller cirlce traced by the fov
	
	float xmax = sin(PI / 2.0)*cos(vstart);
	float xmin = sin(PI / 2.0)*cos(vstart + vfov);
	float xscale = xmax - xmin;
	float xoff = xscale / 2.0;
	float zmax = cos(hstart);
	float zmin = cos(hfov + hstart);
	float zscale = zmax - zmin;
	float zoff = zscale / 2.0;

	//Fill in the map, this is slow but
	//we could probably speed it up
	//since we only calc it once, whatever

	for (int y = 0; y < Hd; y++)
	{
		for (int x = 0; x < Wd; x++)
		{
			float phi = vstart + (vfov*((float(x) / float(Wd))));
			float theta = hstart + (hfov*((float(y) / float(Hd))));
			float xp = ((sin(theta)*cos(phi)) + xoff) / zscale;
			float zp = ((cos(theta)) + zoff) / zscale;
			float xS = Wd - (xp*Wd);
			float yS = Hd - (zp*Hd);

			map_x.at<float>(y, x) = xS;
			map_y.at<float>(y, x) = yS;
		}
	}	
}


int main(int argc, char* argv[])
{
	int retval = parseCmdArgs(argc, argv);
	if (retval) return -1;

	

	for (int i = 0; i < imgs.size(); i++)
	{
		imagesHere = imgs[i];

		int NEW_X = (abs(imagesHere.cols - ROIcrop) / 2) + offsetX;
		int NEW_Y = (abs(imagesHere.rows - ROIcrop) / 2) + offsetY;

		int FinalCropXY = abs(ROIcrop - FinalCrop) / 2;
		
		//this is cam specific
		//Rect cropSize(100, 1050, 3444, 3444);
		Rect cropSize(NEW_X, NEW_Y, ROIcrop, ROIcrop);
		croppedImage = imagesHere(cropSize);

		if (i == 0)
		{
			map_x.create(croppedImage.size(), CV_32FC1);
			map_y.create(croppedImage.size(), CV_32FC1);

			int Ws = 0;
			int	Hs = 0;
			int Wd = croppedImage.cols;
			int	Hd = croppedImage.rows;

			buildMap(Ws, Hs, Wd, Hd);
		}

		//imwrite("Cropped.jpg", croppedImage);

		SphereUndistort = croppedImage.clone();
		remap(croppedImage, SphereUndistort, map_x, map_y, CV_INTER_LINEAR, BORDER_CONSTANT, Scalar(0, 0, 0));

		//cover the full angle 
		int xyCover = ((1.0 - (_fov / 180)) * ROIcrop)*2;
		Mat BackCanv = Mat(ROIcrop + xyCover, ROIcrop + xyCover, CV_8UC3, cv::Scalar(0, 0, 0));

		//Rect cropAgain(FinalCropXY, FinalCropXY, FinalCrop, FinalCrop);
		//SphereUndistort2 = SphereUndistort(cropAgain);

		//resize(SphereUndistort, SphereUndistort, Size(2973, 2973));

		for (int y = 0; y < SphereUndistort.cols; y++)
		{
			for (int x = 0; x < SphereUndistort.rows; x++)
			{
				BackCanv.at<Vec3b>(y + (xyCover / 2), x + (xyCover / 2))[2] = SphereUndistort.at<Vec3b>(y, x)[2];
				BackCanv.at<Vec3b>(y + (xyCover / 2), x + (xyCover / 2))[1] = SphereUndistort.at<Vec3b>(y, x)[1];
				BackCanv.at<Vec3b>(y + (xyCover / 2), x + (xyCover / 2))[0] = SphereUndistort.at<Vec3b>(y, x)[0];
			}
		}
	
		//imwrite("sphereBefore.jpg", SphereUndistort);

		std::string s = std::to_string(i);
		imwrite("SphericalDistort" + s + ".jpg", BackCanv);
	}


	//imwrite("Cropped.jpg", croppedImage);
	
	
	return 0;
}


void printUsage()
{
	cout <<
		"Distortion model.\n\n"
		"distorting img1 img2 [...imgN]\n\n"
		"Flags:\n"
		"  --try_use_gpu (yes|no)\n"
		"      Try to use GPU. The default value is 'no'. All default values\n"
		"      are for CPU mode.\n"
		"  --output <result_img>\n"
		"      The default is 'result.jpg'.\n";
}


int parseCmdArgs(int argc, char** argv)
{
	if (argc == 1)
	{
		printUsage();
		return -1;
	}
	for (int i = 1; i < argc; ++i)
	{


		Mat img = imread(argv[i]);

		imgs.push_back(img);
	}

	return 0;
}
