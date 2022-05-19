#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

using namespace cv;
using namespace std;

///////////////  Document Scanner  //////////////////////

Mat imgOriginal, imgGray, imgBlur, imgCanny, imgThre, imgDil, imgErode, imgWarp, imgCrop;//матрицы с изображениями для преобразований
vector<Point> initialPoints, docPoints;//создаём вектора точек с начальным изображением и точек документа
float w = 420, h = 596;//размеры документа

Mat preProcessing(Mat img)//подготовка изображения
{
	if (!img.empty()) {
		cvtColor(img, imgGray, COLOR_BGR2GRAY);//конвертируем в серую картинку
		GaussianBlur(imgGray, imgBlur, Size(3, 3), 3, 0);//размываем картинку, для последующего использования Canny
		Canny(imgBlur, imgCanny, 25, 75);//отрисовываем изображение линиями 
		Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));//матрица kernel, для использования dilate
		dilate(imgCanny, imgDil, kernel);//расширим линии отрисованные на картинке
		return imgDil;//возвращаем полученное изображение
	}
}

vector<Point> getContours(Mat image) {//поиск контура документа

	vector<vector<Point>> contours;//матрица всей площади точек документа
	vector<Vec4i> hierarchy;

	findContours(image, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);//найдем контур и запишем точки по всей площади в матрицу
	vector<vector<Point>> conPoly(contours.size());
	vector<Rect> boundRect(contours.size());

	vector<Point> biggest;
	int maxArea = 0;

	for (int i = 0; i < contours.size(); i++)
	{
		int area = contourArea(contours[i]);//площадь контура

		string objectType;

		if (area > 1000)//если площадь достаточно большая
		{
			float peri = arcLength(contours[i], true);
			approxPolyDP(contours[i], conPoly[i], 0.02 * peri, true);

			if (area > maxArea&& conPoly[i].size() == 4) {

				biggest = { conPoly[i][0],conPoly[i][1] ,conPoly[i][2] ,conPoly[i][3] };
				maxArea = area;
			}
		}
	}
	return biggest;
}

vector<Point> reorder(vector<Point> points)//из точек контура, записать только 4, края документа
{
	vector<Point> newPoints;
	vector<int>  sumPoints, subPoints;

	for (int i = 0; i < 4; i++)
	{
		sumPoints.push_back(points[i].x + points[i].y);
		subPoints.push_back(points[i].x - points[i].y);
	}

	newPoints.push_back(points[min_element(sumPoints.begin(), sumPoints.end()) - sumPoints.begin()]); // 0
	newPoints.push_back(points[max_element(subPoints.begin(), subPoints.end()) - subPoints.begin()]); //1
	newPoints.push_back(points[min_element(subPoints.begin(), subPoints.end()) - subPoints.begin()]); //2
	newPoints.push_back(points[max_element(sumPoints.begin(), sumPoints.end()) - sumPoints.begin()]); //3

	return newPoints;
}

Mat getWarp(Mat img, vector<Point> points, float w, float h)//преобразование документа из перспективы
{
	Point2f src[4] = { points[0],points[1],points[2],points[3] };//начальные точки документа
	Point2f dst[4] = { {0.0f,0.0f},{w,0.0f},{0.0f,h},{w,h} };

	Mat matrix = getPerspectiveTransform(src, dst);
	warpPerspective(img, imgWarp, matrix, Point(w, h));

	return imgWarp;
}

void main() {

	string path = "paper.jpg";//путь до фото с документом
	imgOriginal = imread(path);//записываем в матрицу оригинальную картинку
	if (!imgOriginal.empty())
		resize(imgOriginal, imgOriginal, Size(), 0.5, 0.5);
	// Преобразование картинки - шаг 1 
	imgThre = preProcessing(imgOriginal);

	// Поиск контура документа  - шаг 2
	initialPoints = getContours(imgThre);
	docPoints = reorder(initialPoints);

	// Деформация - шаг 3 
	imgWarp = getWarp(imgOriginal, docPoints, w, h);

	//Обрезка - шаг 4
	int cropVal = 5;//для сглаживания краёв (чтобы небыло лишнего фона)
	Rect roi(cropVal, cropVal, w - (2 * cropVal), h - (2 * cropVal));//создаём прямоугольник для обрезки
	imgCrop = imgWarp(roi);//обрезаем

	imshow("Image", imgOriginal);//вывести оригинальную картинку
	imshow("Image Crop", imgCrop);//вывести обрезанный документ
	waitKey(0);

}