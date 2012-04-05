#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

String cascadeName ="/home/project/Desktop/final/haar.xml";
String nestedCascadeName = "/home/project/Desktop/haarcascades/haarcascade_mcs_nose.xml";
//String cascadeName2 = "/home/project/Desktop/final/pen.xml";

CascadeClassifier cascade,nestedCascade;

int facecounter=0;

/*int g_switch_value = 0;
// This will be the callback that we give to the
// trackbar.
//


void switch_callback( int position ) 
{
	if( position != 0 ) 
	{
        cascade.load(cascadeName);	
	nestedCascade.load(nestedCascadeName);	
   	}
	else
	{
		cascade.load(cascadeName2);	
		nestedCascade.load(cascadeName2);
		

	}

}*/

void detectAndDraw( Mat& img,
                   CascadeClassifier& cascade, CascadeClassifier& nestedCascade,
                   double scale);



int main( int argc, const char** argv )
{
    CvCapture* capture = 0;
    Mat image;
double scale = 1;
    	
cascade.load(cascadeName);	
nestedCascade.load(nestedCascadeName);	
   		


    cvNamedWindow( "result", 1 );
    	
	   

	image = imread( argv[1], 1 );
	if( !image.empty() )
        {
		
		
            	detectAndDraw( image, cascade, nestedCascade, scale );
		cout<<endl<<" Number of face(s) detected :"<<facecounter+1<<endl;
		waitKey(0);	
        }
       
    

    cvDestroyWindow("result");

    return 0;
}

void detectAndDraw( Mat& img,
                   CascadeClassifier& cascade, CascadeClassifier& nestedCascade,
                   double scale)
{
    int i = 0;
    double t = 0;
    vector<Rect> faces;
    const static Scalar colors[] =  { CV_RGB(0,0,255),
        CV_RGB(0,128,255),
        CV_RGB(0,255,255),
        CV_RGB(0,255,0),
        CV_RGB(255,128,0),
        CV_RGB(255,255,0),
        CV_RGB(255,0,0),
        CV_RGB(255,0,255)} ;
    Mat gray, smallImg( cvRound (img.rows/scale), cvRound(img.cols/scale), CV_8UC1 );

    cvtColor( img, gray, CV_BGR2GRAY );
    resize( gray, smallImg, smallImg.size(), 0, 0, INTER_LINEAR );
    equalizeHist( smallImg, smallImg );

    t = (double)cvGetTickCount();
    cascade.detectMultiScale( smallImg, faces,
        1.1, 2, 0
        //|CV_HAAR_FIND_BIGGEST_OBJECT
        //|CV_HAAR_DO_ROUGH_SEARCH
        |CV_HAAR_SCALE_IMAGE
        ,
        Size(30, 30) );
    t = (double)cvGetTickCount() - t;
    printf( "detection time = %g ms\n", t/((double)cvGetTickFrequency()*1000.) );
    for( vector<Rect>::const_iterator r = faces.begin(); r != faces.end(); r++, i++ )
    {facecounter=i;

        Mat smallImgROI;
        vector<Rect> nestedObjects;
        Point center;
        Scalar color = colors[i%8];
        int radius;
        center.x = cvRound((r->x + r->width*0.5)*scale);
        center.y = cvRound((r->y + r->height*0.5)*scale);
        radius = cvRound((r->width + r->height)*0.25*scale);
        rectangle(img,Point(center.x+radius,center.y+radius),Point(center.x-radius,center.y-radius),color,1,8,0);//circle( img, center, radius, color, 3, 8, 0 );
        if( nestedCascade.empty() )
            continue;
        smallImgROI = smallImg(*r);
        nestedCascade.detectMultiScale( smallImgROI, nestedObjects,
            1.1, 2, 0
            //|CV_HAAR_FIND_BIGGEST_OBJECT
            //|CV_HAAR_DO_ROUGH_SEARCH
            //|CV_HAAR_DO_CANNY_PRUNING
            |CV_HAAR_SCALE_IMAGE
            ,
            Size(30, 30) );
        for( vector<Rect>::const_iterator nr = nestedObjects.begin(); nr != nestedObjects.end(); nr++ )
        {
            center.x = cvRound((r->x + nr->x + nr->width*0.5)*scale);
            center.y = cvRound((r->y + nr->y + nr->height*0.5)*scale);
            radius = cvRound((nr->width + nr->height)*0.25*scale);
	   // rectangle(img,Point(center.x+radius,center.y+radius),Point(center.x-radius,center.y-radius),color,1.5,8,0);
	    circle( img, center, radius, color, 2, 8, 0 );
        }
    }


    cv::imshow( "result", img );
}
