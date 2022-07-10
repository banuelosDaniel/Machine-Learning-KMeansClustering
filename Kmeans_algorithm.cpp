#include <string>
#include <fstream>
#include <vector>
#include <utility>
#include <stdexcept>
#include <sstream>
#include <iostream>
#include<time.h>
#include<set>
#include<math.h>
#include <omp.h>


using namespace std;


struct Point {
    double x, y;     // coordinates
    int cluster;     // no default cluster
    double minDist;  // default infinite dist to nearest cluster

    Point() :
        x(0.0),
        y(0.0),
        cluster(-1),
        minDist(__DBL_MAX__) {}

    Point(double x, double y) :
        x(x),
        y(y),
        cluster(-1),
        minDist(__DBL_MAX__) {}

    double distance(Point p) {
        return (p.x - x) * (p.x - x) + (p.y - y) * (p.y - y);
    }
};

//this is a modified function of a function I found online
//source website: https://www.gormanalysis.com/blog/reading-and-writing-csv-files-with-cpp/#reading-from-csv
vector<Point> readcsv(string filename){
    vector<Point> points;
    ifstream myFile(filename);
    if(!myFile.is_open()) throw runtime_error("Could not open file");
    string line, colname;
    string tmp;

    if(myFile.good()){
        getline(myFile, line);
    }

    while(getline(myFile, line)){
        stringstream ss(line);
        int colIdx = 0;
        string bit;
        double x, y;

        getline(ss, bit, ',');
        getline(ss, bit, ',');
        getline(ss, bit, ',');

        getline(ss, bit, ',');
        x = stof(bit);
        getline(ss, bit, ',');
        y= stof(bit);
        points.push_back(Point(x,y));

        getline(ss, bit, '\n');
    }

    myFile.close();
    return points;
}

double Euclidean(Point val1, Point cluster){
    return sqrt((cluster.x-val1.x)*(cluster.x-val1.x)+
    (cluster.y-val1.y)*(cluster.y-val1.y));
}

void computeCenter(vector<Point> *points, vector<Point> *clusters){
    double xmeanClust0 = 0.0, xmeanClust1 = 0.0, xmeanClust2 = 0.0, xmeanClust3 = 0.0, xmeanClust4 = 0.0;
    double ymeanClust0 = 0.0, ymeanClust1 = 0.0, ymeanClust2 = 0.0, ymeanClust3 = 0.0, ymeanClust4 = 0.0;
    int numClust0 = 0, numClust1 = 0, numClust2 = 0, numClust3 = 0, numClust4 = 0;
    for(int i = 0; i < points->size(); i++){
        if((*points)[i].cluster == 0){
            xmeanClust0 += (*points)[i].x;
            ymeanClust0 += (*points)[i].y;
            numClust0 ++;
        }
        if((*points)[i].cluster == 1){
            xmeanClust1 += (*points)[i].x;
            ymeanClust1 += (*points)[i].y;
            numClust1 ++;
        }
        if((*points)[i].cluster == 2){
            xmeanClust2 += (*points)[i].x;
            ymeanClust2 += (*points)[i].y;
            numClust2 ++;
        }
        if((*points)[i].cluster == 3){
            xmeanClust3 += (*points)[i].x;
            ymeanClust3 += (*points)[i].y;
            numClust3 ++;
        }
        if((*points)[i].cluster == 4){
            xmeanClust4 += (*points)[i].x;
            ymeanClust4 += (*points)[i].y;
            numClust4 ++;
        }
    }
    (*clusters)[0].x = xmeanClust0/numClust0;
    (*clusters)[0].x = ymeanClust0/numClust0;
    (*clusters)[0].cluster = 0;
    (*clusters)[1].x = xmeanClust1/numClust1;
    (*clusters)[1].y = ymeanClust1/numClust1;
    (*clusters)[1].cluster = 1;
    (*clusters)[2].x = xmeanClust2/numClust2;
    (*clusters)[2].y = ymeanClust2/numClust2;
    (*clusters)[2].cluster = 2;
    (*clusters)[3].x = xmeanClust3/numClust3;
    (*clusters)[3].y = ymeanClust3/numClust3;
    (*clusters)[3].cluster = 3;
    (*clusters)[4].x = xmeanClust4/numClust4;
    (*clusters)[4].y = ymeanClust4/numClust4;
    (*clusters)[4].cluster = 4;
}

void printCentroids(vector<Point> *cluster){
    for(int i = 0; i < cluster->size(); i++){
        cout << "{" << (*cluster)[i].x << ", " << (*cluster)[i].y << "}" << endl;
    }
}

void kMeansClustering(vector<Point> *points, vector<Point> *clusters, int epochs, int k){
    int counter = 0;
    while(counter<epochs){
        double temp = 100000;
        for(int i = 0; i < points->size(); i++){
            for (int j = 0; j < clusters->size(); j++){
                double temp_dist = Euclidean((*points)[i], (*clusters)[j]);
                if(temp_dist < temp){
                    temp = temp_dist;
                    (*points)[i].cluster = j;
                }
            }
            temp = 100000;
        }

        //all the points have been assigned a cluster with the current centroids
        //now recompute centroids and repeat for the set amount of epochs
        //We will now compute the center and reassign the values in cluster.
        //cout << "=================Epoch #" << counter << "===================\n";
        //printCentroids(clusters);
        computeCenter(points, clusters);
        counter++;
    }
}

void writecsv(string filename, vector<Point>* points){
  ofstream outputfile;
  outputfile.open(filename);
  outputfile << "x,y,c" << endl;

  for (vector<Point>::iterator it = points->begin(); it!= points->end(); it++){
    outputfile << it->x << ", " << it->y << ", " << it->cluster << endl;
  }
  outputfile.close();
}

int main() {
    int k = 5; //must stay hardcoded as 5
    int epochs = 20;//can change number of epochs
    int i;
    int tid;
    vector<Point> datapoints = readcsv("Mall_Customers.csv"); //storing data from file
    vector<Point> clusters; //declaring vector of Points to hold clusters.


    int max_range = datapoints.size(); //for random initialization of clusters.
    srand(time(NULL));
    //randomly select 5 points from datapoints and push it into clusters
    for (int i = 0; i < k; i++){
      int randomNum = rand() % max_range;
      clusters.push_back(datapoints[randomNum]);
    }
    int counter = 0;
    //set number of threads
    omp_set_num_threads(5);
    #pragma omp parallel private(i,tid)
    {
    /* Obtain thread number */
      tid = omp_get_thread_num();
      #pragma omp while
      while(counter<20){
        double temp = 100000;
        for(int i = 0; i < datapoints.size(); i++){
            for (int j = 0; j < clusters.size(); j++){
                double temp_dist = Euclidean(datapoints[i], clusters[j]);
                if(temp_dist < temp){
                    temp = temp_dist;
                    datapoints[i].cluster = j;
                }
            }
            temp = 100000;
        }

        //all the points have been assigned a cluster with the current centroids
        //now recompute centroids and repeat for the set amount of epochs
        //We will now compute the center and reassign the values in cluster.
        //cout << "=================Epoch #" << counter << "===================\n";
        //printCentroids(&clusters);
        computeCenter(&datapoints, &clusters);
        counter++;
     }

    }
    writecsv("processedData.csv", &datapoints);
}
