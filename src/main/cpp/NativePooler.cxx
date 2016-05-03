#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <jni.h>

#include <time.h>
#include <sys/time.h>

#include "NativePooler.h"

typedef struct Metas {
  int xDim;
  int yDim;
  int numChannels;
} Meta;

int pooler(double *output, double *image, int poolStride, int poolSize, double maxVal, double alpha, int **xPools, int **yPools, int *xps, int *yps, Meta meta) {
  int c = 0;
  int x = 0;
  int y = 0;
  int xp = 0;
  int yp = 0;
  int xPool = 0;
  int yPool  = 0;
  double pix = 0.0;
  double upval = 0.0;
  double downval = 0.0;

  while (c < meta.numChannels) {
    y = 0;
    while (y < meta.yDim) {
      x = 0;
      while(x < meta.xDim) {

        //Do symmetric rectification
        pix = image[x+y*meta.xDim+c*meta.xDim*meta.yDim];
        upval = fmax(maxVal, pix-alpha);
        downval = fmax(maxVal, -pix - alpha);

        //Put the pixel in all appropriate pools
        yp = 0;
        while (yp < yps[y]) {
          yPool = yPools[y][yp];

          xp = 0;
          while (xp < xps[x]) {
            xPool = xPools[x][xp];
            output[xPool+yPool*2+c*2*2] += upval;
            output[xPool+yPool*2+(c+meta.numChannels)*2*2] += downval;

            xp++;
          }
          yp++;
        }
        x++;
      }
      y++;
    }
    c++;
  }

  return 0;
}

int **getPoolAssignments(int strideStart, int dim, int poolStride) {
  int numPools = (dim - strideStart)/poolStride;

  int **res = (int **) malloc(sizeof(int *) * dim);


  for(int i = 0; i < dim; i++) {
    int *ptr;
    if (i == 9) {
      ptr = (int *) malloc(sizeof(int) * 2);
      ptr[0]=0;
      ptr[1]=1;
    } else {
      ptr = (int *) malloc(sizeof(int) * 1);
      if (i < 9) ptr[0] = 0; else ptr[0] = 1;
    }
    res[i]=ptr;
  }

  return (int **) res;
}


JNIEXPORT jdoubleArray JNICALL Java_utils_external_NativePooler_pool
  (JNIEnv * env, jobject obj, jint xDim, jint yDim, jint numChannels, jint stride, jint size, jdouble maxVal, jdouble alpha, jdoubleArray im) {

  Meta m;
  m.xDim = xDim;
  m.yDim = yDim;
  m.numChannels = numChannels;

  int numPoolsX = 2;
  int numPoolsY = 2;


  int **xPools = getPoolAssignments(size/2, m.xDim, stride);
  int **yPools = getPoolAssignments(size/2, m.yDim, stride);

  int xps[19] = {1,1,1,1,1,1,1,1,1,2,1,1,1,1,1,1,1,1,1};
  int yps[19] = {1,1,1,1,1,1,1,1,1,2,1,1,1,1,1,1,1,1,1};


  double *res= (double *) calloc(numPoolsX*numPoolsY*m.numChannels*2, sizeof(double));

  //Get no-copy pointer to im.
  jdouble * image = (jdouble *) env->GetPrimitiveArrayCritical(im, JNI_FALSE);

  //call pooler.
  pooler(res, image, stride, size, maxVal, alpha, xPools, yPools, xps, yps, m);

  //Release handle on image.
  env->ReleasePrimitiveArrayCritical(im, image, JNI_FALSE);

  int result_length = 2*2*numChannels*2;
  jdoubleArray result = env->NewDoubleArray(result_length);
  env->SetDoubleArrayRegion(result, 0, result_length, res);

  /*//Free things.
  for(int i = 0; i < 19; i++) {
    free(xPools[i]);
    free(yPools[i]);
  }

  //free(res);  */

  return result;
}
