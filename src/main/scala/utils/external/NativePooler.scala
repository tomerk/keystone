package utils.external

class NativePooler extends Serializable {
  System.loadLibrary("ImageFeatures") // This will load libImageFeatures.{so,dylib} from the library path.

  /**
    * Performs sum pooling on an image represented in row, column, channel order (r+c*numRows+channel*numRows*numColumns).
    *
    * Attempts to do zero-copy reading of the input and returns a float vector returned according to
    * the same indexing scheme.
    *
    * @param imgWidth Image Width.
    * @param imgHeight Image Height.
    * @param imgChannels Step size at which to sample SIFT descriptors.
    * @param poolStride Stride between pools
    * @param poolSize Size of each pool.
    * @param maxVal Value to max with during rectification.
    * @param alpha Value to subtract during rectification.
    * @return Pooled image.
    */
  @native
  def pool(
                imgWidth: Int,
                imgHeight: Int,
                imgChannels: Int,
                poolStride: Int,
                poolSize: Int,
                maxVal: Double,
                alpha: Double,
                image: Array[Double]): Array[Double]

}
